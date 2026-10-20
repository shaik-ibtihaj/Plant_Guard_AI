"""Versioned prediction, explanation, severity, and history endpoints."""

from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ai.severity import SeverityAssessor, measure_affected_area

from ..database import get_db
from ..middleware.security import require_api_key
from ..schemas.prediction import (
    ExplanationResponse,
    HistoryResponse,
    PredictionResponse,
    SeverityResponse,
)
from ..services.history import delete_record, list_records, save_record
from ..services.inference import InferenceService, InvalidImageError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
secured = [Depends(require_api_key)]


def _service(request: Request) -> InferenceService:
    return request.app.state.inference


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    try:
        contents = await file.read(max_bytes + 1)
    finally:
        await file.close()
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the upload limit")
    return contents


def _decode_mask(contents: bytes, expected_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(contents)) as source:
            if source.width * source.height > InferenceService.MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413, detail="Mask dimensions are too large"
                )
            if source.size != expected_size:
                raise HTTPException(
                    status_code=422,
                    detail=f"Mask dimensions must match uploaded image: {expected_size}",
                )
            return np.asarray(source.convert("L"), dtype=np.uint8) > 0
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail="Mask file is not a valid image"
        ) from exc


@router.get("/health", tags=["Health"])
async def api_health(request: Request) -> dict:
    """Return model and database readiness for deployment probes."""
    service = _service(request)
    return {
        "status": (
            "ready"
            if service.ready and request.app.state.database_ready
            else "not_ready"
        ),
        "model": service.settings.model_name,
        "checkpoint_epoch": service.checkpoint_epoch,
        "database": "ready" if request.app.state.database_ready else "not_ready",
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=secured,
)
async def predict_image(
    request: Request,
    file: UploadFile = File(..., description="JPEG, PNG, or WEBP plant image"),
    top_k: int = Query(default=5, ge=1, le=10),
    session: Session = Depends(get_db),
) -> dict:
    """Classify a plant image and persist metadata without the uploaded bytes."""
    service = _service(request)
    if not service.ready:
        raise HTTPException(status_code=503, detail="Inference model is not ready")
    if not request.app.state.database_ready:
        raise HTTPException(status_code=503, detail="History database is not ready")
    contents = await _read_upload(file, service.settings.max_upload_bytes)
    try:
        result = await run_in_threadpool(service.predict_bytes, contents, top_k)
        record = save_record(
            session,
            operation="predict",
            model_name=result["model"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            result=result,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Failed to store prediction history")
        raise HTTPException(
            status_code=503, detail="History database unavailable"
        ) from exc
    except RuntimeError as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=503, detail="Prediction service unavailable"
        ) from exc
    return {**result, "record_id": record.id, "created_at": record.created_at}


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    dependencies=secured,
)
async def explain_image(
    request: Request,
    file: UploadFile = File(...),
    target_class: int | None = Query(default=None, ge=0),
    alpha: float = Query(default=0.4, ge=0, le=1),
    session: Session = Depends(get_db),
) -> dict:
    """Return a Grad-CAM or transformer token attribution PNG as base64."""
    service = _service(request)
    if not service.ready:
        raise HTTPException(status_code=503, detail="Inference model is not ready")
    if not request.app.state.database_ready:
        raise HTTPException(status_code=503, detail="History database is not ready")
    contents = await _read_upload(file, service.settings.max_upload_bytes)
    try:
        result = await run_in_threadpool(
            service.explain_bytes, contents, target_class, alpha
        )
        stored = {
            key: value
            for key, value in result.items()
            if key != "overlay_png_base64"
        }
        record = save_record(
            session,
            operation="explain",
            model_name=result["model"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            result=stored,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Failed to store explanation history")
        raise HTTPException(
            status_code=503, detail="History database unavailable"
        ) from exc
    return {**result, "record_id": record.id}


@router.post(
    "/severity",
    response_model=SeverityResponse,
    tags=["Severity"],
    dependencies=secured,
)
async def assess_severity(
    request: Request,
    image: UploadFile = File(...),
    leaf_mask: UploadFile = File(...),
    lesion_mask: UploadFile = File(...),
    session: Session = Depends(get_db),
) -> dict:
    """Assess lesion area within a supplied leaf mask and return guidance."""
    service = _service(request)
    if not service.ready:
        raise HTTPException(status_code=503, detail="Inference model is not ready")
    if not request.app.state.database_ready:
        raise HTTPException(status_code=503, detail="History database is not ready")
    max_bytes = service.settings.max_upload_bytes
    image_bytes = await _read_upload(image, max_bytes)
    leaf_bytes = await _read_upload(leaf_mask, max_bytes)
    lesion_bytes = await _read_upload(lesion_mask, max_bytes)
    try:
        prediction = await run_in_threadpool(service.predict_bytes, image_bytes, 1)
        with Image.open(io.BytesIO(image_bytes)) as source:
            expected_size = source.size
        leaf = _decode_mask(leaf_bytes, expected_size)
        lesion = _decode_mask(lesion_bytes, expected_size)
        measurement = measure_affected_area(leaf, lesion)
        details = SeverityAssessor().assess_with_details(
            prediction["confidence"], measurement["affected_area_ratio"]
        )
        result = {
            "predicted_class": prediction["predicted_class"],
            "confidence": prediction["confidence"],
            "severity": details["severity"],
            "affected_area_ratio": measurement["affected_area_ratio"],
            "leaf_pixels": measurement["leaf_pixels"],
            "affected_pixels": measurement["affected_pixels"],
            "recommendation": details["recommendation"],
        }
        record = save_record(
            session,
            operation="severity",
            model_name=prediction["model"],
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            result=result,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Failed to store severity history")
        raise HTTPException(
            status_code=503, detail="History database unavailable"
        ) from exc
    return {**result, "record_id": record.id}


@router.get(
    "/records",
    response_model=HistoryResponse,
    tags=["History"],
    dependencies=secured,
)
async def prediction_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> dict:
    """List recent prediction, explanation, and severity metadata."""
    if not request.app.state.database_ready:
        raise HTTPException(status_code=503, detail="History database is not ready")
    records = list_records(session, limit, offset)
    return {"items": records, "limit": limit, "offset": offset}


@router.delete(
    "/records/{record_id}",
    status_code=204,
    tags=["History"],
    dependencies=secured,
)
async def remove_history_record(
    record_id: int,
    session: Session = Depends(get_db),
) -> None:
    """Delete one stored analysis record."""
    if not delete_record(session, record_id):
        raise HTTPException(status_code=404, detail="History record not found")
