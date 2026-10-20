"""FastAPI application entry point."""

from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .api.routes import router
from .core.config import Settings
from .database import initialize_database
from .middleware.rate_limit import RateLimitMiddleware
from .services.inference import InferenceService

logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize the model once and keep liveness available on load failure."""
    application.state.settings = settings
    application.state.database_ready = False
    try:
        initialize_database()
        application.state.database_ready = True
    except Exception:
        logger.exception("Plant Guard history database failed to initialize")
    service = InferenceService(settings)
    application.state.inference = service
    try:
        service.load()
    except Exception as exc:
        service.fail(exc)
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Plant Guard image classification API. Model predictions are advisory "
        "and depend on the selected checkpoint."
    ),
    version=settings.version,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit_per_minute,
)
app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Liveness probe; reports that the web process is responding."""
    return {"status": "ok", "service": settings.app_name, "version": settings.version}


@app.get("/health/ready", tags=["Health"])
async def readiness_check(request: Request) -> dict[str, str | int | None]:
    """Readiness probe; returns 503 until a valid model is loaded."""
    service: InferenceService = request.app.state.inference
    if not service.ready or not request.app.state.database_ready:
        raise HTTPException(status_code=503, detail="API dependencies are not ready")
    return {
        "status": "ready",
        "model": service.settings.model_name,
        "checkpoint_epoch": service.checkpoint_epoch,
    }


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """API welcome response."""
    return {"message": "Welcome to Plant Guard AI. Visit /docs for API documentation."}
