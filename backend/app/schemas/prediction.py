"""Prediction request and response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionCandidate(BaseModel):
    class_name: str
    class_index: int
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    record_id: int | None = None
    created_at: datetime | None = None
    model: str
    checkpoint_epoch: int | None = None
    predicted_class: str
    class_index: int
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[PredictionCandidate]
    image_width: int
    image_height: int
    inference_time_ms: float = Field(ge=0)


class HistoryRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    operation: str
    model_name: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    result: dict


class HistoryResponse(BaseModel):
    items: list[HistoryRecord]
    limit: int
    offset: int


class ExplanationResponse(BaseModel):
    record_id: int
    model: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    explained_class: str
    explained_index: int
    image_width: int
    image_height: int
    overlay_width: int
    overlay_height: int
    overlay_png_base64: str
    inference_time_ms: float = Field(ge=0)


class SeverityResponse(BaseModel):
    record_id: int
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    severity: str
    affected_area_ratio: float = Field(ge=0, le=1)
    leaf_pixels: int
    affected_pixels: int
    recommendation: str
