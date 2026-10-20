"""Persist prediction and analysis metadata without storing uploaded images."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class PredictionRecord(Base):
    """An inference, explanation, or severity assessment audit record."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    operation: Mapped[str] = mapped_column(String(24), index=True)
    model_name: Mapped[str] = mapped_column(String(80))
    predicted_class: Mapped[str] = mapped_column(String(200))
    confidence: Mapped[float] = mapped_column(Float)
    result: Mapped[dict] = mapped_column(JSON)
