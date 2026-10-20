"""Persistence helpers for inference audit history."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.prediction import PredictionRecord


def save_record(
    session: Session,
    *,
    operation: str,
    model_name: str,
    predicted_class: str,
    confidence: float,
    result: dict,
) -> PredictionRecord:
    """Persist result metadata while keeping the submitted image private."""
    record = PredictionRecord(
        operation=operation,
        model_name=model_name,
        predicted_class=predicted_class,
        confidence=confidence,
        result=result,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def list_records(session: Session, limit: int, offset: int) -> list[PredictionRecord]:
    """Return newest records first with caller-bounded pagination."""
    statement = (
        select(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(session.scalars(statement))


def delete_record(session: Session, record_id: int) -> bool:
    """Delete a record and report whether it existed."""
    record = session.get(PredictionRecord, record_id)
    if record is None:
        return False
    session.delete(record)
    session.commit()
    return True
