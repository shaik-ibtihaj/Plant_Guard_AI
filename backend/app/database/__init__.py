"""Database engine and session lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..core.config import Settings


class Base(DeclarativeBase):
    """Base class for persistent backend records."""


def _create_database_engine(database_url: str):
    connect_args = (
        {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    )
    if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
        database_file = Path(database_url.removeprefix("sqlite:///"))
        if not database_file.is_absolute():
            database_file = Path.cwd() / database_file
        database_file.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)


_settings = Settings()
engine = _create_database_engine(_settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def initialize_database() -> None:
    """Create tables for the initial API schema."""
    from ..models.prediction import PredictionRecord

    del PredictionRecord  # Import registers the table on Base.metadata.
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped SQLAlchemy session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
