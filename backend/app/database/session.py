"""SQLAlchemy spatial engine and session factory.

Uses GeoAlchemy2-compatible PostgreSQL (PostGIS). Sessions are request-scoped
via FastAPI dependencies; do not hold them across GPU inference calls.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import settings

Base = declarative_base()

try:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=8,
        max_overflow=16,
        future=True,
    )
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
except Exception:  # noqa: BLE001 — FastAPI must boot without a local DB driver
    engine = None
    SessionLocal = None


def get_db() -> Generator[Session, None, None]:
    if SessionLocal is None:
        raise RuntimeError("PostgreSQL driver/session is not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_optional_db() -> Generator[Session | None, None, None]:
    """Yield a live PostGIS session, or None when the database is unreachable."""
    if SessionLocal is None:
        yield None
        return
    db: Session | None = None
    try:
        db = SessionLocal()
        from sqlalchemy import text

        db.execute(text("SELECT 1"))
    except Exception:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
        yield None
        return
    try:
        yield db
    finally:
        if db is not None:
            db.close()
