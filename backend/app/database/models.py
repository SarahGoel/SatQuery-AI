"""Declarative SQLAlchemy models matching init_db.sql (PostGIS Geometry)."""

from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class AuditableExecutionTrace(Base):
    __tablename__ = "auditable_execution_traces"

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    crs: Mapped[str | None] = mapped_column(String(128), nullable=True)
    affine_transform_matrix: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    bounding_box_geometry = mapped_column(Geometry("POLYGON", srid=4326), nullable=True)
    overall_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    model_executions: Mapped[list["TraceModelExecution"]] = relationship(
        "TraceModelExecution",
        back_populates="trace",
        cascade="all, delete-orphan",
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    local_weights_path: Mapped[str] = mapped_column(Text, nullable=False)

    executions: Mapped[list["TraceModelExecution"]] = relationship(
        "TraceModelExecution",
        back_populates="model",
    )


class TraceModelExecution(Base):
    __tablename__ = "trace_model_executions"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    trace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("auditable_execution_traces.trace_id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("model_registry.model_name"),
        nullable=False,
    )
    parameter_configuration: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    execution_order: Mapped[int] = mapped_column(Integer, nullable=False)

    trace: Mapped[AuditableExecutionTrace] = relationship(
        "AuditableExecutionTrace",
        back_populates="model_executions",
    )
    model: Mapped[ModelRegistry] = relationship(
        "ModelRegistry",
        back_populates="executions",
    )
