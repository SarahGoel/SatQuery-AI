"""Strict Pydantic v2 models for auditable SatQuery execution traces."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputMetadataSchema(BaseModel):
    """GeoTIFF-derived spatial metadata recorded on every trace."""

    model_config = ConfigDict(extra="forbid")

    crs: str
    bounds: List[float] = Field(..., min_length=4, max_length=4)
    affine_transform: List[float] = Field(..., min_length=6, max_length=9)
    modalities: List[str] = Field(default_factory=list)

    @field_validator("bounds")
    @classmethod
    def _validate_bounds(cls, value: List[float]) -> List[float]:
        minx, miny, maxx, maxy = value
        if minx >= maxx or miny >= maxy:
            raise ValueError("bounds must be [minx, miny, maxx, maxy] with positive area")
        return value


class RegistryExecutionSchema(BaseModel):
    """One model invocation recorded against model_registry."""

    model_config = ConfigDict(extra="forbid")

    model: str
    params: Dict[str, Any] = Field(default_factory=dict)


class AuditableTraceLogSchema(BaseModel):
    """Canonical response payload persisted to auditable_execution_traces."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    task: str
    query: str
    input_metadata: InputMetadataSchema
    registry_execution: List[RegistryExecutionSchema]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    output: str
