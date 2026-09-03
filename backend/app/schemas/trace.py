"""Strict Pydantic v2.6.1 models for auditable SatQuery execution traces."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class InputMetadataSchema(BaseModel):
    crs: str = Field(..., description="Coordinate Reference System of the input satellite imagery [68, 70]")
    bounds: List[float] = Field(..., description="Geospatial bounding box coordinates [left, bottom, right, top] [68, 70]")
    affine_transform: List[float] = Field(..., description="6-parameter affine transform matrix [68, 70]")
    modalities: List[str] = Field(..., description="Identified image bands/modalities (Optical/SAR) [68, 71]")


class RegistryExecutionSchema(BaseModel):
    model: str = Field(..., description="Name of the selected specialist model [68, 71]")
    params: Dict[str, Any] = Field(..., description="Runtime parameter configuration passed to the model [69, 71]")


class AuditableTraceLogSchema(BaseModel):
    trace_id: str = Field(..., description="Unique, trackable session identifier [69, 71]")
    task: str = Field(..., description="Classified task category [69, 71]")
    query: str = Field(..., description="Original user natural language query [69, 72]")
    input_metadata: InputMetadataSchema
    registry_execution: List[RegistryExecutionSchema]
    confidence_score: float = Field(..., description="Calculated confidence of the output [69, 72]")
    output: str = Field(..., description="Generated natural language response and spatial mappings [72, 73]")
