"""Strict Pydantic v2.6.1 models for auditable SatQuery execution traces."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class InputMetadataSchema(BaseModel):
    crs: str = Field(..., description="Coordinate Reference System of the input satellite imagery [68, 70]")
    bounds: List[float] = Field(..., description="Geospatial bounding box coordinates [left, bottom, right, top] [68, 70]")
    affine_transform: List[float] = Field(..., description="6-parameter affine transform matrix [68, 70]")
    modalities: List[str] = Field(..., description="Identified image bands/modalities (Optical/SAR) [68, 71]")
    sensor: Optional[str] = Field(default=None, description="Identified sensor platform (e.g. Cartosat-2S, Sentinel-1)")
    resolution: Optional[str] = Field(default=None, description="Spatial resolution")
    band_count: Optional[int] = Field(default=None, description="Number of spectral/radar bands")


class RegistryExecutionSchema(BaseModel):
    model: str = Field(..., description="Name of the selected specialist model [68, 71]")
    params: Dict[str, Any] = Field(default_factory=dict, description="Runtime parameter configuration passed to the model [69, 71]")


class AuditableTraceLogSchema(BaseModel):
    trace_id: str = Field(..., description="Unique, trackable session identifier [69, 71]")
    task: str = Field(..., description="Classified task category [69, 71]")
    task_type: Optional[str] = Field(default=None, description="Standardized task type [single_grounding, single_vqa, bitemporal_change, cross_modal]")
    query: str = Field(..., description="Original user natural language query [69, 72]")
    input_metadata: InputMetadataSchema
    registry_execution: List[RegistryExecutionSchema] = Field(default_factory=list, description="List of executed specialist model configurations")
    tools_executed: List[RegistryExecutionSchema] = Field(default_factory=list, description="List of executed specialist tool configurations")
    models_executed: List[str] = Field(default_factory=list, description="List of model identifiers executed")
    confidence_score: float = Field(..., description="Calculated confidence of the output [69, 72]")
    confidence: Optional[float] = Field(default=None, description="Standard numeric confidence score")
    output: str = Field(..., description="Generated natural language response and spatial mappings [72, 73]")
    geojson: Optional[Dict[str, Any]] = Field(default=None, description="Standard FeatureCollection of discrete instances")
    intent_classification: Optional[Dict[str, Any]] = Field(default=None, description="Semantic intent router classification decision")
    geospatial_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Calculated surface area and geospatial bounding metrics")
    scratchpad: Optional[Dict[str, Any]] = Field(default=None, description="Agent intermediate scratchpad log")

    @model_validator(mode="before")
    @classmethod
    def sync_tools_and_registry(cls, data: Any) -> Any:
        if isinstance(data, dict):
            reg = data.get("registry_execution")
            tools = data.get("tools_executed")
            if tools is not None and not reg:
                data["registry_execution"] = tools
            elif reg is not None and not tools:
                data["tools_executed"] = reg
            elif tools is None and reg is None:
                data["registry_execution"] = []
                data["tools_executed"] = []
        return data
