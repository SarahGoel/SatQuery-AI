"""API models for incoming multipart analysis payloads.

FastAPI binds UploadFile fields separately; these models validate the
non-file form fields after the controller has parsed GeoTIFFs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    BI_TEMPORAL_CHANGE_ANALYSIS = "bi_temporal_change_analysis"
    SINGLE_IMAGE_GROUNDING = "single_image_grounding"
    CROSS_MODAL_JOINT_ANALYSIS = "cross_modal_joint_analysis"
    SINGLE_IMAGE_VQA = "single_image_vqa"
    SINGLE_GROUNDING = "single_grounding"
    SINGLE_VQA = "single_vqa"
    BITEMPORAL_CHANGE = "bitemporal_change"
    CROSS_MODAL = "cross_modal"
    DOMAIN_KNOWLEDGE_QA = "domain_knowledge_qa"


class AnalyzeFormFields(BaseModel):
    """Non-file portion of POST /api/v1/satquery/analyze."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=3, max_length=4000)
    modality_optical: str = Field(default="cartosat-2s")
    modality_sar: Optional[str] = Field(default=None)
    force_task: Optional[TaskType] = None
    use_mobilesam: bool = True


class AnalyzeResponseEnvelope(BaseModel):
    """HTTP wrapper around the auditable trace plus optional GeoJSON."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    task_type: Optional[str] = None
    models_executed: list[str] = Field(default_factory=list)
    input_metadata: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    geojson: Optional[dict] = None
    change_overlay_uri: Optional[str] = None
    trace: dict


class QueryResponseEnvelope(BaseModel):
    """POST /api/v1/query — answer, OpenLayers geometry, and audit summary."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    answer: str
    task_type: Optional[str] = None
    headline: Optional[str] = None
    models_executed: list[str] = Field(default_factory=list)
    input_metadata: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    geojson: Optional[dict] = None
    bbox: Optional[list[float]] = None
    change_mask: Optional[dict] = None
    change_overlay_uri: Optional[str] = None
    audit_summary: dict
    trace: dict
    report: dict = Field(default_factory=dict)
