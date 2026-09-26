"""API models for incoming multipart analysis payloads.

FastAPI binds UploadFile fields separately; these models validate the
non-file form fields after the controller has parsed GeoTIFFs.
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy arrays, scalars, tensors, and non-serializable objects into native Python types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int) and not isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not isinstance(obj, (np.floating,)):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (str, bytes)):
        return str(obj) if isinstance(obj, str) else obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)

    # NumPy ndarray -> python lists
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(item) for item in obj.tolist()]

    # NumPy scalar types -> native float / int / bool
    if isinstance(obj, (np.floating,)):
        val = float(obj.item())
        return 0.0 if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.integer,)):
        return int(obj.item())
    if isinstance(obj, (np.bool_,)):
        return bool(obj.item())

    # PyTorch Tensor
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            return sanitize_for_json(obj.detach().cpu().numpy())
        except Exception:
            pass

    # Generic converters
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return sanitize_for_json(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return sanitize_for_json(obj.item())
        except Exception:
            pass

    # Pydantic models
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return sanitize_for_json(obj.model_dump())
        except Exception:
            pass
    elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return sanitize_for_json(obj.dict())
        except Exception:
            pass

    # Nested structures
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]

    return obj


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

    @model_validator(mode="before")
    @classmethod
    def sanitize_inputs(cls, data: Any) -> Any:
        return sanitize_for_json(data)

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        raw = {k: getattr(self, k) for k in self.model_fields}
        return sanitize_for_json(raw)


class QueryResponseEnvelope(BaseModel):
    """POST /api/v1/query — answer, OpenLayers geometry, and audit summary."""

    model_config = ConfigDict(extra="ignore")

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
    t1_preview_url: Optional[str] = None
    t2_preview_url: Optional[str] = None
    original_image: Optional[str] = None
    baseline_image: Optional[str] = None
    base_image: Optional[str] = None
    image_t0: Optional[str] = None
    preview_url: Optional[str] = None
    overlay_image: Optional[str] = None
    evidence_image: Optional[str] = None
    visual_evidence: Optional[str] = None
    evidence_type: Optional[str] = None
    detected_task: Optional[str] = None
    leaflet_bounds: Optional[list[list[float]]] = None
    audit_summary: dict
    trace: dict
    report: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def sanitize_inputs(cls, data: Any) -> Any:
        return sanitize_for_json(data)

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        raw = {k: getattr(self, k) for k in self.model_fields}
        return sanitize_for_json(raw)

