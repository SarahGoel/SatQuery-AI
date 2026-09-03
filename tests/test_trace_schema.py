"""Phase 5 Step 10 — strict Pydantic auditable trace schemas."""

from __future__ import annotations

from app.schemas.trace import AuditableTraceLogSchema


def test_auditable_trace_log_schema_constructs() -> None:
    data = {
        "trace_id": "TEST-123",
        "task": "single_image_vqa",
        "query": "Is there vegetation?",
        "input_metadata": {
            "crs": "EPSG:4326",
            "bounds": [0.0, 0.0, 1.0, 1.0],
            "affine_transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "modalities": ["Optical"],
        },
        "registry_execution": [{"model": "VQA-Model", "params": {}}],
        "confidence_score": 0.95,
        "output": "Yes, dense vegetation is present.",
    }
    log = AuditableTraceLogSchema(**data)
    assert log.trace_id == "TEST-123"
    assert log.task == "single_image_vqa"
    assert log.input_metadata.crs == "EPSG:4326"
    assert log.registry_execution[0].model == "VQA-Model"
    assert log.confidence_score == 0.95
