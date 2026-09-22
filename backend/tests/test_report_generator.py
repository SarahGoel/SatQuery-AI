"""Audit report generator — JSON/PDF from traces and LangGraph state."""

from __future__ import annotations

from app.utils.report_generator import (
    build_audit_summary,
    extract_execution_trace,
    generate_audit_report,
)
from app.utils.trace_store import remember_trace
from backend.utils.report_generator import generate_audit_report as public_generate


def _sample_trace() -> dict:
    return {
        "trace_id": "ISRO-SQ-2026-TEST01",
        "task": "single_image_grounding",
        "query": "Where is the reservoir",
        "input_metadata": {
            "crs": "EPSG:4326",
            "bounds": [77.0, 28.0, 77.2, 28.2],
            "affine_transform": [1.0, 0.0, 77.0, 0.0, -1.0, 28.2],
            "modalities": ["RGB"],
        },
        "registry_execution": [
            {"model": "RS-Grounding-V3", "params": {"threshold": 0.75}},
            {"model": "mobilesam", "params": {"confidence": 0.81}},
        ],
        "confidence_score": 0.88,
        "output": "Reservoir highlighted along the northern bank.",
    }


def test_build_audit_summary_includes_task_models_params_confidence() -> None:
    summary = build_audit_summary(_sample_trace())
    assert summary["selected_task"] == "single_image_grounding"
    assert "RS-Grounding-V3" in summary["model_names"]
    assert "mobilesam" in summary["tool_names"]
    assert summary["key_parameters"]["RS-Grounding-V3"]["threshold"] == 0.75
    assert summary["confidence_score"] == 0.88
    assert summary["confidence_scores"]["overall"] == 0.88
    assert summary["confidence_scores"]["mobilesam"] == 0.81


def test_extract_from_langgraph_state_and_cache() -> None:
    trace = _sample_trace()
    from_state = extract_execution_trace(langgraph_state={"trace": trace})
    assert from_state["trace_id"] == "ISRO-SQ-2026-TEST01"

    remember_trace(trace["trace_id"], trace)
    from_cache = extract_execution_trace(trace_id=trace["trace_id"])
    assert from_cache["task"] == "single_image_grounding"


def test_generate_json_and_pdf_reports() -> None:
    json_body, json_type, json_name = generate_audit_report(
        langgraph_state={"trace": _sample_trace()},
        fmt="json",
    )
    assert json_type == "application/json"
    assert json_name.endswith(".json")
    assert b"selected_task" in json_body
    assert b"RS-Grounding-V3" in json_body

    pdf_body, pdf_type, pdf_name = public_generate(
        langgraph_state={"trace": _sample_trace()},
        fmt="pdf",
    )
    assert pdf_type == "application/pdf"
    assert pdf_name.endswith(".pdf")
    assert pdf_body.startswith(b"%PDF")
    assert b"SatQuery AI" in pdf_body
