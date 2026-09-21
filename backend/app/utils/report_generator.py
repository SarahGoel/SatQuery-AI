"""Build downloadable JSON/PDF audit reports from PostGIS or LangGraph state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.schemas.trace import AuditableTraceLogSchema
from app.utils.trace_store import recall_trace, remember_trace


def build_audit_summary(
    trace: AuditableTraceLogSchema | dict[str, Any],
    *,
    geojson: dict[str, Any] | None = None,
    change_overlay_uri: str | None = None,
) -> dict[str, Any]:
    """Compact audit payload: task, tools, parameters, and confidence scores."""
    data = _as_trace_dict(trace)
    steps = list(data.get("tools_executed") or data.get("registry_execution") or [])
    models = [str(step.get("model")) for step in steps if step.get("model")]
    key_parameters = {
        str(step.get("model")): dict(step.get("params") or {})
        for step in steps
        if step.get("model")
    }
    confidence_scores: dict[str, float] = {
        "overall": float(data.get("confidence_score") or 0.0),
    }
    for step in steps:
        name = str(step.get("model") or "unknown")
        params = dict(step.get("params") or {})
        if "confidence" in params:
            try:
                confidence_scores[name] = float(params["confidence"])
            except (TypeError, ValueError):
                continue
    metadata = dict(data.get("input_metadata") or {})
    std_task = data.get("task_type") or data.get("task")
    return {
        "trace_id": data.get("trace_id"),
        "task_type": std_task,
        "selected_task": data.get("task"),
        "query": data.get("query"),
        "models_executed": data.get("models_executed") or models,
        "model_names": models,
        "tool_names": models,
        "key_parameters": key_parameters,
        "confidence": confidence_scores["overall"],
        "confidence_score": confidence_scores["overall"],
        "confidence_scores": confidence_scores,
        "input_metadata": metadata,
        "crs": metadata.get("crs"),
        "bounds": metadata.get("bounds"),
        "sensor": metadata.get("sensor"),
        "resolution": metadata.get("resolution"),
        "band_count": metadata.get("band_count"),
        "modalities": metadata.get("modalities") or [],
        "output": data.get("output"),
        "geojson": geojson,
        "geojson_feature_count": _feature_count(geojson),
        "change_overlay_uri": change_overlay_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_execution_trace(
    *,
    db: Any | None = None,
    trace_id: str | None = None,
    langgraph_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an auditable trace from LangGraph state, PostGIS, or the local cache."""
    if langgraph_state:
        nested = langgraph_state.get("trace")
        if isinstance(nested, AuditableTraceLogSchema):
            return nested.model_dump()
        if isinstance(nested, dict) and nested.get("trace_id"):
            return nested
        if langgraph_state.get("trace_id"):
            return dict(langgraph_state)

    if trace_id:
        cached = recall_trace(trace_id)
        if cached:
            return cached

    if db is not None and trace_id:
        from_db = _load_trace_from_postgis(db, trace_id)
        if from_db:
            remember_trace(trace_id, from_db)
            return from_db

    raise KeyError(f"Execution trace not found: {trace_id or '<missing id>'}")


def generate_audit_report(
    *,
    db: Any | None = None,
    trace_id: str | None = None,
    langgraph_state: dict[str, Any] | None = None,
    geojson: dict[str, Any] | None = None,
    change_overlay_uri: str | None = None,
    fmt: str = "json",
) -> tuple[bytes, str, str]:
    """Return (body, media_type, filename) for a JSON or PDF audit report."""
    trace = extract_execution_trace(
        db=db,
        trace_id=trace_id,
        langgraph_state=langgraph_state,
    )
    if geojson is None:
        geojson = trace.get("geojson") if isinstance(trace, dict) else None
    if change_overlay_uri is None and isinstance(trace, dict):
        change_overlay_uri = trace.get("change_overlay_uri")
    summary = build_audit_summary(
        trace,
        geojson=geojson,
        change_overlay_uri=change_overlay_uri,
    )
    payload = {
        "audit_summary": summary,
        "trace": trace,
        "geojson": geojson,
    }
    report_id = str(summary.get("trace_id") or trace_id or "anonymous")
    kind = (fmt or "json").lower().strip()
    if kind == "pdf":
        body = render_audit_pdf(payload)
        return body, "application/pdf", f"satquery-{report_id}.pdf"
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    return body, "application/json", f"satquery-{report_id}.json"


def render_audit_pdf(payload: dict[str, Any]) -> bytes:
    summary = dict(payload.get("audit_summary") or {})
    lines = [
        "SatQuery AI — Auditable Execution Report",
        "Problem statement: SIH26167",
        "",
        f"Trace ID: {summary.get('trace_id')}",
        f"Selected task: {summary.get('selected_task')}",
        f"Overall confidence: {summary.get('confidence_score')}",
        f"CRS: {summary.get('crs')}",
        f"Bounds: {summary.get('bounds')}",
        f"Generated: {summary.get('generated_at')}",
        "",
        "Query:",
        str(summary.get("query") or ""),
        "",
        "Textual answer:",
        str(summary.get("output") or ""),
        "",
        "Models / tools:",
    ]
    for name in summary.get("model_names") or []:
        params = (summary.get("key_parameters") or {}).get(name, {})
        conf = (summary.get("confidence_scores") or {}).get(name)
        extra = f"  confidence={conf}" if conf is not None else ""
        lines.append(f"- {name}{extra}")
        if params:
            lines.append(f"    params: {json.dumps(params, default=str)}")
    if payload.get("geojson"):
        lines.append("")
        lines.append(f"GeoJSON features: {_feature_count(payload.get('geojson'))}")
    return _simple_pdf(lines)


def _as_trace_dict(trace: AuditableTraceLogSchema | dict[str, Any]) -> dict[str, Any]:
    if isinstance(trace, AuditableTraceLogSchema):
        return trace.model_dump()
    return dict(trace)


def _feature_count(geojson: dict[str, Any] | None) -> int:
    if not geojson:
        return 0
    if geojson.get("type") == "FeatureCollection":
        return len(geojson.get("features") or [])
    if geojson.get("type") == "Feature":
        return 1
    return 0


def _load_trace_from_postgis(db: Any, trace_id: str) -> dict[str, Any] | None:
    from app.database.models import AuditableExecutionTrace, TraceModelExecution

    record = (
        db.query(AuditableExecutionTrace)
        .filter(AuditableExecutionTrace.trace_id == trace_id)
        .one_or_none()
    )
    if record is None:
        return None
    steps = (
        db.query(TraceModelExecution)
        .filter(TraceModelExecution.trace_id == trace_id)
        .order_by(TraceModelExecution.execution_order.asc())
        .all()
    )
    bounds = _bounds_from_geometry(record.bounding_box_geometry)
    return {
        "trace_id": record.trace_id,
        "task": record.task_type,
        "query": record.user_query,
        "input_metadata": {
            "crs": record.crs,
            "bounds": bounds,
            "affine_transform": list(record.affine_transform_matrix or []),
            "modalities": [],
        },
        "registry_execution": [
            {
                "model": step.model_name,
                "params": dict(step.parameter_configuration or {}),
            }
            for step in steps
        ],
        "confidence_score": float(record.overall_confidence),
        "output": record.final_output,
    }


def _bounds_from_geometry(geom: Any) -> list[float]:
    try:
        from geoalchemy2.shape import to_shape

        shape = to_shape(geom)
        b = shape.bounds if shape and hasattr(shape, "bounds") and shape.bounds else [0.0, 0.0, 0.0, 0.0]
        if len(b) < 4:
            b = [0.0, 0.0, 0.0, 0.0]
        minx, miny, maxx, maxy = b[:4]
        return [float(minx), float(miny), float(maxx), float(maxy)]
    except Exception:  # noqa: BLE001
        return []


def _simple_pdf(lines: list[str]) -> bytes:
    """Minimal PDF 1.4 writer (Helvetica) — no extra reporting dependency."""
    wrapped: list[str] = []
    for line in lines:
        text = (line or "").replace("\t", "    ")
        while len(text) > 92:
            wrapped.append(text[:92])
            text = text[92:]
        wrapped.append(text)

    y = 770
    commands = ["BT", "/F1 11 Tf"]
    first = True
    for line in wrapped:
        if y < 48:
            break
        escaped = _pdf_escape(line)
        if first:
            commands.append(f"1 0 0 1 48 {y} Tm ({escaped}) Tj")
            first = False
        else:
            commands.append("0 -14 Td")
            commands.append(f"({escaped}) Tj")
        y -= 14
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


def _pdf_escape(text: str) -> str:
    latin = text.encode("latin-1", errors="replace").decode("latin-1")
    return latin.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
