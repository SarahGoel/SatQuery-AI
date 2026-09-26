"""POST /api/v1/query endpoint and router.

Accepts multipart imagery and natural language queries, running the agentic orchestrator.
STRICT LIVE EXECUTION: Every query bypasses pre-computed artifact reports in
backend/artifacts/reports/ and executes live PyTorch and VLM models directly.
"""

from __future__ import annotations

try:
    from api.routes import query_pipeline, router, sanitize_for_json
except ImportError:
    from backend.api.routes import query_pipeline, router, sanitize_for_json

from app.services.geospatial.preview import (
    generate_raster_preview,
    ensure_data_uri,
    create_mask_overlay_data_uri,
)

__all__ = [
    "router",
    "query_pipeline",
    "generate_raster_preview",
    "ensure_data_uri",
    "create_mask_overlay_data_uri",
    "sanitize_for_json",
]
