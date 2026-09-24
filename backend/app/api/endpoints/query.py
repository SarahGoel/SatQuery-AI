"""POST /api/v1/query endpoint and router.

Accepts multipart imagery and natural language queries, running the agentic orchestrator.
"""

from __future__ import annotations

try:
    from api.routes import query_pipeline, router
except ImportError:
    from backend.api.routes import query_pipeline, router

__all__ = ["router", "query_pipeline"]
