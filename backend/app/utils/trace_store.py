"""In-memory execution-trace cache when PostGIS is unavailable."""

from __future__ import annotations

from typing import Any

_TRACE_CACHE: dict[str, dict[str, Any]] = {}


def remember_trace(trace_id: str, payload: dict[str, Any]) -> None:
    if not trace_id:
        return
    _TRACE_CACHE[str(trace_id)] = payload


def recall_trace(trace_id: str) -> dict[str, Any] | None:
    return _TRACE_CACHE.get(str(trace_id))
