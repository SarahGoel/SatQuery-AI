"""Entrypoint re-export for app.main:app compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from main import app, health_check
except ImportError:
    from backend.main import app, health_check

__all__ = ["app", "health_check"]
