"""Entrypoint re-export for app.main:app compatibility."""

from main import app, health_check

__all__ = ["app", "health_check"]
