"""Application services."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agent import SatQueryController


def __getattr__(name: str):
    if name == "SatQueryController":
        from app.services.agent import SatQueryController

        return SatQueryController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SatQueryController"]
