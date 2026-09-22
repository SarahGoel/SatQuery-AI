"""Agents and routing nodes for SatQuery AI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.router import (
    InputInspectorNode,
    STANDARDIZED_TASK_MAP,
    TASK_BITEMPORAL_CHANGE,
    TASK_CROSS_MODAL,
    TASK_SINGLE_GROUNDING,
    TASK_SINGLE_VQA,
)

if TYPE_CHECKING:
    from app.services.agent import SatQueryController


def __getattr__(name: str):
    if name == "SatQueryController":
        from app.services.agent import SatQueryController

        return SatQueryController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SatQueryController",
    "InputInspectorNode",
    "STANDARDIZED_TASK_MAP",
    "TASK_SINGLE_GROUNDING",
    "TASK_SINGLE_VQA",
    "TASK_BITEMPORAL_CHANGE",
    "TASK_CROSS_MODAL",
]
