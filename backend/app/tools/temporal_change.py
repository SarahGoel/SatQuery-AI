"""RemoteCLIP & PyTorch Siamese Bi-Temporal Change Detection Tool.

Exposes SiameseChangeNet and TemporalChangeTool for bi-temporal satellite image differencing.
Re-exports from backend/app/services/analytical/temporal_change.py.
"""

from __future__ import annotations

from app.services.analytical.temporal_change import (
    RemoteCLIPTemporalEncoder,
    SiameseChangeNet,
    SiameseChangeResult,
    TemporalChangeTool,
)

__all__ = [
    "SiameseChangeNet",
    "SiameseChangeResult",
    "TemporalChangeTool",
    "RemoteCLIPTemporalEncoder",
]
