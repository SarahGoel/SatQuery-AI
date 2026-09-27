"""GeoChat-RS compatibility module delegating to RemoteSensingVLMClient and LocalVisionLanguageClient."""

from __future__ import annotations

from app.services.models.base import LocalVisionLanguageClient, VLMResult
from app.services.models.rs_vlm import (
    GEOCHAT_SYSTEM_PROMPT,
    TEMPORAL_VLM_SYSTEM_PROMPT,
    RemoteSensingVLMClient,
)

__all__ = [
    "RemoteSensingVLMClient",
    "LocalVisionLanguageClient",
    "VLMResult",
    "GEOCHAT_SYSTEM_PROMPT",
    "TEMPORAL_VLM_SYSTEM_PROMPT",
]
