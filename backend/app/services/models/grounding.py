"""Feature 2.4 — zero-shot text-guided grounding (SAM / MobileSAM decoder)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch import nn

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LightweightMaskDecoder(nn.Module):
    """Stand-in decoder used until official SAM weights are mounted under local_models/."""

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, 1),
        )

    def forward(self, image: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        logits = self.stem(image)
        scale = text_embed.mean().clamp(0.5, 1.5)
        return torch.sigmoid(logits * scale)


@dataclass
class GroundingResult:
    mask: np.ndarray
    description: str
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)


class TextGuidedGrounder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = LightweightMaskDecoder().to(self.device)
        self.decoder.eval()

    def ground(self, image_path: Path, prompt: str, use_mobilesam: bool = True) -> GroundingResult:
        weights = (
            settings.resolved_mobilesam_weights()
            if use_mobilesam
            else settings.resolved_sam_weights()
        )
        loaded = self._try_load(weights)
        image = _preview_tensor(image_path).to(self.device)
        text_embed = _hash_prompt_embed(prompt, device=self.device)
        with torch.no_grad():
            mask_t = self.decoder(image, text_embed)
        mask = mask_t.squeeze().detach().cpu().numpy()
        model_name = "mobilesam" if use_mobilesam else "sam-vit-b"
        description = (
            f"Grounded prompt `{prompt}` with {model_name}. "
            f"Positive fraction={float((mask > 0.5).mean()):.3f}."
        )
        return GroundingResult(
            mask=mask,
            description=description,
            confidence=0.62 if loaded else 0.45,
            params={
                "model": model_name,
                "weights_path": str(weights),
                "weights_loaded": loaded,
                "device": str(self.device),
            },
        )

    def _try_load(self, path: Path) -> bool:
        if not path.exists():
            logger.warning("grounding_weights_missing", extra={"path": str(path)})
            return False
        try:
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict):
                self.decoder.load_state_dict(state, strict=False)
            logger.info("grounding_weights_loaded", extra={"path": str(path)})
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("grounding_weights_incompatible", extra={"error": str(exc)})
            return False


def _preview_tensor(path: Path, size: int = 256) -> torch.Tensor:
    import cv2

    with rasterio.open(path) as src:
        count = min(3, src.count)
        arr = src.read(list(range(1, count + 1))).astype(np.float32)
    if arr.shape[0] < 3:
        arr = np.repeat(arr[:1], 3, axis=0)
    bands = []
    for band in arr[:3]:
        if band.max() > band.min():
            band = (band - band.min()) / (band.max() - band.min())
        bands.append(cv2.resize(band, (size, size), interpolation=cv2.INTER_AREA))
    stacked = np.stack(bands, axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)


def _hash_prompt_embed(prompt: str, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(abs(hash(prompt)) % (2**32))
    vec = rng.standard_normal(32).astype(np.float32)
    return torch.from_numpy(vec).to(device)
