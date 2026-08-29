"""Feature 2.5 — temporal attention Change-VQA for bi-temporal T1 / T2 inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch import nn

from app.core.config import settings
from app.services.models.base import LocalVisionLanguageClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.enc = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.head = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        e1 = self.enc(t1)
        e2 = self.enc(t2)
        b, c, h, w = e1.shape
        seq1 = e1.flatten(2).transpose(1, 2)
        seq2 = e2.flatten(2).transpose(1, 2)
        fused, _ = self.attn(seq2, seq1, seq1)
        fused_map = fused.transpose(1, 2).view(b, c, h, w)
        delta = torch.abs(e2 - e1) + fused_map
        return torch.sigmoid(self.head(delta))


@dataclass
class ChangeVQAResult:
    answer: str
    change_mask: np.ndarray
    confidence: float
    overlay_uri: str | None
    params: dict[str, Any] = field(default_factory=dict)


class TemporalChangeVQA:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TemporalAttention().to(self.device)
        self.net.eval()
        self.vlm = LocalVisionLanguageClient()
        ckpt = settings.LOCAL_MODELS_DIR / "change_vqa" / "temporal_attn.pt"
        if settings.CHANGE_VQA_CHECKPOINT:
            ckpt = Path(settings.CHANGE_VQA_CHECKPOINT)
        if ckpt.exists():
            self.net.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
            logger.info("change_vqa_weights_loaded", extra={"path": str(ckpt)})

    def analyze(self, t1_path: Path, t2_path: Path, query: str) -> ChangeVQAResult:
        t1 = _preview_tensor(t1_path).to(self.device)
        t2 = _preview_tensor(t2_path).to(self.device)
        with torch.no_grad():
            mask_t = self.net(t1, t2)
        mask = mask_t.squeeze().detach().cpu().numpy()
        overlay = settings.ARTIFACT_DIR / "change_overlays" / f"{t1_path.stem}_vs_{t2_path.stem}.npy"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        np.save(overlay, mask)

        vlm = self.vlm.generate(
            prompt=(
                f"Bi-temporal EO change analysis. User question: {query}. "
                f"Estimated change fraction: {float((mask > 0.5).mean()):.3f}."
            ),
            image_path=t2_path,
            extra_context={"change_fraction": float((mask > 0.5).mean())},
        )
        return ChangeVQAResult(
            answer=vlm.text,
            change_mask=mask,
            confidence=float(np.clip((vlm.confidence + float(mask.mean())) / 2, 0.0, 1.0)),
            overlay_uri=str(overlay),
            params={
                "module": "TemporalAttention",
                "device": str(self.device),
                "change_fraction": float((mask > 0.5).mean()),
                "vlm": vlm.params,
            },
        )


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
    return torch.from_numpy(np.stack(bands, axis=0)).unsqueeze(0)
