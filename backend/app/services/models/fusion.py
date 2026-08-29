"""Feature 2.3 — multi-sensor cross-attention (Cartosat-2S optical × RISAT SAR)."""

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


class CrossAttentionFusion(nn.Module):
    """Lightweight optical-as-query / SAR-as-key-value fusion head."""

    def __init__(self, dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.opt_proj = nn.Conv2d(3, dim, kernel_size=1)
        self.sar_proj = nn.Conv2d(1, dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.out = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        # optical: (B, 3, H, W)  sar: (B, 1, H, W)
        q = self.opt_proj(optical)
        k = self.sar_proj(sar)
        b, c, h, w = q.shape
        q_seq = q.flatten(2).transpose(1, 2)
        k_seq = k.flatten(2).transpose(1, 2)
        fused, _ = self.attn(q_seq, k_seq, k_seq)
        fused_map = fused.transpose(1, 2).view(b, c, h, w)
        return torch.sigmoid(self.out(fused_map))


@dataclass
class FusionResult:
    salient_mask: np.ndarray
    attention_energy: float
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)


class OpticalSarFusion:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CrossAttentionFusion().to(self.device)
        self.net.eval()
        ckpt = settings.LOCAL_MODELS_DIR / "fusion" / "cross_attention.pt"
        if settings.FUSION_CHECKPOINT:
            ckpt = Path(settings.FUSION_CHECKPOINT)
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            self.net.load_state_dict(state, strict=False)
            logger.info("fusion_weights_loaded", extra={"path": str(ckpt)})
        else:
            logger.warning("fusion_weights_missing_random_init", extra={"expected": str(ckpt)})

    def fuse(self, optical_path: Path, sar_path: Path) -> FusionResult:
        opt = _read_rgb(optical_path)
        sar = _read_single(sar_path, shape=opt.shape[-2:])
        with torch.no_grad():
            mask_t = self.net(opt.to(self.device), sar.to(self.device))
        mask = mask_t.squeeze().detach().cpu().numpy()
        energy = float(mask.mean())
        return FusionResult(
            salient_mask=mask,
            attention_energy=energy,
            confidence=float(np.clip(0.4 + energy, 0.0, 0.95)),
            params={
                "module": "CrossAttentionFusion",
                "device": str(self.device),
                "mask_mean": energy,
            },
        )


def _read_rgb(path: Path, size: int = 256) -> torch.Tensor:
    with rasterio.open(path) as src:
        count = min(3, src.count)
        arr = src.read(list(range(1, count + 1)))
    if arr.shape[0] < 3:
        arr = np.repeat(arr[:1], 3, axis=0)
    arr = _resize(arr[:3], size)
    arr = _minmax(arr)
    return torch.from_numpy(arr).unsqueeze(0)


def _read_single(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    with rasterio.open(path) as src:
        arr = src.read(1)
    arr = _minmax(arr[np.newaxis, ...])
    arr = _resize(arr, shape[0])
    return torch.from_numpy(arr).unsqueeze(0)


def _minmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    import cv2

    channels = []
    for band in arr:
        channels.append(cv2.resize(band, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(channels, axis=0).astype(np.float32)
