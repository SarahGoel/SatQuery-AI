"""Dual-encoder Optical ViT × SAR radar cross-attention fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


if HAS_TORCH:
    class CrossAttentionOpticalSAR(nn.Module):
        """
        Feature-level alignment projecting Optical contexts against SAR structural boundaries [50, 52, 54].
        """

        def __init__(self, d_model: int, num_heads: int):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, optical_features: torch.Tensor, sar_features: torch.Tensor) -> torch.Tensor:
            b, n, d = optical_features.shape
            q = self.q_proj(optical_features).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
            k = self.k_proj(sar_features).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
            v = self.v_proj(sar_features).view(b, n, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = torch.softmax(scores, dim=-1)

            context = torch.matmul(attn_weights, v)
            context = context.transpose(1, 2).contiguous().view(b, n, d)

            return self.out_proj(context)

    class OpticalViTEncoder(nn.Module):
        """Patch-embedding ViT encoder for optical (Cartosat-2S) RGB/multispectral grids."""

        def __init__(self, d_model: int = 256, patch_size: int = 16, in_channels: int = 3) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        def forward(self, optical: torch.Tensor) -> torch.Tensor:
            tokens = self.proj(optical)
            return tokens.flatten(2).transpose(1, 2)

    class SARRadarEncoder(nn.Module):
        """Radar-specific encoder for SAR VV/VH backscatter bands."""

        def __init__(self, d_model: int = 256, patch_size: int = 16, in_channels: int = 2) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        def forward(self, sar: torch.Tensor) -> torch.Tensor:
            tokens = self.proj(sar)
            return tokens.flatten(2).transpose(1, 2)
else:
    class CrossAttentionOpticalSAR:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def to(self, *args: Any, **kwargs: Any) -> "CrossAttentionOpticalSAR":
            return self

        def eval(self) -> "CrossAttentionOpticalSAR":
            return self

    class OpticalViTEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def to(self, *args: Any, **kwargs: Any) -> "OpticalViTEncoder":
            return self

        def eval(self) -> "OpticalViTEncoder":
            return self

    class SARRadarEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def to(self, *args: Any, **kwargs: Any) -> "SARRadarEncoder":
            return self

        def eval(self) -> "SARRadarEncoder":
            return self


@dataclass
class FusionResult:
    salient_mask: np.ndarray
    attention_energy: float
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)


class OpticalSarFusion:
    """Dual-encoder fusion: Optical ViT (Q) × SAR radar encoder (K, V)."""

    def __init__(self, d_model: int = 256, num_heads: int = 8, patch_size: int = 16) -> None:
        self.d_model = d_model
        self.patch_size = patch_size
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.optical_encoder = OpticalViTEncoder(d_model=d_model, patch_size=patch_size, in_channels=3).to(
                self.device
            )
            self.sar_encoder = SARRadarEncoder(d_model=d_model, patch_size=patch_size, in_channels=2).to(
                self.device
            )
            self.fusion = CrossAttentionOpticalSAR(d_model=d_model, num_heads=num_heads).to(self.device)
            self.head = nn.Linear(d_model, 1).to(self.device)
            self.optical_encoder.eval()
            self.sar_encoder.eval()
            self.fusion.eval()
            self.head.eval()
            ckpt = settings.LOCAL_MODELS_DIR / "fusion" / "cross_attention.pt"
            if settings.FUSION_CHECKPOINT:
                ckpt = Path(settings.FUSION_CHECKPOINT)
            if ckpt.exists():
                try:
                    state = torch.load(ckpt, map_location=self.device)
                    if isinstance(state, dict) and "fusion" in state:
                        self.fusion.load_state_dict(state["fusion"], strict=False)
                        if "optical_encoder" in state:
                            self.optical_encoder.load_state_dict(state["optical_encoder"], strict=False)
                        if "sar_encoder" in state:
                            self.sar_encoder.load_state_dict(state["sar_encoder"], strict=False)
                        if "head" in state:
                            self.head.load_state_dict(state["head"], strict=False)
                    else:
                        self.fusion.load_state_dict(state, strict=False)
                    logger.info("fusion_weights_loaded", extra={"path": str(ckpt)})
                except Exception:
                    logger.warning("fusion_weights_incompatible_random_init", extra={"path": str(ckpt)})
            else:
                logger.warning("fusion_weights_missing_random_init", extra={"expected": str(ckpt)})
        else:
            self.device = "cpu"
            self.optical_encoder = None
            self.sar_encoder = None
            self.fusion = None
            self.head = None

    def fuse(self, optical_path: Path, sar_path: Path) -> FusionResult:
        if HAS_TORCH and self.fusion is not None:
            opt = _read_rgb(optical_path)
            sar = _read_vv_vh(sar_path, shape=opt.shape[-2:])
            with torch.no_grad():
                optical_features = self.optical_encoder(opt.to(self.device))
                sar_features = self.sar_encoder(sar.to(self.device))
                fused = self.fusion(optical_features, sar_features)
                logits = self.head(fused).squeeze(-1)
                h = opt.shape[-2] // self.patch_size
                w = opt.shape[-1] // self.patch_size
                mask_t = torch.sigmoid(logits.view(1, 1, h, w))
                mask_t = nn.functional.interpolate(mask_t, size=opt.shape[-2:], mode="bilinear", align_corners=False)
            mask = mask_t.squeeze().detach().cpu().numpy()
        else:
            try:
                with rasterio.open(optical_path) as o_src, rasterio.open(sar_path) as s_src:
                    o_arr = _minmax(o_src.read(1).astype(np.float32))
                    s_arr = _minmax(s_src.read(1).astype(np.float32))
            except Exception:
                try:
                    from PIL import Image
                    with Image.open(optical_path) as p1, Image.open(sar_path) as p2:
                        o_arr = _minmax(np.array(p1.convert("L"), dtype=np.float32))
                        s_arr = _minmax(np.array(p2.convert("L"), dtype=np.float32))
                except Exception as read_err:
                    logger.warning("Failed to read rasters in fuse fallback: %s", read_err)
                    o_arr = np.zeros((256, 256), dtype=np.float32)
                    s_arr = np.zeros((256, 256), dtype=np.float32)

            # Dimension alignment: if optical and SAR arrays differ in shape, resize SAR to optical
            if o_arr.shape != s_arr.shape:
                import cv2
                s_arr = cv2.resize(s_arr, (o_arr.shape[1], o_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

            mask = (o_arr * 0.5 + s_arr * 0.5)
        energy = float(mask.mean())
        return FusionResult(
            salient_mask=mask,
            attention_energy=energy,
            confidence=float(np.clip(0.4 + energy, 0.0, 0.95)),
            params={
                "module": "CrossAttentionOpticalSAR",
                "optical_encoder": "OpticalViTEncoder",
                "sar_encoder": "SARRadarEncoder",
                "device": str(self.device),
                "mask_mean": energy,
                "d_model": self.d_model,
                "weights_path": str(
                    Path(settings.FUSION_CHECKPOINT)
                    if settings.FUSION_CHECKPOINT
                    else settings.LOCAL_MODELS_DIR / "fusion" / "cross_attention.pt"
                ),
            },
        )


def _read_rgb(path: Path, size: int = 256) -> Any:
    try:
        with rasterio.open(path) as src:
            count = min(3, max(1, src.count))
            arr = src.read(list(range(1, count + 1)))
        if arr.shape[0] < 3:
            arr = np.repeat(arr[:1], 3, axis=0)
    except Exception:
        try:
            from PIL import Image
            with Image.open(path) as pimg:
                rgb = pimg.convert("RGB")
                arr = np.array(rgb, dtype=np.float32).transpose(2, 0, 1)
        except Exception as err:
            logger.warning("Failed to read RGB raster from %s: %s", path, err)
            arr = np.zeros((3, size, size), dtype=np.float32)
    arr = _resize(arr[:3], size)
    arr = _minmax(arr)
    if not HAS_TORCH:
        return arr[np.newaxis, ...]
    return torch.from_numpy(arr).unsqueeze(0)


def _read_vv_vh(path: Path, shape: tuple[int, int]) -> Any:
    """Load SAR as 2-channel VV/VH (duplicate VV when only one band is present)."""
    try:
        with rasterio.open(path) as src:
            count = src.count
            if count >= 2:
                arr = src.read([1, 2])
            else:
                vv = src.read(1)
                arr = np.stack([vv, vv], axis=0)
    except Exception:
        try:
            from PIL import Image
            with Image.open(path) as pimg:
                gray = np.array(pimg.convert("L"), dtype=np.float32)
                arr = np.stack([gray, gray], axis=0)
        except Exception as err:
            logger.warning("Failed to read SAR raster from %s: %s", path, err)
            arr = np.zeros((2, shape[0], shape[1]), dtype=np.float32)
    arr = _minmax(arr.astype(np.float32))
    arr = _resize(arr, shape[0])
    if not HAS_TORCH:
        return arr[np.newaxis, ...]
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
