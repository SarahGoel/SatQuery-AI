"""Feature 2.2 — multi-band extractor (NDVI / NDWI) stacked onto PyTorch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_index(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(denominator) < 1e-6, np.nan, denominator)
    result = numerator / denom
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass
class SpectralPack:
    feature_tensor: torch.Tensor
    ndvi: np.ndarray
    ndwi: np.ndarray
    ndvi_mean: float
    ndwi_mean: float
    band_count: int


class SpectralExtractor:
    """Read GeoTIFF bands, compute vegetation/water indices, stack as NCHW tensor."""

    def extract(self, path: Path) -> SpectralPack:
        with rasterio.open(path) as src:
            count = src.count
            data = src.read().astype(np.float32)
        red = self._band(data, settings.SPECTRAL_RED_BAND_INDEX, fallback=1)
        nir = self._band(data, settings.SPECTRAL_NIR_BAND_INDEX, fallback=min(count, 4))
        green = self._band(data, settings.SPECTRAL_GREEN_BAND_INDEX, fallback=min(count, 2))
        swir = self._band(data, settings.SPECTRAL_SWIR_BAND_INDEX, fallback=nir)

        ndvi = _safe_index(nir - red, nir + red)
        ndwi = _safe_index(green - nir, green + nir)
        # McFeeters NDWI uses green/NIR; keep SWIR ratio as extra channel when present.
        mndwi = _safe_index(green - swir, green + swir)

        stack = np.concatenate(
            [data, ndvi[np.newaxis, ...], ndwi[np.newaxis, ...], mndwi[np.newaxis, ...]],
            axis=0,
        )
        tensor = torch.from_numpy(stack).unsqueeze(0)  # (1, C, H, W)
        pack = SpectralPack(
            feature_tensor=tensor,
            ndvi=ndvi,
            ndwi=ndwi,
            ndvi_mean=float(np.mean(ndvi)),
            ndwi_mean=float(np.mean(ndwi)),
            band_count=count,
        )
        logger.info(
            "spectral_extracted",
            extra={"path": str(path), "shape": list(tensor.shape), "ndvi_mean": pack.ndvi_mean},
        )
        return pack

    @staticmethod
    def _band(data: np.ndarray, one_based: int, fallback: int) -> np.ndarray:
        idx = one_based - 1
        if 0 <= idx < data.shape[0]:
            return data[idx]
        return data[max(0, min(fallback - 1, data.shape[0] - 1))]
