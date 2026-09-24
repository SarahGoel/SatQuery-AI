"""Multi-band spectral indices (NDVI / NDWI) stacked onto visual tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import torch

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_spectral_indices(
    red_band: np.ndarray, green_band: np.ndarray, nir_band: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates NDVI and NDWI physical indices from raw bands as floating-point tensors [46-48].

    NDVI = (B8 - B4) / (B8 + B4)  — vegetation (NIR, Red)
    NDWI = (B3 - B8) / (B3 + B8)  — water (Green, NIR)
    """
    # Convert inputs to float32 to bypass arithmetic truncation [46, 47]
    red = red_band.astype(np.float32)
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)

    # Calculate NDVI: (NIR - Red) / (NIR + Red) [46-48]
    ndvi_denominator = nir + red
    # Avoid zero-division cases [46, 47]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(ndvi_denominator == 0, 0, (nir - red) / ndvi_denominator)

    # Calculate NDWI: (Green - NIR) / (Green + NIR) [46-48]
    ndwi_denominator = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(ndwi_denominator == 0, 0, (green - nir) / ndwi_denominator)

    return ndvi.astype(np.float32), ndwi.astype(np.float32)


def generate_n_channel_tensor(
    rgb_array: np.ndarray, ndvi: np.ndarray, ndwi: np.ndarray
) -> torch.Tensor:
    """
    Stacks index maps onto the visual input tensor, expanding standard 3-channel RGB to N-channel inputs [46, 47, 49].
    """
    # RGB tensor shape (3, H, W)
    rgb = np.asarray(rgb_array)
    if rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.shape[0] != 3:
        rgb = np.transpose(rgb, (2, 0, 1))
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).float()
    if rgb_tensor.ndim != 3:
        raise ValueError(f"rgb_array must be (3, H, W), got {tuple(rgb_tensor.shape)}")
    if rgb_tensor.shape[0] != 3:
        raise ValueError(f"rgb_array must have 3 channels first, got {tuple(rgb_tensor.shape)}")

    ndvi_tensor = torch.from_numpy(np.asarray(ndvi, dtype=np.float32)).unsqueeze(0).float()
    ndwi_tensor = torch.from_numpy(np.asarray(ndwi, dtype=np.float32)).unsqueeze(0).float()

    # Concatenate features along the channel axis to create a 5-channel tensor [46, 47]
    n_channel_tensor = torch.cat([rgb_tensor, ndvi_tensor, ndwi_tensor], dim=0)
    return n_channel_tensor


@dataclass
class SpectralPack:
    feature_tensor: torch.Tensor
    ndvi: np.ndarray
    ndwi: np.ndarray
    ndvi_mean: float
    ndwi_mean: float
    band_count: int


class SpectralExtractor:
    """Read GeoTIFF bands, compute NDVI/NDWI, stack as 5-channel (RGB+indices) tensor."""

    def extract(self, path: Path) -> SpectralPack:
        with rasterio.open(path) as src:
            count = src.count
            data = src.read().astype(np.float32)
        red = self._band(data, settings.SPECTRAL_RED_BAND_INDEX, fallback=1)
        nir = self._band(data, settings.SPECTRAL_NIR_BAND_INDEX, fallback=min(count, 4))
        green = self._band(data, settings.SPECTRAL_GREEN_BAND_INDEX, fallback=min(count, 2))

        ndvi, ndwi = calculate_spectral_indices(red, green, nir)
        rgb = self._rgb_stack(data)
        tensor_5ch = generate_n_channel_tensor(rgb, ndvi, ndwi)
        tensor = tensor_5ch.unsqueeze(0)  # (1, 5, H, W) for CNN/ViT batching

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
    def _rgb_stack(data: np.ndarray) -> np.ndarray:
        if data.shape[0] >= 3:
            return data[:3]
        return np.repeat(data[:1], 3, axis=0)

    @staticmethod
    def _band(data: np.ndarray, one_based: int, fallback: int) -> np.ndarray:
        idx = one_based - 1
        if 0 <= idx < data.shape[0]:
            return data[idx]
        return data[max(0, min(fallback - 1, data.shape[0] - 1))]
