"""Physical spectral indices for Sentinel-2-style optical stacks.

NDVI uses NIR (B8) and red (B4). NDWI (McFeeters) uses green (B3) and NIR (B8).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_ZERO_DIVISION_EPS = 1e-6


class SpectralIndexError(ValueError):
    """Raised when spectral index inputs are incompatible."""


def calculate_ndvi(b8_nir: np.ndarray, b4_red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index: (B8 - B4) / (B8 + B4).

    Zero-division and non-finite samples map to 0. Result is float32 in
    approximately ``[-1, 1]`` for physically valid reflectance.
    """
    nir, red = _as_pair(b8_nir, b4_red, "NDVI")
    ndvi = _safe_normalized_difference(nir - red, nir + red)
    logger.debug("calculate_ndvi mean=%.5f", float(np.mean(ndvi)))
    return ndvi


def calculate_ndwi(b3_green: np.ndarray, b8_nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters): (B3 - B8) / (B3 + B8)."""
    green, nir = _as_pair(b3_green, b8_nir, "NDWI")
    ndwi = _safe_normalized_difference(green - nir, green + nir)
    logger.debug("calculate_ndwi mean=%.5f", float(np.mean(ndwi)))
    return ndwi


def _as_pair(left: np.ndarray, right: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise SpectralIndexError(
            f"{label} band shapes must match, got {a.shape} and {b.shape}."
        )
    if a.size == 0:
        raise SpectralIndexError(f"{label} received an empty array.")
    return a, b


def _safe_normalized_difference(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Ratio with protection for near-zero denominators and NaN/Inf."""
    safe_denom = np.where(np.abs(denominator) < _ZERO_DIVISION_EPS, np.nan, denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / safe_denom
    finite = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(finite, -1.0, 1.0).astype(np.float32)
