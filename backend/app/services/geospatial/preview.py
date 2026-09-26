"""Web-Ready Raster Preview Generator for Satellite Earth Observation.

Extracts RGB visual representations from single-band and multi-band GeoTIFFs,
normalizes dynamic reflectance ranges to 8-bit uint8, and encodes map-ready
Base64 PNG data URIs alongside WGS84 Leaflet bounds for frontend imageOverlay rendering.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def generate_raster_preview(
    raster_path: Union[str, Path],
    max_dim: int = 1024,
) -> Tuple[Optional[str], Optional[List[List[float]]]]:
    """Generates a Base64-encoded PNG data URI and WGS84 Leaflet bounds ([[south, west], [north, east]]).

    Args:
        raster_path: Path to the GeoTIFF or image file.
        max_dim: Maximum width or height of the generated preview PNG.

    Returns:
        Tuple of (base64_data_uri, leaflet_bounds).
    """
    p = Path(raster_path)
    if not p.exists():
        logger.warning("Preview generation skipped: raster file not found at %s", p)
        return None, None

    rgb_array: Optional[np.ndarray] = None
    leaflet_bounds: Optional[List[List[float]]] = None

    # Strategy 1: Open via rasterio for georeferenced metadata and multi-band extraction
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(p) as src:
            # Extract WGS84 Leaflet bounds: [[min_lat, min_lon], [max_lat, max_lon]]
            try:
                crs = src.crs or "EPSG:4326"
                w, s, e, n = transform_bounds(
                    crs,
                    "EPSG:4326",
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                )
                leaflet_bounds = [[round(float(s), 6), round(float(w), 6)], [round(float(n), 6), round(float(e), 6)]]
            except Exception as bounds_err:
                logger.debug("Bounding box reprojection fallback for %s: %s", p, bounds_err)
                leaflet_bounds = [
                    [round(float(src.bounds.bottom), 6), round(float(src.bounds.left), 6)],
                    [round(float(src.bounds.top), 6), round(float(src.bounds.right), 6)],
                ]

            count = src.count
            if count == 1:
                band1 = src.read(1).astype(np.float32)
                norm = _normalize_band_u8(band1)
                rgb_array = np.stack([norm, norm, norm], axis=-1)
            elif count == 2:
                b1 = _normalize_band_u8(src.read(1).astype(np.float32))
                b2 = _normalize_band_u8(src.read(2).astype(np.float32))
                rgb_array = np.stack([b1, b2, b1], axis=-1)
            else:
                # Take first 3 bands (RGB)
                b1 = _normalize_band_u8(src.read(1).astype(np.float32))
                b2 = _normalize_band_u8(src.read(2).astype(np.float32))
                b3 = _normalize_band_u8(src.read(3).astype(np.float32))
                rgb_array = np.stack([b1, b2, b3], axis=-1)
    except Exception as rio_err:
        logger.debug("Rasterio reading deferred or failed for %s: %s; trying PIL fallback", p, rio_err)

    # Strategy 2: Fallback to PIL for standard PNG/JPEG/TIFF images
    if rgb_array is None:
        try:
            with Image.open(p) as img:
                rgb_img = img.convert("RGB")
                rgb_array = np.array(rgb_img, dtype=np.uint8)
                if leaflet_bounds is None:
                    leaflet_bounds = [[0.0, 0.0], [1.0, 1.0]]
        except Exception as pil_err:
            logger.warning("Failed to open raster for preview %s: %s", p, pil_err)
            return None, None

    if rgb_array is None:
        return None, None

    # Convert to PIL Image and resize if dimension exceeds max_dim
    try:
        pil_img = Image.fromarray(rgb_array)
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_str}"
        return data_uri, leaflet_bounds
    except Exception as enc_err:
        logger.warning("Failed to encode preview image %s: %s", p, enc_err)
        return None, leaflet_bounds


def _normalize_band_u8(band: np.ndarray) -> np.ndarray:
    """Normalizes single band array to 8-bit uint8 (0..255) using 2-98% percentile stretching."""
    valid = band[np.isfinite(band)]
    if len(valid) == 0:
        return np.zeros_like(band, dtype=np.uint8)

    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    if p98 > p2:
        stretched = np.clip((band - p2) / (p98 - p2), 0.0, 1.0) * 255.0
        return stretched.astype(np.uint8)

    b_min, b_max = valid.min(), valid.max()
    if b_max > b_min:
        stretched = np.clip((band - b_min) / (b_max - b_min), 0.0, 1.0) * 255.0
        return stretched.astype(np.uint8)

    if b_max > 0:
        return np.full_like(band, 128, dtype=np.uint8)
    return np.zeros_like(band, dtype=np.uint8)


def get_raster_base64_preview(raster_path: Union[str, Path], max_dim: int = 1024) -> Optional[str]:
    """Convenience helper returning just the base64 data URI."""
    uri, _ = generate_raster_preview(raster_path, max_dim=max_dim)
    return uri


def get_leaflet_bounds(raster_path: Union[str, Path]) -> Optional[List[List[float]]]:
    """Convenience helper returning just the WGS84 Leaflet bounds."""
    _, bounds = generate_raster_preview(raster_path)
    return bounds


def create_mask_overlay_data_uri(
    mask: Union[np.ndarray, List[Any]],
    color: Tuple[int, int, int, int] = (239, 68, 68, 200),
    max_dim: int = 1024,
) -> Optional[str]:
    """Converts a binary or probability mask into an RGBA PNG Base64 data URI."""
    if mask is None:
        return None
    try:
        arr = np.array(mask)
        if arr.ndim > 2:
            arr = arr.squeeze()
        if arr.ndim != 2:
            return None

        h, w = arr.shape
        if h == 0 or w == 0:
            return None

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_bool = arr > 0
        if mask_bool.any():
            rgba[mask_bool] = color

        pil_img = Image.fromarray(rgba, "RGBA")
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception as err:
        logger.warning("Failed to create mask overlay data URI: %s", err)
        return None


def ensure_data_uri(
    img_data: Union[str, Path, bytes, np.ndarray, None],
    mime: str = "image/png",
) -> Optional[str]:
    """Ensures an image string or data structure is formatted with the standard data URI prefix."""
    if not img_data:
        return None

    if isinstance(img_data, str):
        s = img_data.strip()
        if not s:
            return None
        if s.startswith("data:image/"):
            return s
        if s.startswith("http://") or s.startswith("https://") or s.startswith("/satellite/"):
            return s

        # Check if it's a file path on disk
        p = Path(s)
        if p.exists() and p.is_file():
            if p.suffix.lower() == ".npy":
                try:
                    arr = np.load(p)
                    return create_mask_overlay_data_uri(arr)
                except Exception:
                    return None
            uri, _ = generate_raster_preview(p)
            return uri

        # It's a raw base64 string without data URI scheme
        raw_b64 = "".join(s.split())
        return f"data:{mime};base64,{raw_b64}"

    if isinstance(img_data, Path):
        if img_data.exists():
            if img_data.suffix.lower() == ".npy":
                try:
                    arr = np.load(img_data)
                    return create_mask_overlay_data_uri(arr)
                except Exception:
                    return None
            uri, _ = generate_raster_preview(img_data)
            return uri
        return None

    if isinstance(img_data, np.ndarray):
        return create_mask_overlay_data_uri(img_data)

    if isinstance(img_data, bytes):
        b64_str = base64.b64encode(img_data).decode("utf-8")
        return f"data:{mime};base64,{b64_str}"

    return None
