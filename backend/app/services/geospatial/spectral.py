"""Unified Spectral Indices Engine & Dynamic Sensor Band Profiles for Earth Observation.

Provides:
1. Dynamic Sensor Band Profile Detection (Sentinel-2, Landsat-8/9, Cartosat-2S, generic 4-band/3-band rasters).
2. Core Spectral Index Calculations: NDVI, NDWI, MNDWI, and NDBI with division-by-zero protection.
3. False-Color / Composite Visualizations: Color-Infrared (CIR), calibrated SAR dB conversion,
   and colormap PNG rendering.
4. Backwards-compatible interfaces for SpectralExtractor and legacy tensor pipelines.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from app.core.config import settings

logger = logging.getLogger("SpectralIndicesEngine")


# ---------------------------------------------------------------------------
# 1. Sensor Band Profiles & Mappings
# ---------------------------------------------------------------------------

@dataclass
class BandMapping:
    """Zero-based channel indices mapped to standard remote sensing spectral bands."""
    blue: Optional[int] = None
    green: Optional[int] = None
    red: Optional[int] = None
    nir: Optional[int] = None
    swir: Optional[int] = None
    sensor_name: str = "generic"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blue": self.blue,
            "green": self.green,
            "red": self.red,
            "nir": self.nir,
            "swir": self.swir,
            "sensor_name": self.sensor_name,
            "description": self.description,
            "available_composites": self.available_composites,
        }

    @property
    def is_sar(self) -> bool:
        name = (self.sensor_name or "").lower()
        return "sar" in name or "radar" in name or "sentinel-1" in name

    @property
    def available_composites(self) -> List[str]:
        if self.is_sar:
            return ["sar_db"]
        if self.nir is not None:
            comps = ["rgb", "cir", "ndvi", "ndwi"]
        else:
            comps = ["rgb"]
        if self.swir is not None:
            comps.extend(["mndwi", "ndbi"])
        return list(dict.fromkeys(comps))


# Standard Pre-defined Sensor Profiles
# Sentinel-2 (L2A): B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1
PROFILE_SENTINEL2_12B = BandMapping(
    blue=1, green=2, red=3, nir=7, swir=10,
    sensor_name="Sentinel-2", description="Sentinel-2 L2A 12/13-Band Stack (B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR)"
)
PROFILE_SENTINEL2_6B = BandMapping(
    blue=0, green=1, red=2, nir=3, swir=4,
    sensor_name="Sentinel-2", description="Sentinel-2 6-Band Subset (B2, B3, B4, B8, B11, B12)"
)
PROFILE_SENTINEL2_4B = BandMapping(
    blue=0, green=1, red=2, nir=3, swir=None,
    sensor_name="Sentinel-2", description="Sentinel-2 10m VNIR 4-Band (B2, B3, B4, B8)"
)

# Landsat-8/9 OLI: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1
PROFILE_LANDSAT89_7B = BandMapping(
    blue=1, green=2, red=3, nir=4, swir=5,
    sensor_name="Landsat-8/9", description="Landsat-8/9 OLI 7-Band Stack (B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR)"
)
PROFILE_LANDSAT89_6B = BandMapping(
    blue=0, green=1, red=2, nir=3, swir=4,
    sensor_name="Landsat-8/9", description="Landsat-8/9 OLI 6-Band (B2, B3, B4, B5, B6, B7)"
)

# Cartosat-2S / Cartosat-3 Multispectral: B1=Blue, B2=Green, B3=Red, B4=NIR
PROFILE_CARTOSAT_4B = BandMapping(
    blue=0, green=1, red=2, nir=3, swir=None,
    sensor_name="Cartosat-2S", description="Cartosat-2S Multispectral (B1=Blue, B2=Green, B3=Red, B4=NIR)"
)

# Generic Planet / Aerial / Orthomosaic 4-Band (BGRN or RGBN)
PROFILE_GENERIC_4B_BGRN = BandMapping(
    blue=0, green=1, red=2, nir=3, swir=None,
    sensor_name="Generic-4Band-BGRN", description="Generic 4-Band (Blue, Green, Red, NIR)"
)
PROFILE_GENERIC_4B_RGBN = BandMapping(
    red=0, green=1, blue=2, nir=3, swir=None,
    sensor_name="Generic-4Band-RGBN", description="Generic 4-Band (Red, Green, Blue, NIR)"
)

# Generic 3-Band Standard True-Color RGB
PROFILE_GENERIC_RGB = BandMapping(
    red=0, green=1, blue=2, nir=None, swir=None,
    sensor_name="Generic-RGB", description="Standard 3-Band Optical (Red, Green, Blue)"
)

# Grayscale / Single-Band / SAR
PROFILE_GRAYSCALE = BandMapping(
    red=0, green=0, blue=0, nir=None, swir=None,
    sensor_name="Grayscale-SAR", description="Single-Band Grayscale / SAR Backscatter"
)


# ---------------------------------------------------------------------------
# 2. Dynamic Band Profile Detector
# ---------------------------------------------------------------------------

def detect_sensor_profile(
    source: Any,
    tags: Optional[Dict[str, Any]] = None,
    descriptions: Optional[Union[List[Optional[str]], Tuple[Optional[str], ...]]] = None,
    colorinterp: Optional[List[Any]] = None,
) -> BandMapping:
    """Inspects raster tags, descriptions, wavelength markers, and channel counts to map bands.

    Args:
        source: A rasterio DatasetReader, a file path (str/Path), a numpy array, or an integer band count.
        tags: Optional dictionary of raster metadata tags.
        descriptions: Optional sequence of band description strings.
        colorinterp: Optional sequence of rasterio ColorInterp enums or strings.

    Returns:
        BandMapping resolving channel indices for Blue, Green, Red, NIR, and SWIR.
    """
    count: int = 4
    file_path: Optional[Path] = None

    # Case A: File path provided
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists() and rasterio is not None and p.suffix.lower() in (".tif", ".tiff", ".gtiff"):
            try:
                with rasterio.open(p) as src:
                    return detect_sensor_profile(
                        source=src.count,
                        tags=src.tags(),
                        descriptions=src.descriptions,
                        colorinterp=src.colorinterp,
                    )
            except Exception as exc:
                logger.debug("rasterio_open_failed for %s: %s", p, exc)
        file_path = p
        count = 4

    # Case B: rasterio DatasetReader instance
    elif hasattr(source, "count") and hasattr(source, "tags"):
        return detect_sensor_profile(
            source=source.count,
            tags=source.tags(),
            descriptions=getattr(source, "descriptions", None),
            colorinterp=getattr(source, "colorinterp", None),
        )

    # Case C: Metadata dictionary
    elif isinstance(source, dict):
        count = int(source.get("band_count") or source.get("count") or 4)
        tags = tags or source.get("tags") or {}
        descriptions = descriptions or source.get("descriptions")
        colorinterp = colorinterp or source.get("colorinterp")

    # Case D: Numpy ndarray
    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            count = 1
        elif source.ndim == 3:
            # Determine whether (C, H, W) or (H, W, C)
            if source.shape[0] <= 16 and source.shape[0] < source.shape[1]:
                count = source.shape[0]
            elif source.shape[-1] <= 16:
                count = source.shape[-1]
            else:
                count = source.shape[0]
        else:
            count = 4

    # Case E: Integer count
    elif isinstance(source, (int, np.integer)):
        count = int(source)

    tags = tags or {}
    tag_str = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
    if file_path:
        tag_str += f" filename={file_path.name.lower()}"

    # 1. Band Descriptions Inspection (highest accuracy when present)
    if descriptions and any(descriptions):
        mapping = BandMapping(sensor_name="custom_descriptions")
        matched = 0
        for idx, desc in enumerate(descriptions):
            if not desc:
                continue
            d = str(desc).strip().lower()
            if any(k in d for k in ["blue", "b02", "b2"]) and mapping.blue is None:
                mapping.blue = idx
                matched += 1
            elif any(k in d for k in ["green", "b03", "b3"]) and mapping.green is None:
                mapping.green = idx
                matched += 1
            elif any(k in d for k in ["red", "b04", "b4"]) and mapping.red is None:
                mapping.red = idx
                matched += 1
            elif any(k in d for k in ["nir", "near-infrared", "b08", "b8", "b5"]) and mapping.nir is None:
                mapping.nir = idx
                matched += 1
            elif any(k in d for k in ["swir", "b11", "b12", "b6", "b7"]) and mapping.swir is None:
                mapping.swir = idx
                matched += 1

        if matched >= 3:
            mapping.description = f"Mapped from band descriptions ({matched} matched)"
            return mapping

    # 2. Metadata / Sensor Tags Matching
    # A. Sentinel-2
    if any(k in tag_str for k in ["sentinel-2", "sentinel 2", "s2a", "s2b", "msi"]):
        if count >= 11:
            return PROFILE_SENTINEL2_12B
        if count == 6:
            return PROFILE_SENTINEL2_6B
        if count == 4:
            return PROFILE_SENTINEL2_4B

    # B. Landsat-8 / Landsat-9
    if any(k in tag_str for k in ["landsat", "lc08", "lc09", "lo08", "lo09", "oli"]):
        if count >= 7:
            return PROFILE_LANDSAT89_7B
        if count == 6:
            return PROFILE_LANDSAT89_6B
        if count == 4:
            return BandMapping(blue=0, green=1, red=2, nir=3, sensor_name="Landsat-8/9", description="Landsat-8/9 4-Band")

    # C. Cartosat-2S / Cartosat-3
    if any(k in tag_str for k in ["cartosat", "isro", "cartosat-2s"]):
        if count == 4:
            return PROFILE_CARTOSAT_4B

    # 3. ColorInterp Tag Checks (Photometric / Color interpretation)
    if colorinterp and len(colorinterp) >= 3:
        ci_names = [str(getattr(c, "name", c)).lower() for c in colorinterp]
        if "red" in ci_names and "green" in ci_names and "blue" in ci_names:
            r_idx = ci_names.index("red")
            g_idx = ci_names.index("green")
            b_idx = ci_names.index("blue")
            nir_idx = ci_names.index("undefined") if "undefined" in ci_names else (3 if count >= 4 else None)
            return BandMapping(
                red=r_idx, green=g_idx, blue=b_idx, nir=nir_idx,
                sensor_name="ColorInterp-Derived",
                description=f"Derived from raster colorinterp ({ci_names})"
            )

    # 4. Band Count Heuristic Fallback
    if count >= 11:
        return PROFILE_SENTINEL2_12B
    if count in (7, 8):
        return PROFILE_LANDSAT89_7B
    if count == 6:
        return PROFILE_SENTINEL2_6B
    if count == 5:
        return BandMapping(
            blue=0, green=1, red=2, nir=3, swir=4,
            sensor_name="Generic-5Band", description="Generic 5-Band (B, G, R, NIR, SWIR)"
        )
    if count == 4:
        # Standard satellite 4-band order: B, G, R, NIR
        return PROFILE_GENERIC_4B_BGRN
    if count == 3:
        return PROFILE_GENERIC_RGB
    if count in (1, 2):
        return PROFILE_GRAYSCALE

    return PROFILE_GENERIC_4B_BGRN


def get_available_composites(
    band_map: Optional[BandMapping] = None,
    band_count: int = 4,
    is_sar: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    include_fallbacks: bool = True,
) -> List[str]:
    """Dynamically determine available spectral composites based on band mapping, count, and tags."""
    tag_str = " ".join(f"{k}={v}" for k, v in (tags or {}).items()).lower() if tags else ""
    is_sar_flag = (
        is_sar
        or (band_map.is_sar if band_map else False)
        or any(k in tag_str for k in ["sar", "sentinel-1", "sigma0", "backscatter", "radar"])
    )
    if is_sar_flag or (band_count in (1, 2) and (is_sar or any(k in tag_str for k in ["sar", "db", "sigma0", "radar", "grayscale"]))):
        return ["sar_db"]

    comps = ["rgb"]
    has_nir = (band_map and band_map.nir is not None) or band_count >= 4
    if has_nir or include_fallbacks:
        comps.extend(["cir", "ndvi", "ndwi"])

    has_swir = (band_map and band_map.swir is not None) or band_count >= 5
    if has_swir or band_count >= 5:
        comps.extend(["mndwi", "ndbi"])

    return list(dict.fromkeys(comps))


# ---------------------------------------------------------------------------
# 3. Core Spectral Index Calculations
# ---------------------------------------------------------------------------

def compute_ndvi(
    red: np.ndarray,
    nir: Optional[np.ndarray] = None,
    green: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes Normalized Difference Vegetation Index with division-by-zero protection.

    - If NIR is present: standard (NIR - Red) / (NIR + Red + 1e-7).
    - If NIR is missing (3-band RGB): Visible Band Vegetation Index (GLI/GRVI):
      (Green - Red) / (Green + Red + 1e-7) normalized between -1.0 and 1.0.
    """
    r = red.astype(np.float32)
    if nir is not None:
        n = nir.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (n - r) / (n + r + 1e-7)
        return np.clip(res, -1.0, 1.0).astype(np.float32)
    elif green is not None:
        g = green.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (g - r) / (g + r + 1e-7)
        return np.clip(res, -1.0, 1.0).astype(np.float32)
    return np.zeros_like(r, dtype=np.float32)


def compute_ndwi(
    green: np.ndarray,
    nir: Optional[np.ndarray] = None,
    blue: Optional[np.ndarray] = None,
    red: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes Normalized Difference Water Index (McFeeters, 1996) with division-by-zero protection.

    - If NIR is present: standard (Green - NIR) / (Green + NIR + 1e-7).
    - If NIR is missing (3-band RGB): Visible Water Contrast:
      (Blue - Red) / (Blue + Red + 1e-7) normalized between -1.0 and 1.0.
    """
    g = green.astype(np.float32)
    if nir is not None:
        n = nir.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (g - n) / (g + n + 1e-7)
        return np.clip(res, -1.0, 1.0).astype(np.float32)
    elif blue is not None and red is not None:
        b = blue.astype(np.float32)
        r = red.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (b - r) / (b + r + 1e-7)
        return np.clip(res, -1.0, 1.0).astype(np.float32)
    return np.zeros_like(g, dtype=np.float32)


def compute_mndwi(
    green: np.ndarray,
    swir: Optional[np.ndarray] = None,
    nir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes Modified Normalized Difference Water Index (Xu, 2006).

    Formula: (Green - SWIR) / (Green + SWIR + 1e-7)
    Falls back gracefully to NDWI using (Green, NIR) if SWIR is unavailable,
    or returns an array of zeros if neither is present.
    """
    g = green.astype(np.float32)
    if swir is not None:
        s = swir.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (g - s) / (g + s + 1e-7)
        return np.clip(res, -1.0, 1.0).astype(np.float32)

    # Graceful fallback to standard NDWI using NIR if available
    if nir is not None:
        return compute_ndwi(green, nir)

    return np.zeros_like(g, dtype=np.float32)


def compute_ndbi(swir: Optional[np.ndarray], nir: np.ndarray) -> np.ndarray:
    """Computes Normalized Difference Built-Up Index (Zha et al., 2003).

    Formula: (SWIR - NIR) / (SWIR + NIR + 1e-7)
    Falls back gracefully to zeros if SWIR is unavailable.
    """
    n = nir.astype(np.float32)
    if swir is not None:
        s = swir.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (s - n) / (s + n + 1e-7)
        return np.clip(res, -1.0, 1.0).astype(np.float32)

    return np.zeros_like(n, dtype=np.float32)


# Backwards-compatible alias for existing tests
def calculate_spectral_indices(
    red_band: np.ndarray, green_band: np.ndarray, nir_band: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates NDVI and NDWI physical indices from raw bands as floating-point tensors."""
    red = red_band.astype(np.float32)
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)

    ndvi_denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(ndvi_denom == 0, 0.0, (nir - red) / ndvi_denom)

    ndwi_denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(ndwi_denom == 0, 0.0, (green - nir) / ndwi_denom)

    return ndvi.astype(np.float32), ndwi.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Composite Generation (CIR False-Color, SAR Decibels, Colormap Rendering)
# ---------------------------------------------------------------------------

def _percentile_stretch_8bit(channel: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Performs remote-sensing contrast enhancement stretching into [0, 255] uint8."""
    ch = channel.astype(np.float32)
    ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
    p_min, p_max = np.percentile(ch, (p_low, p_high))
    if p_max > p_min:
        stretched = np.clip((ch - p_min) / (p_max - p_min) * 255.0, 0, 255)
    else:
        c_min, c_max = ch.min(), ch.max()
        if c_max > c_min:
            stretched = np.clip((ch - c_min) / (c_max - c_min) * 255.0, 0, 255)
        else:
            stretched = np.zeros_like(ch)
    return stretched.astype(np.uint8)


def generate_cir_composite(
    raster_array: np.ndarray,
    band_map: Optional[Union[BandMapping, Dict[str, Any]]] = None,
) -> np.ndarray:
    """Generates a Color-Infrared (CIR) composite mapping (NIR, Red, Green) normalized to 8-bit RGB [0, 255].
    When true NIR is absent (e.g. 3-band RGB), synthesizes a visible-spectrum pseudo-NIR channel
    from Green and Blue channels to yield realistic vegetation contrast.

    Args:
        raster_array: Multiband raster array of shape (C, H, W) or (H, W, C).
        band_map: Optional BandMapping or dict. If omitted, profile is auto-detected.

    Returns:
        np.ndarray of shape (H, W, 3) and dtype uint8.
    """
    arr = np.asarray(raster_array, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)

    # Convert (H, W, C) -> (C, H, W) if needed
    if arr.ndim == 3 and arr.shape[-1] <= 16 and arr.shape[0] > 16:
        arr = np.transpose(arr, (2, 0, 1))

    c, h, w = arr.shape
    if band_map is None:
        bmap = detect_sensor_profile(c)
    elif isinstance(band_map, dict):
        bmap = BandMapping(**band_map)
    else:
        bmap = band_map

    red_idx = bmap.red if bmap.red is not None and 0 <= bmap.red < c else (min(c - 1, 2) if c >= 3 else 0)
    green_idx = bmap.green if bmap.green is not None and 0 <= bmap.green < c else (min(c - 1, 1) if c >= 2 else 0)
    blue_idx = bmap.blue if bmap.blue is not None and 0 <= bmap.blue < c else 0

    red_ch = arr[red_idx]
    green_ch = arr[green_idx]
    blue_ch = arr[blue_idx]

    has_real_nir = bmap.nir is not None and 0 <= bmap.nir < c
    if has_real_nir:
        nir_ch = arr[bmap.nir]
    elif c >= 4:
        nir_ch = arr[min(c - 1, 3)]
    else:
        # Pseudo-NIR synthesis for 3-band RGB: vegetation is strong in green, absorbed in blue
        max_val = max(float(np.max(green_ch)), 255.0)
        nir_ch = np.clip(green_ch * 1.5 - blue_ch * 0.5, 0.0, max_val)

    # Map (NIR -> R, Red -> G, Green -> B)
    cir_rgb = np.stack(
        [
            _percentile_stretch_8bit(nir_ch),
            _percentile_stretch_8bit(red_ch),
            _percentile_stretch_8bit(green_ch),
        ],
        axis=-1,
    )
    return cir_rgb


def generate_sar_db(sar_array: np.ndarray) -> np.ndarray:
    """Converts linear amplitude to decibels sigma_0 = 10 * log10(amplitude^2 + 1e-7),
    normalized to 8-bit grayscale [0, 255].

    Args:
        sar_array: Single-channel SAR amplitude array of shape (H, W) or (1, H, W).

    Returns:
        np.ndarray of shape (H, W) and dtype uint8.
    """
    arr = np.asarray(sar_array, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[0]

    amp = np.abs(arr)
    # sigma_0 in decibels
    sigma0_db = 10.0 * np.log10(amp ** 2 + 1e-7)

    # Natural SAR sigma-0 contrast stretch: 1st to 99th percentile
    return _percentile_stretch_8bit(sigma0_db, p_low=1.0, p_high=99.0)


def render_colormap(index_array: np.ndarray, colormap: str = "RdYlGn") -> bytes:
    """Uses matplotlib to render an 8-bit PNG byte buffer of the normalized index map.

    Args:
        index_array: 2D array of index values (e.g. NDVI or NDWI, typically in [-1.0, 1.0]).
        colormap: Name of matplotlib colormap (default 'RdYlGn').

    Returns:
        Bytes containing the encoded PNG image.
    """
    arr = np.asarray(index_array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Normalize [-1.0, 1.0] -> [0.0, 1.0]
    norm = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)

    try:
        cmap = plt.get_cmap(colormap)
    except Exception:
        cmap = plt.get_cmap("RdYlGn")

    rgba = cmap(norm)  # (H, W, 4) in [0, 1]
    rgba_8bit = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba_8bit).save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# 5. Unified SpectralIndicesEngine Class
# ---------------------------------------------------------------------------

@dataclass
class SpectralIndicesResult:
    """Encapsulates all computed spectral index arrays, mappings, and summary metrics."""
    ndvi: np.ndarray
    ndwi: np.ndarray
    mndwi: np.ndarray
    ndbi: np.ndarray
    band_mapping: BandMapping
    cir_composite: Optional[np.ndarray] = None
    sar_db: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "band_mapping": self.band_mapping.to_dict(),
            "metrics": self.metrics,
            "has_cir": self.cir_composite is not None,
            "has_sar_db": self.sar_db is not None,
        }


class SpectralIndicesEngine:
    """Unified engine for dynamic sensor band profile detection, spectral index computation,
    and false-color / SAR composite generation.
    """

    def __init__(self, default_colormap: str = "RdYlGn") -> None:
        self.default_colormap = default_colormap

    @classmethod
    def detect_band_profile(
        cls,
        source: Any,
        tags: Optional[Dict[str, Any]] = None,
        descriptions: Optional[Union[List[Optional[str]], Tuple[Optional[str], ...]]] = None,
        colorinterp: Optional[List[Any]] = None,
    ) -> BandMapping:
        """Dynamically detect sensor band mapping."""
        return detect_sensor_profile(source, tags=tags, descriptions=descriptions, colorinterp=colorinterp)

    detect_sensor_profile = detect_band_profile

    @staticmethod
    def get_available_composites(
        band_map: Optional[BandMapping] = None,
        band_count: int = 4,
        is_sar: bool = False,
        tags: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        return get_available_composites(band_map=band_map, band_count=band_count, is_sar=is_sar, tags=tags)

    @staticmethod
    def compute_ndvi(
        red: np.ndarray,
        nir: Optional[np.ndarray] = None,
        green: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return compute_ndvi(red, nir=nir, green=green)

    @staticmethod
    def compute_ndwi(
        green: np.ndarray,
        nir: Optional[np.ndarray] = None,
        blue: Optional[np.ndarray] = None,
        red: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return compute_ndwi(green, nir=nir, blue=blue, red=red)

    @staticmethod
    def compute_mndwi(
        green: np.ndarray,
        swir: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return compute_mndwi(green, swir=swir, nir=nir)

    @staticmethod
    def compute_ndbi(swir: Optional[np.ndarray], nir: np.ndarray) -> np.ndarray:
        return compute_ndbi(swir, nir)

    @staticmethod
    def generate_cir_composite(
        raster_array: np.ndarray,
        band_map: Optional[Union[BandMapping, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        return generate_cir_composite(raster_array, band_map=band_map)

    @staticmethod
    def generate_sar_db(sar_array: np.ndarray) -> np.ndarray:
        return generate_sar_db(sar_array)

    @staticmethod
    def render_colormap(index_array: np.ndarray, colormap: str = "RdYlGn") -> io.BytesIO:
        """Use matplotlib.pyplot to apply colormap and return an 8-bit PNG byte buffer io.BytesIO()."""
        res = render_colormap(index_array, colormap=colormap)
        if isinstance(res, bytes):
            b = io.BytesIO(res)
            b.seek(0)
            return b
        return res

    def process_raster(
        self,
        raster_input: Union[str, Path, np.ndarray],
        band_map: Optional[BandMapping] = None,
        generate_cir: bool = True,
        is_sar: bool = False,
    ) -> SpectralIndicesResult:
        """Processes an entire satellite raster dataset into physical indices and composites.

        Args:
            raster_input: Path to GeoTIFF or in-memory numpy array (C, H, W).
            band_map: Optional pre-defined BandMapping. If None, auto-detects dynamically.
            generate_cir: Whether to generate the Color-Infrared (CIR) composite.
            is_sar: Set True if the input represents radar backscatter (SAR).

        Returns:
            SpectralIndicesResult containing index arrays, CIR composite, and statistical metrics.
        """
        arr: np.ndarray
        detected_map: BandMapping

        if isinstance(raster_input, (str, Path)):
            p = Path(raster_input)
            if not p.exists():
                raise FileNotFoundError(f"Raster file not found: {p}")
            if rasterio is None:
                raise RuntimeError("rasterio is required to read raster files from disk")
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)
                detected_map = band_map or self.detect_band_profile(src)
        else:
            arr = np.asarray(raster_input, dtype=np.float32)
            detected_map = band_map or self.detect_band_profile(arr)

        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim == 3 and arr.shape[-1] <= 16 and arr.shape[0] > 16:
            arr = np.transpose(arr, (2, 0, 1))

        c, h, w = arr.shape

        # Resolve bands safely
        def _get_ch(idx: Optional[int], fallback: int = 0) -> Optional[np.ndarray]:
            if idx is not None and 0 <= idx < c:
                return arr[idx]
            if 0 <= fallback < c:
                return arr[fallback]
            return None

        ch_r = _get_ch(detected_map.red, fallback=min(c - 1, 2))
        red_ch = ch_r if ch_r is not None else np.zeros((h, w), dtype=np.float32)

        ch_g = _get_ch(detected_map.green, fallback=min(c - 1, 1))
        green_ch = ch_g if ch_g is not None else np.zeros((h, w), dtype=np.float32)

        ch_b = _get_ch(detected_map.blue, fallback=0)
        blue_ch = ch_b if ch_b is not None else np.zeros((h, w), dtype=np.float32)

        has_real_nir = detected_map.nir is not None and 0 <= detected_map.nir < c
        nir_ch = arr[detected_map.nir] if has_real_nir else (arr[min(c - 1, 3)] if c >= 4 else None)

        swir_ch = _get_ch(detected_map.swir) if detected_map.swir is not None else None

        # Calculate Indices
        ndvi = compute_ndvi(red_ch, nir=nir_ch, green=green_ch)
        ndwi = compute_ndwi(green_ch, nir=nir_ch, blue=blue_ch, red=red_ch)
        mndwi = self.compute_mndwi(green_ch, swir=swir_ch, nir=nir_ch)
        ndbi = self.compute_ndbi(swir=swir_ch, nir=nir_ch if nir_ch is not None else np.zeros((h, w), dtype=np.float32))

        cir = self.generate_cir_composite(arr, detected_map) if generate_cir else None
        sar = self.generate_sar_db(arr[0]) if is_sar else None

        metrics = {
            "ndvi_mean": float(np.mean(ndvi)),
            "ndvi_max": float(np.max(ndvi)),
            "ndvi_min": float(np.min(ndvi)),
            "ndwi_mean": float(np.mean(ndwi)),
            "ndwi_max": float(np.max(ndwi)),
            "ndwi_min": float(np.min(ndwi)),
            "mndwi_mean": float(np.mean(mndwi)),
            "ndbi_mean": float(np.mean(ndbi)),
        }

        return SpectralIndicesResult(
            ndvi=ndvi,
            ndwi=ndwi,
            mndwi=mndwi,
            ndbi=ndbi,
            band_mapping=detected_map,
            cir_composite=cir,
            sar_db=sar,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# 6. Legacy Interfaces (Backwards-compatibility with Phase 3 & ViT Pipelines)
# ---------------------------------------------------------------------------

def generate_n_channel_tensor(
    rgb_array: np.ndarray, ndvi: np.ndarray, ndwi: np.ndarray
) -> Any:
    """Stacks index maps onto visual tensor, expanding 3-channel RGB to 5-channel tensor."""
    if torch is None:
        raise RuntimeError("torch is required to generate n-channel PyTorch tensors")

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

    return torch.cat([rgb_tensor, ndvi_tensor, ndwi_tensor], dim=0)


@dataclass
class SpectralPack:
    feature_tensor: Any
    ndvi: np.ndarray
    ndwi: np.ndarray
    ndvi_mean: float
    ndwi_mean: float
    band_count: int


class SpectralExtractor:
    """Read GeoTIFF bands, compute NDVI/NDWI, stack as 5-channel (RGB+indices) tensor."""

    def extract(self, path: Path) -> SpectralPack:
        if rasterio is None:
            raise RuntimeError("rasterio is required for SpectralExtractor")
        with rasterio.open(path) as src:
            count = src.count
            data = src.read().astype(np.float32)

        red = self._band(data, getattr(settings, "SPECTRAL_RED_BAND_INDEX", 3), fallback=1)
        nir = self._band(data, getattr(settings, "SPECTRAL_NIR_BAND_INDEX", 4), fallback=min(count, 4))
        green = self._band(data, getattr(settings, "SPECTRAL_GREEN_BAND_INDEX", 2), fallback=min(count, 2))

        ndvi, ndwi = calculate_spectral_indices(red, green, nir)
        rgb = self._rgb_stack(data)
        tensor_5ch = generate_n_channel_tensor(rgb, ndvi, ndwi)
        tensor = tensor_5ch.unsqueeze(0)

        pack = SpectralPack(
            feature_tensor=tensor,
            ndvi=ndvi,
            ndwi=ndwi,
            ndvi_mean=float(np.mean(ndvi)),
            ndwi_mean=float(np.mean(ndwi)),
            band_count=count,
        )
        logger.info(
            "spectral_extracted: path=%s, shape=%s, ndvi_mean=%.3f",
            str(path), list(tensor.shape), pack.ndvi_mean
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
