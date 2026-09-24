"""GeoTIFF spatial metadata parser — CRS, affine, bounds, and sensor modality."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import rasterio

_BACKSCATTER_TOKENS = (
    "sigma0",
    "sigma_0",
    "sigma-nought",
    "gamma0",
    "gamma_0",
    "beta0",
    "backscatter",
    "vv",
    "vh",
    "hh",
    "hv",
    "risat",
    "sar",
    "c-band",
    "cband",
)


def _collect_tag_haystack(src: rasterio.io.DatasetReader) -> str:
    parts: List[str] = []
    parts.extend(str(v) for v in src.tags().values())
    parts.extend(str(k) for k in src.tags().keys())
    for idx in range(1, src.count + 1):
        band_tags = src.tags(idx)
        parts.extend(str(v) for v in band_tags.values())
        parts.extend(str(k) for k in band_tags.keys())
        desc = src.descriptions[idx - 1] if src.descriptions else None
        if desc:
            parts.append(desc)
    return " ".join(parts).lower()


def _has_backscatter_tags(src: rasterio.io.DatasetReader) -> bool:
    haystack = _collect_tag_haystack(src)
    return any(token in haystack for token in _BACKSCATTER_TOKENS)


def _sar_polarizations(src: rasterio.io.DatasetReader, band_count: int) -> List[str]:
    haystack = _collect_tag_haystack(src)
    found: List[str] = []
    for pol in ("VV", "VH", "HH", "HV"):
        if pol.lower() in haystack and pol not in found:
            found.append(pol)
    if found:
        return found[:band_count]
    return ["VV", "VH"] if band_count == 2 else ["VV"]


def parse_geotiff_metadata(filepath: str) -> Dict[str, Any]:
    """
    Ingests a raw GeoTIFF, parses critical metadata attributes and spatial transform vectors [31, 34].
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing target file at path: {filepath}")

    with rasterio.open(filepath) as src:
        bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        affine_transform = list(src.transform)[:6]
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        band_count = src.count
        data_types = [src.dtypes[i] for i in range(band_count)]

        # Heuristic rules to extract modality based on bands [31, 33]
        # Optical/Multispectral (Cartosat-2S) when bands >= 4.
        # SAR (RISAT C-Band) when bands are 1 or 2 (backscatter tags refine polarizations).
        if band_count >= 4:
            modality = "Optical/Multispectral"
            modalities = ["Red", "Green", "Blue", "NIR"]
        elif band_count in [1, 2]:
            modality = "SAR"
            modalities = _sar_polarizations(src, band_count)
            if not _has_backscatter_tags(src):
                # Still SAR by band count; polarizations default to VV[/VH].
                modalities = ["VV", "VH"] if band_count == 2 else ["VV"]
        else:
            modality = "RGB"
            modalities = ["Red", "Green", "Blue"]

        return {
            "crs": crs,
            "bounds": bounds,
            "affine_transform": affine_transform,
            "modality": modality,
            "modalities": modalities,
            "shape": (src.width, src.height),
            "band_count": band_count,
            "dtypes": data_types,
        }
