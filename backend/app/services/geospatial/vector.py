"""Convert generated raster masks to GeoJSON polygons via Shapely."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def raster_mask_to_geojson(
    geotiff_path: Path,
    mask: np.ndarray,
    min_area: float = 0.0,
) -> dict[str, Any]:
    """Polygonize a binary/float mask in the GeoTIFF's native CRS, emit GeoJSON."""
    binary = (mask > 0.5).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary[0]
    features: list[dict[str, Any]] = []
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
        for geom, value in shapes(binary, mask=binary.astype(bool), transform=transform):
            poly = shape(geom)
            if poly.is_empty or poly.area <= min_area:
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {"value": int(value)},
                    "geometry": mapping(poly),
                }
            )
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs or "EPSG:4326"}},
        "features": features,
    }
    return collection


def dissolve_geojson(geojson: dict[str, Any]) -> dict[str, Any]:
    geoms = [shape(feat["geometry"]) for feat in geojson.get("features", [])]
    if not geoms:
        return geojson
    dissolved = unary_union(geoms)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(dissolved),
            }
        ],
    }
