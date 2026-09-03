"""Convert generated raster masks to georeferenced GeoJSON via Shapely."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import unary_union


def convert_raster_mask_to_geojson(
    mask: np.ndarray,
    transform: List[float],
    crs: str,
) -> Dict[str, Any]:
    """
    Converts raw raster arrays into georeferenced GeoJSON structures via Shapely geometries [55, 58, 60].
    """
    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[0]
    binary = (binary > 0).astype(np.uint8)

    affine = Affine(*transform) if not isinstance(transform, Affine) else transform
    # Extract geometries as polygon paths
    shapes = rasterio.features.shapes(
        binary.astype(np.int16),
        mask=(binary > 0),
        transform=affine,
    )

    polygons = []
    for geom, value in shapes:
        if value == 1:
            polygons.append(shape(geom))

    if not polygons:
        return {"type": "FeatureCollection", "features": []}

    # Unify overlapping polygons to prevent visual anomalies
    union_poly = unary_union(polygons)
    union_poly = _as_multipolygon(union_poly)

    coordinates = [_polygon_rings(poly) for poly in union_poly.geoms]

    # Build GeoJSON structure
    return {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": coordinates,
        },
        "properties": {
            "crs": crs,
        },
    }


def raster_mask_to_geojson(
    geotiff_path: Path,
    mask: np.ndarray,
    min_area: float = 0.0,
) -> dict[str, Any]:
    """Polygonize a binary/float mask in the GeoTIFF's native CRS, emit GeoJSON."""
    with rasterio.open(geotiff_path) as src:
        affine = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        transform = [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]

    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[0]
    if min_area > 0:
        labeled = (binary > 0.5).astype(np.uint8)
        feature = convert_raster_mask_to_geojson(labeled, transform, crs)
        if feature.get("type") == "FeatureCollection":
            return feature
        geom = shape(feature["geometry"])
        kept = [poly for poly in getattr(geom, "geoms", [geom]) if poly.area > min_area]
        if not kept:
            return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}}, "features": []}
        dissolved = _as_multipolygon(unary_union(kept))
        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs}},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"value": 1, "crs": crs},
                    "geometry": mapping(dissolved),
                }
            ],
        }

    feature = convert_raster_mask_to_geojson(binary, transform, crs)
    if feature.get("type") == "FeatureCollection":
        feature["crs"] = {"type": "name", "properties": {"name": crs}}
        return feature
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [feature],
    }


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


def _as_multipolygon(geom) -> MultiPolygon:
    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    polygons = [part for part in getattr(geom, "geoms", []) if part.geom_type == "Polygon"]
    return MultiPolygon(polygons)


def _polygon_rings(poly) -> list[list[list[float]]]:
    rings = [[list(coord) for coord in poly.exterior.coords]]
    for interior in poly.interiors:
        rings.append([list(coord) for coord in interior.coords])
    return rings
