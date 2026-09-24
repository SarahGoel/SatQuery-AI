"""Geospatial coordinate extraction, transform, and GeoJSON utilities for SatQuery AI.

Provides robust translation between raster pixel space and real-world WGS84
geographic coordinates (Latitude/Longitude in EPSG:4326), with resilient fallbacks
for non-georeferenced images (PNG/JPEG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as crs_transform
from shapely.geometry import Polygon, mapping

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    pyproj = None  # type: ignore[assignment]
    HAS_PYPROJ = False

logger = logging.getLogger(__name__)

# Standard reference location for non-georeferenced imagery (ISRO / Indian EO calibration standard)
DEFAULT_CENTER_LON = 77.5946  # Bengaluru / ISRO Satellite Centre
DEFAULT_CENTER_LAT = 12.9716
DEFAULT_SPAN_DEG = 0.045     # ~5 km scene extent


def _is_wgs84(crs_str: str) -> bool:
    """Check if CRS is already standard WGS84 geographic (EPSG:4326)."""
    c = (crs_str or "").strip().upper()
    return c in ("EPSG:4326", "WGS 84", "WGS84", "4326", "+PROJ=LONGLAT +DATUM=WGS84 +NO_DEFS")


def _reproject_xy_to_latlon(xs: Sequence[float], ys: Sequence[float], src_crs: str) -> Tuple[List[float], List[float]]:
    """Reprojects coordinates from arbitrary source CRS to WGS84 [longitude, latitude]."""
    if _is_wgs84(src_crs) or not src_crs or src_crs == "N/A":
        return list(xs), list(ys)

    try:
        if HAS_PYPROJ and pyproj is not None:
            transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lons, lats = transformer.transform(xs, ys)
            return list(lons), list(lats)
    except Exception as pyproj_err:
        logger.debug("pyproj reprojection fallback to rasterio.warp: %s", pyproj_err)

    try:
        lons, lats = crs_transform(src_crs, "EPSG:4326", xs, ys)
        return list(lons), list(lats)
    except Exception as warp_err:
        logger.warning("CRS transformation to EPSG:4326 failed (%s): %s; returning unprojected coords", src_crs, warp_err)
        return list(xs), list(ys)


def extract_geospatial_metadata(image_path: str | Path) -> Dict[str, Any]:
    """Reads GeoTIFF affine transform (src.transform), CRS (src.crs), and calculates

    the outer bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84.
    If the image is not georeferenced or corrupt, gracefully synthesizes consistent
    benchmarking coordinates so downstream map components never crash.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Target image not found at: {path}")

    # 1. Attempt GeoTIFF rasterio read
    try:
        with rasterio.open(path) as src:
            w, h = src.width, src.height
            count = src.count
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
            transform = src.transform

            # Check if dataset has actual valid geo-transform (not default identity)
            has_geotransform = (
                src.transform != Affine.identity()
                and abs(transform.a) > 1e-9
                and abs(transform.e) > 1e-9
            )

            if has_geotransform:
                bounds = src.bounds
                xs = [bounds.left, bounds.right, bounds.right, bounds.left]
                ys = [bounds.bottom, bounds.bottom, bounds.top, bounds.top]
                lons, lats = _reproject_xy_to_latlon(xs, ys, crs)
                min_lon, max_lon = float(min(lons)), float(max(lons))
                min_lat, max_lat = float(min(lats)), float(max(lats))
                bbox_4326 = [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)]

                return {
                    "crs": crs,
                    "transform": list(transform)[:6],
                    "affine": transform,
                    "bounds": bbox_4326,
                    "width": w,
                    "height": h,
                    "band_count": count,
                    "is_georeferenced": True,
                }
    except Exception as exc:
        logger.debug("Rasterio open/read failed for %s (%s); proceeding to PIL fallback", path, exc)

    # 2. Fallback for non-georeferenced formats (PNG, JPEG, unreferenced TIFF)
    w, h = 512, 512
    count = 3
    try:
        from PIL import Image

        with Image.open(path) as pimg:
            w, h = pimg.size
            count = len(pimg.getbands()) if hasattr(pimg, "getbands") else 3
    except Exception as pil_err:
        logger.warning("PIL failed to read dimensions from %s: %s", path, pil_err)

    # Derive standardized benchmark geographic boundaries centered at DEFAULT_CENTER
    span_x = DEFAULT_SPAN_DEG
    span_y = DEFAULT_SPAN_DEG * (h / max(w, 1))
    min_lon = round(DEFAULT_CENTER_LON - span_x / 2.0, 6)
    max_lon = round(DEFAULT_CENTER_LON + span_x / 2.0, 6)
    min_lat = round(DEFAULT_CENTER_LAT - span_y / 2.0, 6)
    max_lat = round(DEFAULT_CENTER_LAT + span_y / 2.0, 6)

    # Construct synthetic affine: mapping pixel (0, 0) -> (min_lon, max_lat)
    res_x = span_x / max(w, 1)
    res_y = span_y / max(h, 1)
    synthetic_transform = Affine(res_x, 0.0, min_lon, 0.0, -res_y, max_lat)

    return {
        "crs": "EPSG:4326",
        "transform": list(synthetic_transform)[:6],
        "affine": synthetic_transform,
        "bounds": [min_lon, min_lat, max_lon, max_lat],
        "width": w,
        "height": h,
        "band_count": count,
        "is_georeferenced": False,
    }


def pixel_to_latlon(
    col: float,
    row: float,
    transform: Any,
    src_crs: Any = "EPSG:4326",
) -> Tuple[float, float]:
    """Converts a pixel (column, row) coordinate to real-world WGS84 (lon, lat) (EPSG:4326)."""
    if isinstance(transform, (list, tuple)):
        aff = Affine(*transform[:6])
    elif isinstance(transform, Affine):
        aff = transform
    else:
        aff = Affine.identity()

    # Apply affine matrix: pixel -> projected map coordinates
    x, y = aff * (float(col), float(row))

    crs_str = str(src_crs) if src_crs else "EPSG:4326"
    lons, lats = _reproject_xy_to_latlon([x], [y], crs_str)
    return float(round(lons[0], 6)), float(round(lats[0], 6))


def bbox_pixel_to_geojson(
    pixel_bbox: Sequence[float],
    transform: Any,
    src_crs: Any = "EPSG:4326",
    properties: Optional[Dict[str, Any]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Transforms a pixel bounding box into a valid GeoJSON Feature Polygon with [longitude, latitude] ring coordinates.

    Supports formats:
    - [ymin, xmin, ymax, xmax] (standard VLM format)
    - [xmin, ymin, xmax, ymax]
    - Normalized 0-1000 scale or 0.0-1.0 scale if image_shape (h, w) is supplied.
    """
    if len(pixel_bbox) < 4:
        raise ValueError(f"pixel_bbox must contain at least 4 coordinates, got {pixel_bbox}")

    b = [float(v) for v in pixel_bbox[:4]]

    # Detect normalized coordinates (0.0 to 1.0 or 0 to 1000)
    if image_shape and (all(0.0 <= v <= 1.0 for v in b) or all(0.0 <= v <= 1000.0 for v in b)):
        h, w = image_shape
        scale = 1000.0 if any(v > 1.0 for v in b) else 1.0
        # Assume [ymin, xmin, ymax, xmax] standard for VLM
        y_min = (b[0] / scale) * h
        x_min = (b[1] / scale) * w
        y_max = (b[2] / scale) * h
        x_max = (b[3] / scale) * w
    else:
        # Standard pixel coordinates: check order
        # If b[0] < b[2] and b[1] < b[3], determine orientation
        # Default interpretation: [ymin, xmin, ymax, xmax] if ymin appears first, or [xmin, ymin, xmax, ymax]
        if b[0] <= b[2] and b[1] <= b[3]:
            # Could be [xmin, ymin, xmax, ymax]
            x_min, y_min, x_max, y_max = b[0], b[1], b[2], b[3]
        else:
            x_min, x_max = min(b[0], b[2]), max(b[0], b[2])
            y_min, y_max = min(b[1], b[3]), max(b[1], b[3])

    # Convert the 4 corners to real-world WGS84 coordinates
    lon_tl, lat_tl = pixel_to_latlon(x_min, y_min, transform, src_crs)
    lon_tr, lat_tr = pixel_to_latlon(x_max, y_min, transform, src_crs)
    lon_br, lat_br = pixel_to_latlon(x_max, y_max, transform, src_crs)
    lon_bl, lat_bl = pixel_to_latlon(x_min, y_max, transform, src_crs)

    ring = [
        [lon_tl, lat_tl],
        [lon_tr, lat_tr],
        [lon_br, lat_br],
        [lon_bl, lat_bl],
        [lon_tl, lat_tl],  # closed ring
    ]

    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)

    props = dict(properties or {})
    props.setdefault("label", "Grounded Feature")
    props.setdefault("confidence", 0.90)
    props.setdefault("pixel_bbox", [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)])
    min_lon = min(lon_tl, lon_tr, lon_br, lon_bl)
    max_lon = max(lon_tl, lon_tr, lon_br, lon_bl)
    min_lat = min(lat_tl, lat_tr, lat_br, lat_bl)
    max_lat = max(lat_tl, lat_tr, lat_br, lat_bl)
    props["bbox_wgs84"] = [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)]

    return {
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": props,
    }


def feature_collection_from_features(
    features: List[Dict[str, Any]],
    crs_name: str = "EPSG:4326",
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wraps a list of GeoJSON features into a valid FeatureCollection with CRS."""
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "properties": properties or {"feature_count": len(features)},
        "features": features,
    }


def compute_geojson_bbox(geojson: Dict[str, Any]) -> Optional[List[float]]:
    """Calculates [min_lon, min_lat, max_lon, max_lat] from any valid GeoJSON object
    (FeatureCollection, Feature, or Geometry).
    Returns None if no coordinates are found.
    """
    if not isinstance(geojson, dict):
        return None

    # If GeoJSON already specifies a valid top-level bbox
    if "bbox" in geojson and isinstance(geojson["bbox"], (list, tuple)) and len(geojson["bbox"]) >= 4:
        return [round(float(v), 6) for v in geojson["bbox"][:4]]

    lons: List[float] = []
    lats: List[float] = []

    def _extract_coords(obj: Any) -> None:
        if not obj:
            return
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                lons.append(float(obj[0]))
                lats.append(float(obj[1]))
            else:
                for item in obj:
                    _extract_coords(item)
        elif isinstance(obj, dict):
            if "coordinates" in obj:
                _extract_coords(obj["coordinates"])
            elif "geometry" in obj:
                _extract_coords(obj["geometry"])
            elif "features" in obj:
                for f in obj.get("features", []):
                    _extract_coords(f)

    _extract_coords(geojson)

    if not lons or not lats:
        return None

    return [
        round(min(lons), 6),
        round(min(lats), 6),
        round(max(lons), 6),
        round(max(lats), 6),
    ]

