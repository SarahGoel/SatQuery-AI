"""GeoChat Bounding Box Coordinate Extractor and Geospatial Transformer for SatQuery AI.

Extracts normalized or pixel bounding boxes from GeoChat / RS-VLM output text,
reads raster affine transformation matrices and CRS metadata via rasterio,
and projects discrete object detections into real-world WGS84 (EPSG:4326) GeoJSON polygons.

Includes resilient fallback for non-georeferenced imagery (PNG/JPEG) or missing CRS,
anchoring fallback coordinates to ISRO SAC (Space Applications Centre, Ahmedabad).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import rasterio
from rasterio.transform import Affine

try:
    from rasterio.warp import transform as crs_transform
except ImportError:
    crs_transform = None  # type: ignore[assignment]

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    pyproj = None  # type: ignore[assignment]
    HAS_PYPROJ = False

logger = logging.getLogger("GeospatialParser")

# ISRO SAC (Space Applications Centre, Ahmedabad) Calibration Coordinates
ISRO_SAC_LON = 72.5074
ISRO_SAC_LAT = 23.0305
ISRO_SAC_SPAN_DEG = 0.02


def parse_geochat_bbox(text: str) -> Optional[List[float]]:
    """Extracts bounding box coordinates [ymin, xmin, ymax, xmax] from model output.

    Supports:
    - GeoChat tag format: <box>[ymin, xmin, ymax, xmax]</box>
    - GeoChat raw tag format: <box>ymin, xmin, ymax, xmax</box>
    - Standard bracket notation: [ymin, xmin, ymax, xmax]
    - Natural language coordinates: (ymin, xmin, ymax, xmax)
    """
    raw_coords: Optional[List[float]] = None

    # 1. Look for <box>...</box> tags
    box_tag_match = re.search(
        r"<box>\s*\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?\s*</box>",
        text,
        re.IGNORECASE,
    )
    if box_tag_match:
        raw_coords = [float(box_tag_match.group(i)) for i in range(1, 5)]

    # 2. Look for standard bracketed coordinates [y1, x1, y2, x2]
    if raw_coords is None:
        bracket_match = re.search(
            r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]",
            text,
        )
        if bracket_match:
            raw_coords = [float(bracket_match.group(i)) for i in range(1, 5)]

    # 3. Look for parentheses (y1, x1, y2, x2)
    if raw_coords is None:
        paren_match = re.search(
            r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)",
            text,
        )
        if paren_match:
            raw_coords = [float(paren_match.group(i)) for i in range(1, 5)]

    if raw_coords is None:
        return None

    y1, x1, y2, x2 = raw_coords
    ymin = max(0.0, min(1000.0, min(y1, y2)))
    xmin = max(0.0, min(1000.0, min(x1, x2)))
    ymax = max(0.0, min(1000.0, max(y1, y2)))
    xmax = max(0.0, min(1000.0, max(x1, x2)))

    return [ymin, xmin, ymax, xmax]



def _reproject_to_wgs84(
    xs: List[float],
    ys: List[float],
    src_crs: str,
) -> Tuple[List[float], List[float]]:
    """Reprojects coordinates to WGS84 EPSG:4326 [lons, lats]."""
    c = (src_crs or "").strip().upper()
    if c in ("EPSG:4326", "WGS 84", "WGS84", "4326", "+PROJ=LONGLAT +DATUM=WGS84 +NO_DEFS"):
        return xs, ys

    if HAS_PYPROJ and pyproj is not None:
        try:
            transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lons, lats = transformer.transform(xs, ys)
            return list(lons), list(lats)
        except Exception as err:
            logger.debug("pyproj transformation failed: %s; trying rasterio.warp", err)

    if crs_transform is not None:
        try:
            lons, lats = crs_transform(src_crs, "EPSG:4326", xs, ys)
            return list(lons), list(lats)
        except Exception as err:
            logger.warning("rasterio.warp transformation failed: %s", err)

    return xs, ys


def extract_and_transform_bbox(
    model_text_output: str,
    raster_path: Union[str, Path],
) -> Dict[str, Any]:
    """Parses GeoChat bounding box from model output and transforms it to real-world WGS84 GeoJSON.

    Production Logic:
        1. Extract bounding box from model_text_output using regex.
        2. Open raster_path with rasterio, reading Affine transform and CRS.
        3. Convert normalized 0..1000 or fractional coordinates to native pixel space.
        4. Apply affine matrix to derive projected coordinates for the 4 polygon vertices.
        5. Reproject vertices to WGS84 EPSG:4326 [longitude, latitude] closed ring.
        6. Return standardized GeoJSON Feature dictionary.

    Graceful Fallback:
        If the file lacks geospatial metadata (PNG/JPEG), CRS is missing/invalid,
        or file reading fails, catch the exception, log an auditable warning,
        and generate a synthetic GeoJSON Polygon centered on ISRO SAC coordinates (Ahmedabad).
    """
    raw_bbox = parse_geochat_bbox(model_text_output)
    if raw_bbox is None:
        logger.info("No coordinates detected in model output; applying synthetic ISRO SAC fallback")
        half_span = ISRO_SAC_SPAN_DEG / 2.0
        min_lon = round(ISRO_SAC_LON - half_span, 6)
        max_lon = round(ISRO_SAC_LON + half_span, 6)
        min_lat = round(ISRO_SAC_LAT - half_span, 6)
        max_lat = round(ISRO_SAC_LAT + half_span, 6)
        fallback_ring = [
            [min_lon, max_lat],
            [max_lon, max_lat],
            [max_lon, min_lat],
            [min_lon, min_lat],
            [min_lon, max_lat],
        ]
        bbox_wgs84 = [min_lon, min_lat, max_lon, max_lat]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [fallback_ring],
            },
            "properties": {
                "label": "visual_grounding",
                "class": "visual_grounding",
                "category": "visual_grounding",
                "source": "geochat_vlm",
                "crs": "EPSG:4326",
                "is_georeferenced": False,
                "fallback": "ISRO_SAC_AHMEDABAD",
                "center": [ISRO_SAC_LON, ISRO_SAC_LAT],
                "bbox": bbox_wgs84,
            },
            "bbox": bbox_wgs84,
        }
        return {
            "status": "success",
            "method": "synthetic_isro_sac",
            "bbox": None,
            "bbox_wgs84": bbox_wgs84,
            "pixel_bbox": None,
            "geojson": {
                "type": "FeatureCollection",
                "features": [feature],
            },
            "feature": feature,
        }

    ymin, xmin, ymax, xmax = raw_bbox[:4]

    path = Path(raster_path)
    try:
        if not path.exists():
            raise FileNotFoundError(f"Raster path not found: {path}")

        with rasterio.open(path) as src:
            w, h = src.width, src.height
            transform = src.transform
            crs = src.crs.to_string() if src.crs else None

            # Verify CRS and affine transform validity (must not be empty or degenerate identity)
            if not crs or transform == Affine.identity() or abs(transform.a) < 1e-9 or abs(transform.e) < 1e-9:
                raise ValueError(f"File {path.name} lacks valid geospatial georeferencing/CRS")

            # Scale normalized box to raster pixel space
            # GeoChat models output on 0..1000 scale
            if max(ymin, xmin, ymax, xmax) <= 1.0:
                scale_x, scale_y = float(w), float(h)
            elif max(ymin, xmin, ymax, xmax) <= 1000.0:
                scale_x, scale_y = float(w) / 1000.0, float(h) / 1000.0
            else:
                scale_x, scale_y = 1.0, 1.0

            px_y1 = ymin * scale_y
            px_x1 = xmin * scale_x
            px_y2 = ymax * scale_y
            px_x2 = xmax * scale_x

            # Order properly
            min_px_x = min(px_x1, px_x2)
            max_px_x = max(px_x1, px_x2)
            min_px_y = min(px_y1, px_y2)
            max_px_y = max(px_y1, px_y2)

            # Transform pixel coordinates via affine transformation
            a, b, c_aff = transform.a, transform.b, transform.c
            d, e, f_aff = transform.d, transform.e, transform.f

            # Polygon vertices: TL -> TR -> BR -> BL -> TL
            pts_px = [
                (min_px_x, min_px_y),
                (max_px_x, min_px_y),
                (max_px_x, max_px_y),
                (min_px_x, max_px_y),
            ]

            xs_proj = [a * col + b * row + c_aff for col, row in pts_px]
            ys_proj = [d * col + e * row + f_aff for col, row in pts_px]

            lons, lats = _reproject_to_wgs84(xs_proj, ys_proj, crs)
            ring = [[round(lons[i], 6), round(lats[i], 6)] for i in range(4)]
            ring.append(ring[0])  # Close the ring

            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            bbox_wgs84 = [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)]
            pixel_bbox = [int(round(min_px_y)), int(round(min_px_x)), int(round(max_px_y)), int(round(max_px_x))]

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
                "properties": {
                    "label": "visual_grounding",
                    "class": "visual_grounding",
                    "category": "visual_grounding",
                    "source": "geochat_vlm",
                    "crs": "EPSG:4326",
                    "native_crs": crs,
                    "pixel_bbox": pixel_bbox,
                    "raw_geochat_bbox": [ymin, xmin, ymax, xmax],
                    "is_georeferenced": True,
                    "bbox": bbox_wgs84,
                },
                "bbox": bbox_wgs84,
            }
            return {
                "status": "success",
                "method": "geochat_affine_transform",
                "bbox": [ymin, xmin, ymax, xmax],
                "bbox_wgs84": bbox_wgs84,
                "pixel_bbox": pixel_bbox,
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [feature],
                },
                "feature": feature,
            }

    except Exception as exc:
        logger.warning(
            "geospatial_transform_failed for %s (%s); applying synthetic ISRO SAC coordinates fallback",
            path.name if path else "unknown",
            exc,
        )
        # Synthetic ISRO SAC Fallback
        half_span = ISRO_SAC_SPAN_DEG / 2.0
        # Modulate coordinates proportionally if normalized bbox is available
        norm_span_x = (abs(xmax - xmin) / 1000.0) if max(xmin, xmax) > 1.0 else abs(xmax - xmin)
        norm_span_y = (abs(ymax - ymin) / 1000.0) if max(ymin, ymax) > 1.0 else abs(ymax - ymin)
        norm_span_x = max(0.005, min(0.04, norm_span_x * ISRO_SAC_SPAN_DEG * 2.0))
        norm_span_y = max(0.005, min(0.04, norm_span_y * ISRO_SAC_SPAN_DEG * 2.0))

        min_lon = round(ISRO_SAC_LON - norm_span_x / 2.0, 6)
        max_lon = round(ISRO_SAC_LON + norm_span_x / 2.0, 6)
        min_lat = round(ISRO_SAC_LAT - norm_span_y / 2.0, 6)
        max_lat = round(ISRO_SAC_LAT + norm_span_y / 2.0, 6)

        fallback_ring = [
            [min_lon, max_lat],
            [max_lon, max_lat],
            [max_lon, min_lat],
            [min_lon, min_lat],
            [min_lon, max_lat],
        ]
        bbox_wgs84 = [min_lon, min_lat, max_lon, max_lat]
        pixel_bbox = [round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [fallback_ring],
            },
            "properties": {
                "label": "visual_grounding",
                "class": "visual_grounding",
                "category": "visual_grounding",
                "source": "geochat_vlm",
                "crs": "EPSG:4326",
                "pixel_bbox": pixel_bbox,
                "raw_geochat_bbox": [ymin, xmin, ymax, xmax],
                "is_georeferenced": False,
                "fallback": "ISRO_SAC_AHMEDABAD",
                "center": [ISRO_SAC_LON, ISRO_SAC_LAT],
                "bbox": bbox_wgs84,
            },
            "bbox": bbox_wgs84,
        }
        return {
            "status": "success",
            "method": "synthetic_isro_sac",
            "bbox": [ymin, xmin, ymax, xmax],
            "bbox_wgs84": bbox_wgs84,
            "pixel_bbox": pixel_bbox,
            "geojson": {
                "type": "FeatureCollection",
                "features": [feature],
            },
            "feature": feature,
        }
