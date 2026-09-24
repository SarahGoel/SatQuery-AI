"""Convert raster masks and pixel boxes to ISRO-standard GeoJSON FeatureCollections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import rasterio
import rasterio.features
from rasterio.transform import Affine
from rasterio.warp import transform as crs_transform
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

NMS_IOU_THRESHOLD = 0.45
GEOJSON_CRS = {"type": "name", "properties": {"name": "EPSG:4326"}}

TASK_SINGLE_GROUNDING = "grounding"
TASK_CHANGE_DETECTION = "change_detection"
TASK_CROSS_MODAL_FUSION = "cross_modal"
TASK_VQA_FOCUS = "grounding"

TASK_TYPE_ALIASES = {
    "single_image_grounding": TASK_SINGLE_GROUNDING,
    "single_grounding": TASK_SINGLE_GROUNDING,
    "grounding": TASK_SINGLE_GROUNDING,
    "bi_temporal_change_analysis": TASK_CHANGE_DETECTION,
    "bitemporal_change": TASK_CHANGE_DETECTION,
    "change_detection": TASK_CHANGE_DETECTION,
    "flood": TASK_CHANGE_DETECTION,
    "cross_modal_joint_analysis": TASK_CROSS_MODAL_FUSION,
    "cross_modal": TASK_CROSS_MODAL_FUSION,
    "cross_modal_fusion": TASK_CROSS_MODAL_FUSION,
    "single_image_vqa": TASK_VQA_FOCUS,
    "single_vqa": TASK_VQA_FOCUS,
    "vqa_focus": TASK_VQA_FOCUS,
    "vqa": TASK_VQA_FOCUS,
}

COLOR_BY_TASK = {
    TASK_SINGLE_GROUNDING: "#06b6d4",
    TASK_CHANGE_DETECTION: "#ef4444",
    TASK_CROSS_MODAL_FUSION: "#f59e0b",
    TASK_VQA_FOCUS: "#06b6d4",
}

CATEGORY_BY_TASK = {
    TASK_SINGLE_GROUNDING: "infrastructure",
    TASK_CHANGE_DETECTION: "flood",
    TASK_CROSS_MODAL_FUSION: "sar_anomaly",
    TASK_VQA_FOCUS: "infrastructure",
}


def normalize_task_type(task: str | None) -> str:
    key = (task or "").strip().lower()
    return TASK_TYPE_ALIASES.get(key, TASK_SINGLE_GROUNDING)


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """IoU of axis-aligned boxes [xmin, ymin, xmax, ymax]."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return 0.0 if union <= 0.0 else inter / union


def apply_nms(
    instances: List[Dict[str, Any]],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Drop duplicate detections over the same structure (IoU >= threshold)."""
    if not instances:
        return []
    ranked = sorted(instances, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []
    for candidate in ranked:
        box = candidate.get("box") or candidate.get("bbox_pixel")
        if not box or len(box) < 4:
            kept.append(candidate)
            continue
        if any(compute_iou(box, existing.get("box") or existing.get("bbox_pixel") or []) >= iou_threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def convert_raster_mask_to_geojson(
    mask: np.ndarray,
    transform: List[float],
    crs: str,
    *,
    dissolve: bool = False,
) -> Dict[str, Any]:
    """Polygonize a binary mask. Default: one Polygon feature per connected region (no hull merge)."""
    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[0]
    binary = (binary > 0).astype(np.uint8)

    affine = Affine(*transform) if not isinstance(transform, Affine) else transform
    shapes = rasterio.features.shapes(
        binary.astype(np.int16),
        mask=(binary > 0),
        transform=affine,
    )

    polygons = []
    for geom, value in shapes:
        if int(value) == 1:
            polygons.append(shape(geom))

    if not polygons:
        return empty_feature_collection(crs)

    union_poly = _as_multipolygon(unary_union(polygons))
    rings = [_polygon_rings(poly) for poly in union_poly.geoms]

    features = []
    for idx, poly in enumerate(polygons, start=1):
        if poly.is_empty or poly.area <= 0:
            continue
        features.append(
            {
                "type": "Feature",
                "id": f"feat-{idx:03d}",
                "geometry": mapping(poly),
                "properties": {"crs": crs, "label": "Detected region"},
            }
        )

    return {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": rings,
        },
        "properties": {"crs": crs, "dissolved": dissolve},
        "features": features,
    }


def raster_mask_to_geojson(
    geotiff_path: Path,
    mask: np.ndarray,
    min_area: float = 0.0,
    *,
    dissolve: bool = False,
    task_type: str | None = None,
    label: str = "Detected region",
    category: str | None = None,
    confidence: float = 0.85,
) -> dict[str, Any]:
    """Polygonize a mask in the GeoTIFF grid; emit EPSG:4326 FeatureCollection (one feature per object)."""
    with rasterio.open(geotiff_path) as src:
        affine = src.transform
        src_crs = src.crs.to_string() if src.crs else "EPSG:4326"
        height, width = src.height, src.width
        src_crs_obj = src.crs

    binary = np.asarray(mask, dtype=np.float32)
    if binary.ndim == 3:
        binary = binary[0]
    binary = _resize_mask(binary, width, height)
    labeled = (binary > 0.5).astype(np.uint8)

    raw = convert_raster_mask_to_geojson(
        labeled,
        [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f],
        src_crs,
        dissolve=dissolve,
    )
    if dissolve and raw.get("type") == "Feature":
        collection = {
            "type": "FeatureCollection",
            "features": [raw],
        }
    else:
        collection = raw if raw.get("type") == "FeatureCollection" else {
            "type": "FeatureCollection",
            "features": [raw] if raw else [],
        }

    if min_area > 0:
        kept = []
        for feat in collection.get("features") or []:
            geom = shape(feat["geometry"])
            if geom.area > min_area:
                kept.append(feat)
        collection["features"] = kept

    instances = []
    for feat in collection.get("features") or []:
        geom = shape(feat["geometry"])
        bbox_px = _geom_pixel_bbox(geom, affine)
        instances.append(
            {
                "geometry": geom,
                "box": bbox_px,
                "confidence": float((feat.get("properties") or {}).get("confidence", confidence)),
                "label": (feat.get("properties") or {}).get("label") or label,
                "category": category,
            }
        )
    instances = apply_nms(instances)
    return _instances_geoms_to_collection(
        instances,
        src_crs_obj,
        affine,
        task_type=task_type or TASK_CHANGE_DETECTION,
        default_label=label,
        default_category=category,
    )


def dissolve_geojson(geojson: dict[str, Any]) -> dict[str, Any]:
    geoms = [shape(feat["geometry"]) for feat in geojson.get("features", []) if feat.get("geometry")]
    if not geoms:
        return geojson
    dissolved = unary_union(geoms)
    task = geojson.get("task_type") or TASK_CHANGE_DETECTION
    return standardize_feature_collection(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(dissolved),
                    "properties": {"label": "Catchment / extent", "category": "water"},
                }
            ],
        },
        task_type=task,
    )


def instances_to_geojson(
    geotiff_path: Path,
    instances: List[Dict[str, Any]],
    default_label: str = "Detected Object",
    *,
    task_type: str | None = None,
    category: str | None = None,
) -> Dict[str, Any]:
    """Map pixel boxes through the GeoTIFF affine into EPSG:4326 polygons (one Feature per object)."""
    with rasterio.open(geotiff_path) as src:
        affine = src.transform
        src_crs = src.crs

    filtered = apply_nms(list(instances), iou_threshold=NMS_IOU_THRESHOLD)
    prepared = []
    for inst in filtered:
        box_px = inst.get("box") or inst.get("bbox_pixel") or [0, 0, 10, 10]
        x1, y1, x2, y2 = [float(v) for v in box_px[:4]]

        poly_coords = inst.get("polygon") or inst.get("coordinates")
        if poly_coords and len(poly_coords) >= 3:
            geo_pts = [affine * (float(pt[0]), float(pt[1])) for pt in poly_coords]
            if geo_pts[0] != geo_pts[-1]:
                geo_pts.append(geo_pts[0])
            poly = Polygon(geo_pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
        else:
            poly = _pixel_box_to_polygon(affine, x1, y1, x2, y2)

        prepared.append(
            {
                "geometry": poly,
                "box": [x1, y1, x2, y2],
                "confidence": float(inst.get("confidence", 0.85)),
                "label": inst.get("label") or default_label,
                "category": inst.get("category") or category,
            }
        )
    return _instances_geoms_to_collection(
        prepared,
        src_crs,
        affine,
        task_type=task_type or TASK_SINGLE_GROUNDING,
        default_label=default_label,
        default_category=category,
    )


def scene_focus_geojson(
    geotiff_path: Path,
    label: str,
    confidence: float,
) -> Dict[str, Any]:
    """Single AOI polygon for VQA focus using the raster footprint."""
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        affine = src.transform
        src_crs = src.crs
        width, height = src.width, src.height
    poly = Polygon(
        [
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top),
            (bounds.left, bounds.bottom),
        ]
    )
    return _instances_geoms_to_collection(
        [
            {
                "geometry": poly,
                "box": [0.0, 0.0, float(width), float(height)],
                "confidence": float(confidence),
                "label": label or "Scene focus",
                "category": "scene_focus",
            }
        ],
        src_crs,
        affine,
        task_type=TASK_VQA_FOCUS,
        default_label=label or "Scene focus",
        default_category="scene_focus",
    )


def standardize_feature_collection(
    geojson: dict[str, Any] | None,
    *,
    task_type: str | None = None,
    default_label: str = "Detected Object",
    default_category: str | None = None,
) -> Dict[str, Any]:
    """Normalize any GeoJSON-like payload to the SatQuery FeatureCollection contract."""
    task = normalize_task_type(task_type or (geojson or {}).get("task_type") or ((geojson or {}).get("properties") or {}).get("task_type"))
    color = COLOR_BY_TASK.get(task, "#06b6d4")
    category = default_category or CATEGORY_BY_TASK.get(task, "infrastructure")

    raw_features: List[Dict[str, Any]] = []
    if not geojson:
        raw_features = []
    elif geojson.get("type") == "FeatureCollection":
        raw_features = list(geojson.get("features") or [])
    elif geojson.get("type") == "Feature":
        raw_features = [geojson]
    else:
        raw_features = []

    features = []
    areas_km2: List[float] = []
    confidences: List[float] = []
    for idx, feat in enumerate(raw_features, start=1):
        props = dict(feat.get("properties") or {})
        geom = feat.get("geometry") or {}
        conf = float(props.get("confidence") or 0.85)
        area_m2 = float(props.get("area_m2") or _geodesic_area_m2(geom))
        area_km2 = area_m2 / 1e6
        areas_km2.append(area_km2)
        confidences.append(conf)
        feat_id = feat.get("id") or props.get("id") or f"feat-{idx:03d}"

        # Standardize class: 'infrastructure', 'flood', 'sar_anomaly', or 'water'
        prop_class = props.get("class")
        if not prop_class:
            lbl = (props.get("label") or "").lower()
            cat = (props.get("category") or props.get("feature_type") or "").lower()
            if task == "change_detection" or "flood" in lbl or "inundat" in lbl or "flood" in cat:
                prop_class = "flood"
            elif task == "cross_modal" and (props.get("source") == "sar" or "sar" in lbl or "anomaly" in cat):
                prop_class = "sar_anomaly"
            elif "water" in lbl or "water" in cat or "lake" in lbl or "river" in lbl:
                prop_class = "water"
            else:
                prop_class = "infrastructure"

        feat_props = dict(props)
        feat_props.update(
            {
                "label": props.get("label") or default_label,
                "confidence": round(conf, 4),
                "class": prop_class,
                "category": props.get("category") or props.get("feature_type") or category,
                "bbox_pixel": list(props.get("bbox_pixel") or props.get("box_px") or []),
                "color_hint": props.get("color_hint") or color,
                "area_m2": round(area_m2, 2),
                "area_km2": round(area_km2, 6),
            }
        )
        if "feature_type" in props:
            feat_props["feature_type"] = props["feature_type"]

        features.append(
            {
                "type": "Feature",
                "id": str(feat_id) if str(feat_id).startswith("feat-") else f"feat-{idx:03d}",
                "geometry": geom,
                "properties": feat_props,
            }
        )

    return {
        "type": "FeatureCollection",
        "crs": GEOJSON_CRS,
        "properties": {
            "task_type": task,
            "feature_count": len(features),
        },
        "task_type": task,
        "summary": {
            "feature_count": len(features),
            "total_area_km2": round(float(sum(areas_km2)), 4),
            "mean_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
        },
        "features": features,
    }


def empty_feature_collection(crs: str | None = None, task_type: str = "grounding") -> Dict[str, Any]:
    task = normalize_task_type(task_type)
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs or "EPSG:4326"}},
        "properties": {
            "task_type": task,
            "feature_count": 0,
        },
        "task_type": task,
        "features": [],
        "summary": {"feature_count": 0, "total_area_km2": 0.0, "mean_confidence": 0.0},
    }


def _instances_geoms_to_collection(
    instances: Iterable[Dict[str, Any]],
    src_crs: Any,
    affine: Affine,
    *,
    task_type: str,
    default_label: str,
    default_category: str | None,
) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for idx, inst in enumerate(instances, start=1):
        geom = inst.get("geometry")
        if geom is None:
            continue
        geom4326 = _to_epsg_4326(geom, src_crs)
        if geom4326.is_empty:
            continue
        box_px = inst.get("box") or _geom_pixel_bbox(geom, affine)
        inst_class = inst.get("class")
        if not inst_class:
            lbl = (inst.get("label") or default_label).lower()
            cat = (inst.get("category") or default_category or "").lower()
            if task_type == "change_detection" or "flood" in lbl or "inundat" in lbl:
                inst_class = "flood"
            elif task_type == "cross_modal" and (inst.get("source") == "sar" or "sar" in lbl or "anomaly" in cat):
                inst_class = "sar_anomaly"
            elif "water" in lbl or "water" in cat or "lake" in lbl or "river" in lbl:
                inst_class = "water"
            else:
                inst_class = "infrastructure"

        features.append(
            {
                "type": "Feature",
                "id": f"feat-{idx:03d}",
                "geometry": mapping(geom4326),
                "properties": {
                    "label": inst.get("label") or default_label,
                    "confidence": float(inst.get("confidence", 0.85)),
                    "class": inst_class,
                    "category": inst.get("category") or default_category,
                    "bbox_pixel": [round(float(v), 2) for v in box_px[:4]],
                },
            }
        )
    return standardize_feature_collection(
        {"type": "FeatureCollection", "features": features},
        task_type=task_type,
        default_label=default_label,
        default_category=default_category,
    )


def _pixel_box_to_polygon(affine: Affine, x1: float, y1: float, x2: float, y2: float) -> Polygon:
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    geo = [affine * (px, py) for px, py in corners]
    return Polygon(geo)


def _geom_pixel_bbox(geom, affine: Affine) -> List[float]:
    if not hasattr(geom, "bounds") or not geom.bounds or len(geom.bounds) < 4:
        return [0.0, 0.0, 100.0, 100.0]
    minx, miny, maxx, maxy = geom.bounds[:4]
    try:
        inv = ~affine
        c1 = inv * (minx, miny)
        c2 = inv * (maxx, maxy)
        xs = (c1[0], c2[0])
        ys = (c1[1], c2[1])
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    except Exception:
        return [0.0, 0.0, 100.0, 100.0]


def _to_epsg_4326(geom, src_crs) -> Any:
    if src_crs is None:
        return geom
    try:
        epsg = src_crs.to_epsg() if hasattr(src_crs, "to_epsg") else None
        if epsg == 4326:
            return geom
        src_str = src_crs.to_string() if hasattr(src_crs, "to_string") else str(src_crs)
        if "4326" in src_str and "3857" not in src_str:
            return geom

        def _project(xs, ys, zs=None):
            lons, lats = crs_transform(src_crs, "EPSG:4326", list(xs), list(ys))
            return (lons, lats) if zs is None else (lons, lats, zs)

        return shp_transform(_project, geom)
    except Exception:
        return geom


def _geodesic_area_m2(geom_mapping: dict[str, Any] | None) -> float:
    if not geom_mapping:
        return 0.0
    try:
        geom = shape(geom_mapping)
        if geom.is_empty:
            return 0.0
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(geom)
        return abs(float(area))
    except Exception:
        try:
            return abs(float(shape(geom_mapping).area))
        except Exception:
            return 0.0


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    import cv2

    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


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
