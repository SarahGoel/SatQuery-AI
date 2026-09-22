"""Comprehensive verification test suite for the 5 Canonical Benchmark Queries and Geo-Grounding.

Canonical Queries:
1. "Describe the land-cover and major objects visible in this image." -> single_vqa
2. "Highlight the water body referred to in the query." -> single_grounding
3. "What changed between these two dates, and where did the change occur?" -> bitemporal_change
4. "Use the optical and SAR images together to identify built-up and water-covered regions." -> cross_modal
5. "Has the built-up area increased, decreased, or remained unchanged?" -> bitemporal_change ([INCREASED] / [DECREASED] / [REMAINED UNCHANGED])
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.agents.router import (
    TASK_BITEMPORAL_CHANGE,
    TASK_CROSS_MODAL,
    TASK_SINGLE_GROUNDING,
    TASK_SINGLE_VQA,
    InputInspectorNode,
)
from app.services.agent import SatQueryController
from app.services.geo_utils import (
    bbox_pixel_to_geojson,
    compute_geojson_bbox,
    extract_geospatial_metadata,
    feature_collection_from_features,
    pixel_to_latlon,
)
from app.services.grounding_service import GroundingService
from app.services.models.change_vqa import TemporalChangeVQA


def _create_synthetic_geotiff(
    path: Path,
    bounds: Tuple[float, float, float, float] = (77.50, 12.90, 77.60, 13.00),
    shape: Tuple[int, int] = (64, 64),
    bands: int = 4,
    pattern: str = "uniform",
) -> Path:
    """Helper to generate standard georeferenced GeoTIFF with controllable patterns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bounds
    h, w = shape
    transform = from_bounds(west, south, east, north, w, h)

    if pattern == "uniform":
        data = np.ones((bands, h, w), dtype=np.float32) * 128.0
    elif pattern == "water":
        # Simulate optical image with dark water reservoir in top-left
        data = np.ones((bands, h, w), dtype=np.float32) * 180.0
        data[:, : h // 2, : w // 2] = 20.0  # low reflectance water region
    elif pattern == "builtup_increase":
        data = np.ones((bands, h, w), dtype=np.float32) * 220.0
    elif pattern == "baseline":
        data = np.ones((bands, h, w), dtype=np.float32) * 90.0
    else:
        data = np.random.uniform(50.0, 200.0, (bands, h, w)).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


# ==============================================================================
# 1. GEO-UTILS TESTS
# ==============================================================================
def test_geo_utils_metadata_extraction_and_conversion(tmp_path: Path) -> None:
    tif_path = tmp_path / "geo_test.tif"
    _create_synthetic_geotiff(tif_path, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64))

    meta = extract_geospatial_metadata(tif_path)
    assert meta["is_georeferenced"] is True
    assert meta["crs"] == "EPSG:4326"
    assert len(meta["bounds"]) == 4
    assert meta["bounds"][0] == pytest.approx(77.50, abs=1e-4)
    assert meta["bounds"][3] == pytest.approx(13.00, abs=1e-4)

    # Pixel to Lat/Lon conversion
    lon_tl, lat_tl = pixel_to_latlon(0, 0, meta["affine"], meta["crs"])
    assert lon_tl == pytest.approx(77.50, abs=1e-4)
    assert lat_tl == pytest.approx(13.00, abs=1e-4)

    # Bounding box pixel to GeoJSON Polygon
    feature = bbox_pixel_to_geojson(
        pixel_bbox=[10, 10, 30, 30],
        transform=meta["affine"],
        src_crs=meta["crs"],
        properties={"label": "Water Reservoir"},
    )
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["label"] == "Water Reservoir"

    # FeatureCollection & compute_geojson_bbox
    fc = feature_collection_from_features([feature])
    bbox = compute_geojson_bbox(fc)
    assert bbox is not None
    assert len(bbox) == 4
    assert bbox[0] < bbox[2]  # min_lon < max_lon
    assert bbox[1] < bbox[3]  # min_lat < max_lat


def test_geo_utils_fallback_for_non_georeferenced(tmp_path: Path) -> None:
    from PIL import Image

    png_path = tmp_path / "test.png"
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    img.save(png_path)

    meta = extract_geospatial_metadata(png_path)
    assert meta["is_georeferenced"] is False
    assert meta["crs"] == "EPSG:4326"
    assert len(meta["bounds"]) == 4
    # Bounds should be safely centered around default reference coordinates
    assert 77.0 < meta["bounds"][0] < 78.0
    assert 12.0 < meta["bounds"][1] < 13.5


# ==============================================================================
# 2. INTENT ROUTING TESTS FOR CANONICAL BENCHMARK QUERIES
# ==============================================================================
def test_router_canonical_benchmark_queries(tmp_path: Path) -> None:
    img1 = str(tmp_path / "img1.tif")
    img2 = str(tmp_path / "img2.tif")

    # Query 1: Single-image scene description
    task1 = InputInspectorNode.inspect(
        query="Describe the land-cover and major objects visible in this image.",
        filepaths=[img1],
    )
    assert task1 == "single_image_vqa"

    # Query 2: Single-image water grounding
    task2 = InputInspectorNode.inspect(
        query="Highlight the water body referred to in the query.",
        filepaths=[img1],
    )
    assert task2 == "single_image_grounding"

    # Query 3: Bi-temporal change detection & localization
    task3 = InputInspectorNode.inspect(
        query="What changed between these two dates, and where did the change occur?",
        filepaths=[img1, img2],
    )
    assert task3 == "bi_temporal_change_analysis"

    # Query 4: Optical + SAR joint analysis
    task4 = InputInspectorNode.inspect(
        query="Use the optical and SAR images together to identify built-up and water-covered regions.",
        filepaths=[img1, img2],
    )
    assert task4 == "cross_modal_joint_analysis"

    # Query 5: Bi-temporal directional change verdict
    task5 = InputInspectorNode.inspect(
        query="Has the built-up area increased, decreased, or remained unchanged?",
        filepaths=[img1, img2],
    )
    assert task5 == "bi_temporal_change_analysis"

    # Mismatch rejections
    with pytest.raises(ValueError, match="Bi-temporal change detection requires two spatially aligned images"):
        InputInspectorNode.inspect(
            query="Has the built-up area increased, decreased, or remained unchanged?",
            filepaths=[img1],
        )

    with pytest.raises(ValueError, match="requires both Optical and SAR imagery"):
        InputInspectorNode.inspect(
            query="Use the optical and SAR images together to identify built-up and water-covered regions.",
            filepaths=[img1],
        )


# ==============================================================================
# 3. END-TO-END EXECUTION OF THE 5 CANONICAL BENCHMARK QUERIES
# ==============================================================================
def test_canonical_query_1_single_image_vqa(tmp_path: Path) -> None:
    """1. 'Describe the land-cover and major objects visible in this image.'"""
    img_path = tmp_path / "scene.tif"
    _create_synthetic_geotiff(img_path, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64))

    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="Describe the land-cover and major objects visible in this image.",
        filepaths=[str(img_path)],
    )

    assert trace.task_type == TASK_SINGLE_VQA
    assert controller.last_bbox is not None
    assert len(controller.last_bbox) == 4
    # Output must provide rich land cover / object description
    assert "land-cover" in trace.output.lower() or "land cover" in trace.output.lower() or "scene" in trace.output.lower()
    assert trace.geojson is not None
    assert trace.geojson.get("features") is not None


def test_canonical_query_2_visual_grounding_water(tmp_path: Path) -> None:
    """2. 'Highlight the water body referred to in the query.'"""
    img_path = tmp_path / "water_scene.tif"
    _create_synthetic_geotiff(img_path, pattern="water", bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64))

    # Test GroundingService directly
    assert GroundingService.is_water_target("Highlight the water body referred to in the query.") is True

    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="Highlight the water body referred to in the query.",
        filepaths=[str(img_path)],
    )

    assert trace.task_type == TASK_SINGLE_GROUNDING
    assert trace.geojson is not None
    features = trace.geojson.get("features", [])
    assert len(features) >= 1

    first_feat = features[0]
    props = first_feat.get("properties", {})
    assert props.get("class") == "water"
    assert props.get("category") == "water"
    assert "Water Body" in props.get("label", "")

    # Grounded bounding box should be updated
    assert controller.last_bbox is not None
    assert len(controller.last_bbox) == 4
    assert controller.last_bbox[0] < controller.last_bbox[2]


def test_canonical_query_3_bitemporal_change_localized(tmp_path: Path) -> None:
    """3. 'What changed between these two dates, and where did the change occur?'"""
    t1 = tmp_path / "t1.tif"
    t2 = tmp_path / "t2.tif"
    _create_synthetic_geotiff(t1, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64), pattern="baseline")
    _create_synthetic_geotiff(t2, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64), pattern="water")

    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="What changed between these two dates, and where did the change occur?",
        filepaths=[str(t1), str(t2)],
    )

    assert trace.task_type == TASK_BITEMPORAL_CHANGE
    assert trace.geojson is not None
    assert "CD-VQA-Pro" in trace.models_executed or "change-vqa" in trace.models_executed
    # Answer should address change and location
    out_lower = trace.output.lower()
    assert "change" in out_lower or "difference" in out_lower
    assert controller.last_bbox is not None


def test_canonical_query_4_cross_modal_joint_analysis(tmp_path: Path) -> None:
    """4. 'Use the optical and SAR images together to identify built-up and water-covered regions.'"""
    opt = tmp_path / "opt.tif"
    sar = tmp_path / "sar.tif"
    _create_synthetic_geotiff(opt, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64), bands=3)
    _create_synthetic_geotiff(sar, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64), bands=2)

    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="Use the optical and SAR images together to identify built-up and water-covered regions.",
        filepaths=[str(opt), str(sar)],
    )

    assert trace.task_type == TASK_CROSS_MODAL
    assert "Opt-SAR-Fusion-Net" in trace.models_executed or "cross_modal_analysis_tool" in trace.models_executed
    assert trace.geojson is not None
    features = trace.geojson.get("features", [])
    assert len(features) >= 1
    out_lower = trace.output.lower()
    assert "optical" in out_lower or "sar" in out_lower or "radar" in out_lower


def test_canonical_query_5_bitemporal_directional_verdict(tmp_path: Path) -> None:
    """5. 'Has the built-up area increased, decreased, or remained unchanged?'"""
    t1 = tmp_path / "t1_urban.tif"
    t2 = tmp_path / "t2_urban.tif"
    _create_synthetic_geotiff(t1, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64), pattern="baseline")
    _create_synthetic_geotiff(t2, bounds=(77.50, 12.90, 77.60, 13.00), shape=(64, 64), pattern="builtup_increase")

    controller = SatQueryController()
    trace = controller.execute_workflow(
        query="Has the built-up area increased, decreased, or remained unchanged?",
        filepaths=[str(t1), str(t2)],
    )

    assert trace.task_type == TASK_BITEMPORAL_CHANGE
    # Must start with explicit verdict: [INCREASED], [DECREASED], or [REMAINED UNCHANGED]
    trimmed_output = trace.output.strip()
    assert (
        trimmed_output.startswith("[INCREASED]")
        or trimmed_output.startswith("[DECREASED]")
        or trimmed_output.startswith("[REMAINED UNCHANGED]")
    ), f"Expected verdict tag at beginning of output, got: {trimmed_output}"

    # Verify percentage delta is reported
    assert "%" in trimmed_output
    assert trace.geojson is not None
    assert controller.last_bbox is not None
