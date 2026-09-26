"""Unit and integration tests for temporal pipeline bugfixes and VLM narrative enhancements.

Verifies:
1. Non-technical, conversational 5-6 sentence VLM narrative with zero robotic templates.
2. Directional water classification distinguishing water retreat/desiccation from flooding.
3. Affine matrix scaling in raster_mask_to_geojson with mismatched mask dimensions.
4. Web-ready raster preview generation (Base64 PNG data URI and Leaflet bounds).
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import shape

from app.schemas.validation import QueryResponseEnvelope
from app.services.analytical.temporal_change import TemporalChangeTool
from app.services.geospatial.preview import (
    generate_raster_preview,
    get_leaflet_bounds,
    get_raster_base64_preview,
)
from app.services.geospatial.vector import raster_mask_to_geojson
from app.services.models.rs_vlm import RemoteSensingVLMClient


def _create_synthetic_geotiff(
    path: Path,
    bounds: Tuple[float, float, float, float] = (77.50, 12.90, 77.60, 13.00),
    shape: Tuple[int, int] = (64, 64),
    bands: int = 3,
    pattern: str = "uniform",
) -> Path:
    """Creates a georeferenced GeoTIFF with specified pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bounds
    h, w = shape
    transform = from_bounds(west, south, east, north, w, h)

    data = np.zeros((bands, h, w), dtype=np.float32)
    if pattern == "water":
        # Simulates high water coverage:
        # High Green (band 1), Low Red (band 0), Low NIR (band 2) -> High positive NDWI (water)
        data[0] = 180.0
        data[1] = 130.0
        data[2] = 120.0
        # Water body occupying top-left sector
        data[0, : h // 2, : w // 2] = 20.0   # Low Red
        data[1, : h // 2, : w // 2] = 220.0  # High Green -> NDWI = (220-20)/(240) = +0.833
        data[2, : h // 2, : w // 2] = 15.0   # Low NIR/Blue
    elif pattern == "dry_land":
        # Simulates massive water loss / desiccation across the entire scene:
        # High Red (band 0), Low Green (band 1), High NIR (band 2) -> Strongly negative NDWI
        data[0] = 230.0   # High Red
        data[1] = 35.0    # Low Green -> NDWI = (35-230)/(265) = -0.736
        data[2] = 220.0   # High NIR/Blue
    elif pattern == "builtup":
        # High reflectance built-up area
        data[:] = 230.0
    else:
        data[:] = 120.0

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(data[i], i + 1)
    return path


def test_task1_vlm_narrative_conversational_and_no_robotic_template() -> None:
    """Task 1: Verify VLM narrative is 5-6 sentences, non-technical, and free of robotic templates."""
    vlm_client = RemoteSensingVLMClient()

    # Test retreat / desiccation narrative
    narrative_retreat = vlm_client._synthesize_temporal_narrative(
        classification="Bi-Temporal Water Body Retreat / Desiccation",
        change_fraction=0.18,
        water_delta=-0.22,
        bbox=[77.50, 12.90, 77.60, 13.00],
        query="What changed regarding the water extent between these two dates?",
    )

    # Must NOT contain robotic template
    assert "Satellite change detection completed for query:" not in narrative_retreat
    assert "Satellite change detection analysis for" not in narrative_retreat
    assert "Newly flooded and water-covered ground" not in narrative_retreat

    # Must contain physical descriptive elements
    assert "retreat" in narrative_retreat.lower() or "dried" in narrative_retreat.lower() or "receded" in narrative_retreat.lower()
    assert "change" in narrative_retreat.lower() or "difference" in narrative_retreat.lower()

    # Sentence count check (5-6 sentences)
    sentences = [s.strip() for s in narrative_retreat.split(".") if len(s.strip()) > 5]
    assert 5 <= len(sentences) <= 6

    # Test urban expansion narrative
    narrative_urban = vlm_client._synthesize_temporal_narrative(
        classification="Bi-Temporal Urban Expansion / Built-up Growth",
        change_fraction=0.12,
        water_delta=0.0,
        bbox=[77.50, 12.90, 77.60, 13.00],
        query="Has the built-up area increased?",
    )
    assert "Satellite change detection completed" not in narrative_urban
    assert "infrastructure" in narrative_urban.lower() or "built-up" in narrative_urban.lower()
    sentences_urban = [s.strip() for s in narrative_urban.split(".") if len(s.strip()) > 5]
    assert 5 <= len(sentences_urban) <= 6


def test_task2_directional_water_retreat_desiccation_classification(tmp_path: Path) -> None:
    """Task 2: When water coverage decreases, classify as Water Retreat / Desiccation, NOT flood."""
    # T1 has water, T2 is dry (water dried up)
    t1 = _create_synthetic_geotiff(tmp_path / "t1_water.tif", pattern="water")
    t2 = _create_synthetic_geotiff(tmp_path / "t2_dry.tif", pattern="dry_land")

    tool = TemporalChangeTool()
    scratchpad: dict = {}
    res = asyncio.run(tool.execute(scratchpad=scratchpad, t1_path=str(t1), t2_path=str(t2), query="Analyze water retreat between dates"))

    assert res["status"] == "success"
    # Directional classification
    assert res["task_classification"] == "Bi-Temporal Water Body Retreat / Desiccation"
    assert "[DECREASED]" in res["directional_verdict"]
    assert res["water_delta"] < 0.0

    # Ensure GeoJSON category is desiccation, NOT flood
    geojson = res["geojson"]
    features = geojson.get("features", [])
    assert len(features) >= 1
    for feat in features:
        props = feat.get("properties", {})
        assert props.get("category") == "desiccation"
        assert props.get("category") != "flood"
        assert "Water Body Retreat" in props.get("label", "")
        assert "Flood" not in props.get("label", "")

    # Scratchpad populated
    assert scratchpad["task_classification"] == "Bi-Temporal Water Body Retreat / Desiccation"


def test_task2_directional_water_expansion_inundation_classification(tmp_path: Path) -> None:
    """Task 2: When water coverage increases, classify as Water Expansion / Inundation (flood)."""
    # T1 is dry land, T2 has water (flooded)
    t1 = _create_synthetic_geotiff(tmp_path / "t1_dry.tif", pattern="dry_land")
    t2 = _create_synthetic_geotiff(tmp_path / "t2_water.tif", pattern="water")

    tool = TemporalChangeTool()
    scratchpad: dict = {}
    res = asyncio.run(tool.execute(scratchpad=scratchpad, t1_path=str(t1), t2_path=str(t2), query="Analyze flooding and water expansion"))

    assert res["status"] == "success"
    assert res["task_classification"] == "Bi-Temporal Water Expansion / Inundation"
    assert "[INCREASED]" in res["directional_verdict"]
    assert res["water_delta"] > 0.0

    # GeoJSON category is flood
    geojson = res["geojson"]
    features = geojson.get("features", [])
    assert len(features) >= 1
    for feat in features:
        props = feat.get("properties", {})
        assert props.get("category") == "flood"


def test_task3_affine_matrix_scaling_with_mismatched_mask_dimensions(tmp_path: Path) -> None:
    """Task 3: Affine scaling correctly handles masks whose dimensions differ from GeoTIFF."""
    # GeoTIFF is 64x64
    tiff_path = _create_synthetic_geotiff(
        tmp_path / "geom_test.tif",
        bounds=(77.50, 12.90, 77.60, 13.00),
        shape=(64, 64),
    )

    # Mask is 32x32 (downsampled feature map)
    mask_small = np.zeros((32, 32), dtype=np.uint8)
    mask_small[8:24, 8:24] = 1

    geojson_scaled = raster_mask_to_geojson(
        geotiff_path=tiff_path,
        mask=mask_small,
        task_type="change_detection",
        label="Scaled Target",
        category="change_detection",
    )

    features = geojson_scaled.get("features", [])
    assert len(features) >= 1
    geom = features[0]["geometry"]
    geom_obj = shape(geom)
    minx, miny, maxx, maxy = geom_obj.bounds

    # Verify geometry bounding box lies within the GeoTIFF WGS84 bounding box [77.50, 12.90, 77.60, 13.00]
    assert 77.49 <= minx <= 77.61
    assert 77.49 <= maxx <= 77.61
    assert 12.89 <= miny <= 13.01
    assert 12.89 <= maxy <= 13.01
    assert minx < maxx
    assert miny < maxy

    props = features[0]["properties"]
    assert props["label"] == "Scaled Target"
    assert props["area_m2"] >= 0.0


def test_task4_raster_preview_generation(tmp_path: Path) -> None:
    """Task 4: Generate 8-bit normalized Base64 PNG preview and Leaflet bounds."""
    tiff_path = _create_synthetic_geotiff(
        tmp_path / "preview_sample.tif",
        bounds=(77.50, 12.90, 77.60, 13.00),
        shape=(64, 64),
        bands=3,
    )

    preview_uri, bounds = generate_raster_preview(tiff_path)
    assert preview_uri is not None
    assert preview_uri.startswith("data:image/png;base64,")

    # Decode and verify it's a valid PNG
    b64_data = preview_uri.replace("data:image/png;base64,", "")
    img_bytes = base64.b64decode(b64_data)
    assert len(img_bytes) > 50

    # Leaflet bounds must be [[south, west], [north, east]]
    assert bounds is not None
    assert len(bounds) == 2
    south_west, north_east = bounds
    assert south_west[0] < north_east[0]  # south < north (lat)
    assert south_west[1] < north_east[1]  # west < east (lon)
    assert pytest.approx(south_west[0], abs=0.01) == 12.90
    assert pytest.approx(south_west[1], abs=0.01) == 77.50
    assert pytest.approx(north_east[0], abs=0.01) == 13.00
    assert pytest.approx(north_east[1], abs=0.01) == 77.60

    # Convenience helpers
    uri_direct = get_raster_base64_preview(tiff_path)
    bounds_direct = get_leaflet_bounds(tiff_path)
    assert uri_direct == preview_uri
    assert bounds_direct == bounds


def test_task4_temporal_tool_and_response_envelope_contain_previews(tmp_path: Path) -> None:
    """Task 4: Verify TemporalChangeTool and QueryResponseEnvelope contain preview URLs and bounds."""
    t1 = _create_synthetic_geotiff(tmp_path / "prev_t1.tif", pattern="water")
    t2 = _create_synthetic_geotiff(tmp_path / "prev_t2.tif", pattern="dry_land")

    tool = TemporalChangeTool()
    scratchpad: dict = {}
    res = asyncio.run(tool.execute(scratchpad=scratchpad, t1_path=str(t1), t2_path=str(t2), query="Water retreat analysis"))

    assert "t1_preview_url" in res
    assert res["t1_preview_url"].startswith("data:image/png;base64,")
    assert "t2_preview_url" in res
    assert res["t2_preview_url"].startswith("data:image/png;base64,")
    assert "leaflet_bounds" in res
    assert len(res["leaflet_bounds"]) == 2

    # Verify QueryResponseEnvelope schema validation with previews
    envelope = QueryResponseEnvelope(
        status="ok",
        answer=res["answer"],
        task_type="bitemporal_change",
        headline="Temporal Analysis",
        models_executed=["SiameseChangeNet", "RemoteSensingVLMClient"],
        confidence=res["confidence"],
        geojson=res["geojson"],
        t1_preview_url=res["t1_preview_url"],
        t2_preview_url=res["t2_preview_url"],
        leaflet_bounds=res["leaflet_bounds"],
        audit_summary={"crs": "EPSG:4326"},
        trace={},
    )
    dumped = envelope.model_dump()
    assert dumped["t1_preview_url"].startswith("data:image/png;base64,")
    assert dumped["t2_preview_url"].startswith("data:image/png;base64,")
    assert dumped["leaflet_bounds"] is not None
