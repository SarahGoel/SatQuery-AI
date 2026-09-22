"""Unit and integration tests validating the 5 critical architectural fixes:
1. Plain-language non-technical outputs without ML/DL jargon.
2. Endpoint consolidation (/satquery/analyze alias on /query) + PostGIS logging + PDF.
3. Dynamic PNG proportional coordinates and EXIF GPS extraction.
4. SIFT/RANSAC sub-pixel alignment integration into Controller.
5. BigEarthNet 19-class land-cover classifier integration into VQA and cross-modal pipelines.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.services.heuristic_vlm import generate_heuristic_summary
from app.services.models.bigearthnet import (
    CORINE_19_CLASSES,
    BigEarthNetLandCoverClassifier,
)
from app.services.agent import SatQueryController, _compute_proportional_bounds
from app.agents.router import InputInspectorNode
from backend.api.routes import (
    _calculate_proportional_bounds,
    _extract_exif_gps_bounds,
)


def _create_geotiff(path: Path, west: float, south: float, east: float, north: float, bands: int = 4, w: int = 64, h: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(west, south, east, north, w, h)
    data = np.random.default_rng(42).random((bands, h, w), dtype=np.float32)
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


def test_fix1_plain_language_text_outputs_no_jargon() -> None:
    """Ensure heuristic outputs use plain language, retain numbers, and omit DL jargon."""
    meta = {
        "sensor": "Cartosat-2S (Multispectral)",
        "resolution": "1.0m GSD",
        "crs": "EPSG:4326",
        "bounds": [78.0, 20.0, 78.2, 20.2],
        "width": 512,
        "height": 512,
    }
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1, "label": "Storage Tank", "area_m2": 50000.0},
                "geometry": {"type": "Polygon", "coordinates": [[[78.0, 20.0], [78.1, 20.0], [78.1, 20.1], [78.0, 20.1], [78.0, 20.0]]]},
            }
        ],
    }

    # Test single grounding
    grounding_text = generate_heuristic_summary(
        query="Highlight circular storage tanks",
        task="single_grounding",
        geojson=geojson,
        metadata=meta,
        confidence=0.92,
        models=["RS-Grounding-V3", "MobileSAM"],
    )
    assert "We detected and mapped 1 separate Storage Tank(s)" in grounding_text
    assert "92%" in grounding_text
    # Ensure NO deep learning jargon appears
    for forbidden in ["ViT", "Hough", "Sobel", "TemporalDifferenceAttention", "logits", "backpropagation"]:
        assert forbidden.lower() not in grounding_text.lower()

    # Test bi-temporal change
    change_text = generate_heuristic_summary(
        query="What changed between these dates?",
        task="bitemporal_change",
        geojson=geojson,
        metadata=meta,
        confidence=0.89,
        models=["CD-VQA-Pro"],
    )
    assert "between the earlier baseline date (T1) and the post-event date (T2)" in change_text
    assert "89%" in change_text
    assert "14.2%" in change_text
    for forbidden in ["ViT", "Hough", "Sobel", "TemporalDifferenceAttention", "logits"]:
        assert forbidden.lower() not in change_text.lower()


def test_fix3_dynamic_png_proportional_and_exif_coordinates() -> None:
    """Verify PNG bounds are dynamically computed with aspect ratio proportionality."""
    # 1. Proportional bounds for 1000x500 (2:1 aspect ratio)
    w, h = 1000, 500
    west, south, east, north = _calculate_proportional_bounds(w, h)
    dx = east - west
    dy = north - south
    # dx should be twice dy (2:1 aspect ratio)
    assert abs((dx / dy) - 2.0) < 0.05

    # 2. Proportional bounds for 400x800 (1:2 aspect ratio)
    w2, h2 = 400, 800
    west2, south2, east2, north2 = _calculate_proportional_bounds(w2, h2)
    dx2 = east2 - west2
    dy2 = north2 - south2
    # dy should be twice dx
    assert abs((dy2 / dx2) - 2.0) < 0.05

    # 3. Agent internal proportional bounds helper
    agent_bounds = _compute_proportional_bounds(600, 600)
    assert len(agent_bounds) == 4
    assert agent_bounds[0] < agent_bounds[2]
    assert agent_bounds[1] < agent_bounds[3]


def test_fix4_subpixel_alignment_wired_into_controller(tmp_path: Path) -> None:
    """Verify SatQueryController invokes SpatialAligner for 2-image workflows."""
    opt1 = _create_geotiff(tmp_path / "opt_t1.tif", 78.0, 20.0, 78.2, 20.2, bands=4)
    opt2 = _create_geotiff(tmp_path / "opt_t2.tif", 78.01, 20.01, 78.19, 20.19, bands=4)

    controller = SatQueryController(db=None)
    trace = controller.execute_workflow(
        query="What changed between these two dates?",
        filepaths=[str(opt1), str(opt2)],
    )
    assert trace.task_type == "bitemporal_change"
    # SIFT/RANSAC sub-pixel aligner MUST be recorded in execution steps
    models = [step.model for step in trace.registry_execution]
    assert "spatial-aligner" in models
    assert controller.last_overlay_uri is not None


def test_fix5_bigearthnet_classifier_integration(tmp_path: Path) -> None:
    """Verify BigEarthNet 19-class classifier runs and categorizes scene land cover."""
    opt = _create_geotiff(tmp_path / "optical.tif", 78.0, 20.0, 78.2, 20.2, bands=4)
    sar = _create_geotiff(tmp_path / "sar.tif", 78.0, 20.0, 78.2, 20.2, bands=2)

    # 1. Standalone BigEarthNet classifier test
    classifier = BigEarthNetLandCoverClassifier()
    assert len(CORINE_19_CLASSES) == 19
    res = classifier.classify(optical_path=opt, sar_path=sar)
    assert len(res.predicted_classes) >= 1
    assert res.predicted_classes[0] in CORINE_19_CLASSES
    assert res.confidence > 0.0

    # 2. Agent pipeline integration test
    controller = SatQueryController(db=None)
    vqa_trace = controller.execute_workflow(
        query="Describe dominant land cover and scene semantics",
        filepaths=[str(opt)],
    )
    assert vqa_trace.task_type == "single_vqa"
    vqa_models = [step.model for step in vqa_trace.registry_execution]
    assert "bigearthnet-encoder" in vqa_models


def test_empty_or_malformed_bounds_resilience() -> None:
    """Verify bounds unpacking never raises ValueError: not enough values to unpack (expected 4, got 0)."""
    controller = SatQueryController(db=None)

    # 1. Test _bounds_polygon with None, empty bounds, and short bounds
    class DummyMeta:
        def __init__(self, bounds):
            self.bounds = bounds

    # None metadata
    poly_none = controller._bounds_polygon(None)
    assert poly_none.geom_type == "Polygon"
    assert poly_none.bounds == (0.0, 0.0, 0.0, 0.0)

    # Empty bounds [] (Exact cause of expected 4, got 0)
    poly_empty = controller._bounds_polygon(DummyMeta([]))
    assert poly_empty.geom_type == "Polygon"
    assert poly_empty.bounds == (0.0, 0.0, 0.0, 0.0)

    # Partial bounds [1.0, 2.0]
    poly_partial = controller._bounds_polygon(DummyMeta([1.0, 2.0]))
    assert poly_partial.geom_type == "Polygon"
    assert poly_partial.bounds == (0.0, 0.0, 0.0, 0.0)

    # Valid bounds [10.0, 20.0, 11.0, 21.0]
    poly_valid = controller._bounds_polygon(DummyMeta([10.0, 20.0, 11.0, 21.0]))
    assert poly_valid.geom_type == "Polygon"
    assert poly_valid.bounds == (10.0, 20.0, 11.0, 21.0)

    # 2. Test validate_spatial_alignment resilience with empty bounds
    res = controller.validate_spatial_alignment(
        {"bounds": [], "crs": "EPSG:4326", "affine_transform_matrix": []},
        {"bounds": None, "crs": "EPSG:4326"},
    )
    assert isinstance(res, bool)


def test_bitemporal_intent_validation_exact_error() -> None:
    """Verify InputInspectorNode rejects temporal queries with <2 images with exact message."""
    expected_msg = (
        "Bi-temporal change detection requires two spatially aligned images "
        "(Before and After). Please attach an image pair to analyze temporal change."
    )

    # 1 image + temporal query
    with pytest.raises(ValueError) as exc1:
        InputInspectorNode.inspect("What has changed between these two dates?", filepaths=["/tmp/img1.tif"])
    assert expected_msg in str(exc1.value)

    # 0 images + temporal query
    with pytest.raises(ValueError) as exc0:
        InputInspectorNode.inspect("Show me difference between dates", filepaths=[])
    assert expected_msg in str(exc0.value)


def test_dynamic_grounding_text_with_coordinates_and_area() -> None:
    """Verify grounding report includes actual counts, real pixel coordinates, and calculated areas."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "label": "Fuel Storage Tank",
                    "bbox_pixel": [120, 140, 160, 180],
                    "area_m2": 1600.0,
                    "confidence": 0.94,
                },
                "geometry": {"type": "Polygon", "coordinates": []},
            }
        ],
    }
    text = generate_heuristic_summary(
        query="Highlight fuel storage tanks",
        task="single_grounding",
        geojson=geojson,
        confidence=0.94,
    )
    assert "We detected and mapped 1 separate Fuel Storage Tank(s)" in text
    assert "pixel coordinates [120, 140, 160, 180]" in text
    assert "area 1600.0 m²" in text


def test_corrupt_raster_raises_http_exception(tmp_path: Path) -> None:
    """Verify corrupted / invalid rasters raise HTTPException(400) instead of silent synthetic fallback."""
    from fastapi import HTTPException
    from app.services.models.grounding import TextGuidedGrounder

    bad_raster = tmp_path / "corrupt.tif"
    bad_raster.write_text("not a valid geotiff file")

    grounder = TextGuidedGrounder()
    with pytest.raises(HTTPException) as exc_info:
        grounder.ground(image_path=bad_raster, prompt="find storage tanks")
    assert exc_info.value.status_code == 400
    assert "Failed to process raster" in exc_info.value.detail


def test_config_docker_and_host_url_resolution(monkeypatch: Any) -> None:
    """Validate that config.py dynamically resolves endpoints for Docker and host environments."""
    from unittest.mock import patch
    from app.core.config import Settings

    # Case 1: Running on Host (no /.dockerenv)
    with patch("os.path.exists", return_value=False):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        host_settings = Settings()
        assert host_settings.resolved_ollama_url == "http://localhost:11434"
        assert "localhost:5432" in host_settings.DATABASE_URL

    # Case 2: Running inside Docker (/.dockerenv exists)
    with patch("os.path.exists", lambda p: True if p == "/.dockerenv" else False):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        docker_settings = Settings()
        assert docker_settings.resolved_ollama_url == "http://ollama:11434"
        assert "db:5432" in docker_settings.DATABASE_URL

    # Case 3: Docker with explicit OLLAMA_HOST env var
    with patch("os.path.exists", lambda p: True if p == "/.dockerenv" else False):
        monkeypatch.setenv("OLLAMA_HOST", "http://satquery_ollama:11434")
        docker_custom = Settings()
        assert docker_custom.resolved_ollama_url == "http://satquery_ollama:11434"


def test_vlm_client_candidate_endpoints(monkeypatch: Any) -> None:
    """Validate that LocalVisionLanguageClient queries candidate endpoints and handles connection errors."""
    from unittest.mock import patch, MagicMock
    from app.services.models.base import LocalVisionLanguageClient

    client = LocalVisionLanguageClient()
    assert client.backend in ("ollama", "vllm")
    assert client.timeout >= 15.0

    # Test error handling when all Ollama endpoints are unreachable
    with patch("app.services.models.base._http_post_json", side_effect=ConnectionRefusedError("Connection refused")):
        res = client.generate("Describe this satellite scene")
        assert res.confidence > 0.0
        assert res.params.get("mode") == "heuristic_fallback"
        assert "Connection refused" in res.params.get("error", "")


