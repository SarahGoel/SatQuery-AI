"""Tests for Phase 2.5 — Deep Inference, Tensors & ISRO-Ready Fallbacks.

Validates:
1. GeoChat coordinate parsing & rasterio affine transformation to WGS84 GeoJSON.
2. Graceful fallback to ISRO SAC coordinates for non-georeferenced imagery (PNG/JPEG) or missing CRS.
3. RemoteCLIP temporal differencing with PyTorch tensors and graceful fallback when weights are missing.
4. Mathematical verification of tensor differencing (cosine distance / thresholding).
5. BigEarthNet LoRA adapter loading and graceful fallback in RemoteSensingVLMClient.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_bounds
import torch

from app.services.geospatial_parser import (
    ISRO_SAC_LAT,
    ISRO_SAC_LON,
    extract_and_transform_bbox,
    parse_geochat_bbox,
)
from app.services.models.base import VLMResult
from app.services.models.rs_vlm import RemoteSensingVLMClient
from app.tools.registry import default_tool_registry
from app.tools.temporal_change import RemoteCLIPTemporalEncoder, TemporalChangeTool


def _create_synthetic_geotiff(
    path: Path,
    west: float = 77.0,
    south: float = 28.0,
    east: float = 77.2,
    north: float = 28.2,
    width: int = 64,
    height: int = 64,
    count: int = 3,
    fill_value: float = 0.5,
) -> Path:
    """Helper to generate a lightweight synthetic GeoTIFF with EPSG:4326 CRS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(west, south, east, north, width, height)
    data = np.ones((count, height, width), dtype=np.float32) * fill_value

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


def _create_synthetic_png(path: Path, width: int = 64, height: int = 64) -> Path:
    """Helper to generate an ordinary non-georeferenced RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.ones((height, width, 3)) * 128).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    return path


# ============================================================================
# 1. GeoChat Coordinate Extractor & Geospatial Transformation Tests
# ============================================================================

class TestGeospatialParser:
    """Validates regex extraction of bounding boxes and affine conversion to WGS84 GeoJSON."""

    def test_parse_geochat_bbox_standard_tag(self) -> None:
        text = "Identified oil storage facility at <box>[150, 250, 600, 700]</box> in north sector."
        bbox = parse_geochat_bbox(text)
        assert bbox is not None
        assert bbox == [150.0, 250.0, 600.0, 700.0]

    def test_parse_geochat_bbox_brackets_without_tags(self) -> None:
        text = "A runway is located at coordinates [100, 200, 400, 800]."
        bbox = parse_geochat_bbox(text)
        assert bbox is not None
        assert bbox == [100.0, 200.0, 400.0, 800.0]

    def test_parse_geochat_bbox_parentheses(self) -> None:
        text = "Aircraft detected at (50, 75, 120, 180)."
        bbox = parse_geochat_bbox(text)
        assert bbox is not None
        assert bbox == [50.0, 75.0, 120.0, 180.0]

    def test_parse_geochat_bbox_inversion_and_clipping(self) -> None:
        # Inverted coordinates: ymin > ymax, xmin > xmax
        text = "Structure detected at <box>[900, 800, 200, 100]</box>."
        bbox = parse_geochat_bbox(text)
        assert bbox is not None
        assert bbox[0] <= bbox[2]  # ymin <= ymax
        assert bbox[1] <= bbox[3]  # xmin <= xmax
        assert bbox == [200.0, 100.0, 900.0, 800.0]

    def test_parse_geochat_bbox_no_coordinates(self) -> None:
        text = "The satellite scene shows agricultural land with no distinct infrastructure."
        bbox = parse_geochat_bbox(text)
        assert bbox is None

    def test_extract_and_transform_bbox_with_geotiff(self, tmp_path: Path) -> None:
        geotiff_path = _create_synthetic_geotiff(
            tmp_path / "scene.tif",
            west=77.0,
            south=28.0,
            east=77.2,
            north=28.2,
            width=100,
            height=100,
        )
        text = "Bridge infrastructure identified at <box>[200, 300, 600, 700]</box>."

        result = extract_and_transform_bbox(text, geotiff_path)

        assert result["status"] == "success"
        assert result["method"] == "geochat_affine_transform"
        assert result["bbox"] == [200.0, 300.0, 600.0, 700.0]
        # Pixel coordinates: 200/1000*100=20, 300/1000*100=30, 600/1000*100=60, 700/1000*100=70
        assert result["pixel_bbox"] == [20, 30, 60, 70]

        geojson = result["geojson"]
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1

        feature = geojson["features"][0]
        assert feature["geometry"]["type"] == "Polygon"
        ring = feature["geometry"]["coordinates"][0]
        assert len(ring) == 5  # Closed 5-point polygon
        assert ring[0] == ring[-1]

        # Verify coordinates lie within the GeoTIFF geographic extent
        for lon, lat in ring:
            assert 77.0 <= lon <= 77.2
            assert 28.0 <= lat <= 28.2

        assert feature["properties"]["class"] == "visual_grounding"
        assert feature["properties"]["source"] == "geochat_vlm"
        assert feature["properties"]["crs"] == "EPSG:4326"

    def test_extract_and_transform_bbox_non_georeferenced_png(self, tmp_path: Path) -> None:
        png_path = _create_synthetic_png(tmp_path / "photo.png", width=128, height=128)
        text = "Target located at <box>[100, 200, 300, 400]</box>."

        result = extract_and_transform_bbox(text, png_path)

        assert result["status"] == "success"
        assert result["method"] == "synthetic_isro_sac"
        geojson = result["geojson"]
        assert geojson["type"] == "FeatureCollection"
        ring = geojson["features"][0]["geometry"]["coordinates"][0]

        # Verify centered around ISRO SAC coordinates
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        assert min(lons) <= ISRO_SAC_LON <= max(lons)
        assert min(lats) <= ISRO_SAC_LAT <= max(lats)

    def test_extract_and_transform_bbox_missing_coords_fallback(self, tmp_path: Path) -> None:
        geotiff_path = _create_synthetic_geotiff(tmp_path / "scene_nocoord.tif")
        text = "No bounding box present in this output."

        result = extract_and_transform_bbox(text, geotiff_path)

        assert result["status"] == "success"
        assert result["method"] == "synthetic_isro_sac"
        assert result["bbox"] is None
        assert "geojson" in result


# ============================================================================
# 2. RemoteCLIP Temporal Differencing & Fallback Tests
# ============================================================================

class TestRemoteCLIPTemporalDifferencing:
    """Validates PyTorch tensor differencing with RemoteCLIP and graceful fallbacks."""

    def test_remoteclip_encoder_missing_weights_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError):
            RemoteCLIPTemporalEncoder(weights_path="non_existent_weights_path.pt")

    def test_temporal_change_tool_fallback_when_weights_missing(self, tmp_path: Path) -> None:
        t1 = _create_synthetic_geotiff(tmp_path / "t1.tif", fill_value=0.2)
        t2 = _create_synthetic_geotiff(tmp_path / "t2.tif", fill_value=0.8)

        tool = TemporalChangeTool()
        scratchpad: Dict[str, Any] = {"filepaths": [str(t1), str(t2)]}

        # Execute tool without RemoteCLIP weights present (synchronous runner)
        res = tool.run(scratchpad, t1_path=str(t1), t2_path=str(t2))


        assert res["status"] == "success"
        assert res["tool"] == "TemporalChangeTool"
        assert "change_mask" in res
        assert "change_pixel_count" in res
        assert "geojson" in res
        assert res["geojson"]["type"] == "FeatureCollection"
        assert any(keyword in res["directional_verdict"] for keyword in ["INCREASED", "DECREASED", "REMAINED UNCHANGED"])

        # Check scratchpad was updated
        assert scratchpad["change_pixel_count"] == res["change_pixel_count"]
        assert scratchpad["geojson"] == res["geojson"]

    def test_remoteclip_tensor_differencing_math(self, tmp_path: Path) -> None:
        """Validates mathematical difference between identical vs altered PyTorch image tensors."""
        t1_path = _create_synthetic_geotiff(tmp_path / "baseline.tif", fill_value=0.1)
        t2_path_same = _create_synthetic_geotiff(tmp_path / "same.tif", fill_value=0.1)
        t2_path_diff = _create_synthetic_geotiff(tmp_path / "diff.tif", fill_value=0.1)

        # Alter region in t2_path_diff to simulate real-world surface changes
        with rasterio.open(t2_path_diff, "r+") as ds:
            arr = ds.read()
            arr[:, 10:30, 10:30] = 0.95
            ds.write(arr)

        # Create a mock RemoteCLIP vision encoder model
        class MockEncoder(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Spatial vision embedding
                return x * 1.5

        # Instantiate encoder with mocked _load_model
        with patch.object(RemoteCLIPTemporalEncoder, "_load_model", return_value=None):
            encoder = RemoteCLIPTemporalEncoder(weights_path=t1_path)
            encoder.model = MockEncoder()
            encoder.device = "cpu"

            # Case A: Identical scenes -> 0 change
            mask_same, stats_same = encoder.compute_tensor_difference(t1_path, t2_path_same, threshold=0.05)
            assert int(mask_same.sum()) == 0
            assert stats_same["change_fraction"] == 0.0

            # Case B: Disparate scenes -> change detected
            mask_diff, stats_diff = encoder.compute_tensor_difference(t1_path, t2_path_diff, threshold=0.05)
            assert int(mask_diff.sum()) > 0
            assert stats_diff["change_fraction"] > 0.0
            assert stats_diff["tensor_distance_mean"] > 0.0
            assert "[INCREASED]" in stats_diff["directional_verdict"]


    def test_tool_registry_resolves_temporal_change_tool(self) -> None:
        tool = default_tool_registry.get("TemporalChangeTool")
        assert tool is not None
        assert isinstance(tool, TemporalChangeTool)

        snake_tool = default_tool_registry.get("temporal_change_tool")
        assert snake_tool is not None
        assert isinstance(snake_tool, TemporalChangeTool)


# ============================================================================
# 3. BigEarthNet LoRA Adapter Loading & VLM Grounding Tests
# ============================================================================

class TestRemoteSensingVLMLoRA:
    """Validates BigEarthNet LoRA adapter loading and resilient fallbacks in RemoteSensingVLMClient."""

    def test_lora_adapter_missing_weights_fallback(self) -> None:
        with patch.dict(os.environ, {"BIGEARTHNET_LORA_PATH": "missing/lora/checkpoint.pt"}):
            client = RemoteSensingVLMClient()
            assert client.lora_loaded is False
            assert client.lora_adapter_name is None

            # Test generation fallback
            res = client.generate("Describe scene")
            assert res.confidence > 0
            assert res.params.get("lora_loaded") is False
            assert res.params.get("lora_adapter") is None

    def test_lora_adapter_loading_success_from_checkpoint(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "bigearthnet_adapter.pt"
        torch.save({"lora_state_dummy": torch.zeros((2, 2))}, ckpt_path)

        client = RemoteSensingVLMClient()
        loaded = client.load_lora_adapter(ckpt_path)

        assert loaded is True
        assert client.lora_loaded is True
        assert client.lora_adapter_name == "bigearthnet_adapter"

        res = client.generate("Describe scene")
        assert res.params.get("lora_loaded") is True
        assert res.params.get("lora_adapter") == "bigearthnet_adapter"

    def test_generate_grounding_attaches_parsed_bbox_geojson(self, tmp_path: Path) -> None:
        geotiff = _create_synthetic_geotiff(
            tmp_path / "grounding_test.tif",
            west=80.0,
            south=13.0,
            east=80.1,
            north=13.1,
        )

        client = RemoteSensingVLMClient()
        mock_vlm_output = VLMResult(
            text="Detected maritime vessel at <box>[250, 250, 750, 750]</box> in port waters.",
            confidence=0.94,
            params={},
        )

        with patch.object(client, "generate", return_value=mock_vlm_output):
            result = client.generate_grounding("maritime vessel", geotiff)

            assert result.params.get("bounding_box") == [250.0, 250.0, 750.0, 750.0]
            assert "geojson" in result.params
            geojson = result.params["geojson"]
            assert geojson["type"] == "FeatureCollection"
            assert len(geojson["features"]) == 1
            feature = geojson["features"][0]
            assert feature["geometry"]["type"] == "Polygon"
            # Coordinates within bounds
            for lon, lat in feature["geometry"]["coordinates"][0]:
                assert 80.0 <= lon <= 80.1
                assert 13.0 <= lat <= 13.1
