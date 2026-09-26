"""Unit tests for Neural VLM LoRA Adapter Integration & Spatial Token Grounding.

Verifies:
1. Parsing and stripping of Qwen2-VL <box>[ymin, xmin, ymax, xmax]</box> coordinate tokens.
2. Accurate normalization to image pixel space.
3. GeoTIFF affine projection to closed WGS84 GeoJSON polygons.
4. RemoteSensingVLMClient detection of fine-tuned adapter weights at backend/local_models/vlm_lora/.
5. Graceful fallback without unhandled exceptions.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.services.geospatial_parser import (
    extract_and_transform_bbox,
    extract_vlm_spatial_tokens,
)
from app.services.models.base import VLMResult
from app.services.models.rs_vlm import RemoteSensingVLMClient


def _create_synthetic_geotiff(
    path: Path,
    west: float = 77.0,
    south: float = 28.0,
    east: float = 77.1,
    north: float = 28.1,
    width: int = 256,
    height: int = 256,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(west, south, east, north, width, height)
    data = np.ones((3, height, width), dtype=np.uint8) * 120
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as ds:
        ds.write(data)
    return path


def test_spatial_token_parsing_and_strip() -> None:
    """Verifies that text containing <box>[145, 230, 480, 610]</box> strips the tag cleanly and computes accurate pixel bounds."""
    raw_text = (
        "Detected industrial fuel storage tanks at <box>[145, 230, 480, 610]</box> "
        "within the southern perimeter."
    )

    cleaned_text, boxes = extract_vlm_spatial_tokens(raw_text, img_width=1000, img_height=1000)

    # 1. Clean natural language prose: tags stripped
    assert "<box>" not in cleaned_text
    assert "</box>" not in cleaned_text
    assert "Detected industrial fuel storage tanks at within the southern perimeter." in cleaned_text or (
        "Detected industrial fuel storage tanks" in cleaned_text and "within the southern perimeter." in cleaned_text
    )

    # 2. Accurate pixel bounds calculation
    # Qwen2-VL format: [ymin, xmin, ymax, xmax] -> y1=145, x1=230, y2=480, x2=610
    # xmin = (230/1000)*1000 = 230.0, ymin = (145/1000)*1000 = 145.0
    # xmax = (610/1000)*1000 = 610.0, ymax = (480/1000)*1000 = 480.0
    assert len(boxes) == 1
    assert boxes[0] == [230.0, 145.0, 610.0, 480.0]

    # Test with custom image dimensions: W=500, H=800
    cleaned_500, boxes_500 = extract_vlm_spatial_tokens(raw_text, img_width=500, img_height=800)
    assert len(boxes_500) == 1
    assert boxes_500[0][0] == pytest.approx(115.0, abs=1e-2)  # xmin: 230/1000 * 500 = 115.0
    assert boxes_500[0][1] == pytest.approx(116.0, abs=1e-2)  # ymin: 145/1000 * 800 = 116.0
    assert boxes_500[0][2] == pytest.approx(305.0, abs=1e-2)  # xmax: 610/1000 * 500 = 305.0
    assert boxes_500[0][3] == pytest.approx(384.0, abs=1e-2)  # ymax: 480/1000 * 800 = 384.0


def test_vlm_lora_path_detection() -> None:
    """Asserts that RemoteSensingVLMClient detects the adapter files in backend/local_models/vlm_lora/."""
    client = RemoteSensingVLMClient()

    # Verify detected path exists and contains required adapter weights
    assert client.vlm_lora_path is not None
    assert client.vlm_lora_path.exists()
    assert (client.vlm_lora_path / "adapter_model.safetensors").exists()
    assert (client.vlm_lora_path / "adapter_config.json").exists()
    assert client.vlm_lora_detected is True
    assert client.vlm_lora_adapter_name == "vlm_lora"


def test_extract_and_transform_bbox_spatial_token_projection(tmp_path: Path) -> None:
    """Verifies affine transform projects spatial tokens into a 5-coordinate closed WGS84 polygon ring."""
    geotiff_path = _create_synthetic_geotiff(
        tmp_path / "lora_grounding.tif",
        west=77.0,
        south=28.0,
        east=77.1,
        north=28.1,
        width=256,
        height=256,
    )
    raw_vlm_response = "Target identified at <box>[145, 230, 480, 610]</box> in central quadrant."

    res = extract_and_transform_bbox(raw_vlm_response, geotiff_path)

    assert res["status"] == "success"
    assert res["method"] == "geochat_affine_transform"
    assert "cleaned_text" in res
    assert "<box>" not in res["cleaned_text"]

    geojson = res["geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1

    feature = geojson["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    ring = feature["geometry"]["coordinates"][0]

    # Verify 5-coordinate closed polygon ring
    assert len(ring) == 5
    assert ring[0] == ring[-1]

    # Verify all coordinates lie within the GeoTIFF spatial bounding box
    for lon, lat in ring:
        assert 77.0 <= lon <= 77.1
        assert 28.0 <= lat <= 28.1


def test_generate_response_graceful_fallback(tmp_path: Path) -> None:
    """Verifies that generate_response executes and returns a valid VLMResult without unhandled exceptions."""
    geotiff_path = _create_synthetic_geotiff(tmp_path / "fallback_sample.tif")
    client = RemoteSensingVLMClient()

    # Should run and return a valid result without throwing exceptions
    result = client.generate_response(
        prompt="Analyze surface land cover and infrastructure.",
        image_path=geotiff_path,
    )

    assert isinstance(result, VLMResult)
    assert result.confidence > 0.0
    assert result.text is not None and len(result.text) > 0
    assert "vlm_lora_detected" in result.params
    assert result.params["vlm_lora_detected"] is True
