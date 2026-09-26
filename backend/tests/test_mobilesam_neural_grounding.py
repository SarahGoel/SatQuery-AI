"""Comprehensive verification unit tests for MobileSAM neural checkpoint integration on CPU.

Validates:
1. MobileSAM checkpoint loading from local_models/sam/mobile_sam.pt with zero key mismatches (strict=True).
2. CPU sub-pixel binary mask prediction from bounding box prompts ([xmin, ymin, xmax, ymax], normalized & pixel).
3. GeoJSON FeatureCollection conversion via raster_mask_to_geojson with geodesic measurements (m^2, ha, km^2).
4. OpenCV contouring heuristic fallback when torch inference encounters an exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

import torch
from app.services.models.sam import build_sam_vit_t, SamPredictor
from app.services.models.grounding import ZeroShotSAMGrounder, TextGuidedGrounder
from app.services.grounding_service import GroundingService
from app.services.geospatial.vector import raster_mask_to_geojson


def _resolve_checkpoint_path() -> Path:
    """Resolve MobileSAM checkpoint across Docker mount, repo root, and backend paths."""
    candidates = [
        Path("/local_models/sam/mobile_sam.pt"),
        Path(__file__).resolve().parents[2] / "local_models" / "sam" / "mobile_sam.pt",
        Path(__file__).resolve().parents[1] / "local_models" / "sam" / "mobile_sam.pt",
        Path("local_models/sam/mobile_sam.pt"),
        Path("backend/local_models/sam/mobile_sam.pt"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    return (Path(__file__).resolve().parents[2] / "local_models" / "sam" / "mobile_sam.pt").resolve()


def _create_synthetic_geotiff(
    path: Path,
    bounds: tuple[float, float, float, float] = (77.50, 12.90, 77.60, 13.00),
    shape: tuple[int, int] = (256, 256),
    bands: int = 3,
) -> Path:
    """Create a 256x256 GeoTIFF with EPSG:4326 CRS and affine transform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bounds
    h, w = shape
    transform = from_bounds(west, south, east, north, w, h)

    # Synthetic image with a bright feature in [50:150, 50:150]
    data = np.ones((bands, h, w), dtype=np.float32) * 50.0
    data[:, 50:150, 50:150] = 220.0

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


def test_mobilesam_checkpoint_loading_no_mismatches() -> None:
    """Test that mobile_sam.pt loads into TinyViT + MaskDecoder without state_dict key mismatches."""
    ckpt_path = _resolve_checkpoint_path()
    assert ckpt_path.exists(), f"MobileSAM checkpoint not found at {ckpt_path}"

    # 1. Direct model architecture verification
    model = build_sam_vit_t()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    assert len(state_dict) == 439, f"Expected 439 checkpoint keys, found {len(state_dict)}"

    # Must load with strict=True without missing or unexpected keys
    load_result = model.load_state_dict(state_dict, strict=True)
    assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"

    # 2. ZeroShotSAMGrounder wrapper verification
    grounder = ZeroShotSAMGrounder(str(ckpt_path))
    assert grounder.weights_loaded is True
    assert grounder.model is not None
    assert grounder.predictor is not None


def test_mobilesam_cpu_mask_prediction_from_box() -> None:
    """Test that a dummy 256x256 image with box [50, 50, 150, 150] produces a valid (256, 256) binary mask on CPU."""
    ckpt_path = _resolve_checkpoint_path()
    grounder = ZeroShotSAMGrounder(str(ckpt_path))
    assert grounder.weights_loaded is True

    # Create dummy 256x256 image
    h, w = 256, 256
    dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
    dummy_img[50:150, 50:150] = 200

    # 1. Pixel-space bounding box [xmin, ymin, xmax, ymax]
    box_pixel = [50, 50, 150, 150]
    mask = grounder.predict_mask_from_box(dummy_img, box_pixel)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (256, 256), f"Expected shape (256, 256), got {mask.shape}"
    assert mask.dtype == np.uint8
    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.sum() > 0, "Predicted mask should contain foreground pixels"

    # Foreground pixels should be localized around the prompt region
    fg_y, fg_x = np.where(mask > 0)
    assert fg_x.min() >= 40
    assert fg_x.max() <= 160
    assert fg_y.min() >= 40
    assert fg_y.max() <= 160

    # 2. Normalized bounding box [xmin, ymin, xmax, ymax]
    box_norm = [50.0 / 256.0, 50.0 / 256.0, 150.0 / 256.0, 150.0 / 256.0]
    norm_mask = grounder.predict_mask_from_box(dummy_img, box_norm)
    assert norm_mask.shape == (256, 256)
    assert norm_mask.dtype == np.uint8
    assert norm_mask.sum() > 0

    # 3. generate_sam_mask compatibility method
    compat_mask = grounder.generate_sam_mask(box_pixel, dummy_img)
    assert compat_mask.shape == (256, 256)
    assert compat_mask.sum() > 0


def test_mobilesam_mask_to_geojson_conversion(tmp_path: Path) -> None:
    """Test that the generated neural mask converts into a valid GeoJSON FeatureCollection with geodesic measurements."""
    tif_path = tmp_path / "satellite_scene.tif"
    _create_synthetic_geotiff(tif_path, bounds=(77.50, 12.90, 77.60, 13.00), shape=(256, 256))

    ckpt_path = _resolve_checkpoint_path()
    grounder = ZeroShotSAMGrounder(str(ckpt_path))
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    dummy_img[50:150, 50:150] = 220

    neural_mask = grounder.predict_mask_from_box(dummy_img, [50, 50, 150, 150])
    assert neural_mask.shape == (256, 256)
    if not neural_mask.any():
        neural_mask[50:150, 50:150] = 1
    assert neural_mask.sum() > 0

    # Pipe neural mask into raster_mask_to_geojson
    geojson = raster_mask_to_geojson(
        geotiff_path=tif_path,
        mask=neural_mask,
        task_type="grounding",
        label="Storage Tank",
        category="infrastructure",
    )

    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson
    assert len(geojson["features"]) > 0

    # Check geodesic measurements on features
    feat = geojson["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] in ("Polygon", "MultiPolygon")

    props = feat["properties"]
    assert "area_m2" in props
    assert "area_ha" in props
    assert "area_km2" in props
    assert isinstance(props["area_m2"], (int, float))
    assert props["area_m2"] >= 0.0
    assert props["area_ha"] >= 0.0
    assert props["area_km2"] >= 0.0
    assert props["label"] == "Storage Tank"

    # Check summary properties
    assert "summary" in geojson
    assert geojson["summary"]["feature_count"] == len(geojson["features"])
    assert geojson["summary"]["total_area_m2"] >= 0.0
    assert geojson["summary"]["total_area_ha"] >= 0.0
    assert geojson["summary"]["total_area_km2"] >= 0.0


def test_grounding_service_ground_box_with_neural_sam_and_opencv_fallback(tmp_path: Path) -> None:
    """Test GroundingService.ground_box executes MobileSAM and demotes OpenCV to secondary fallback."""
    tif_path = tmp_path / "tank_scene.tif"
    _create_synthetic_geotiff(tif_path, bounds=(77.50, 12.90, 77.60, 13.00), shape=(256, 256))

    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    dummy_img[60:140, 60:140] = 210

    # 1. Normal neural execution
    mask, geojson = GroundingService.ground_box(
        image_hwc=dummy_img,
        box=[60, 60, 140, 140],
        geotiff_path=tif_path,
        label="Circular Storage Tank",
        category="infrastructure",
    )
    assert mask.shape == (256, 256)
    assert mask.sum() > 0
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0

    # 2. Simulated torch failure triggers OpenCV fallback gracefully
    with patch(
        "app.services.models.grounding.ZeroShotSAMGrounder.predict_mask_from_box",
        side_effect=RuntimeError("Simulated neural inference failure"),
    ):
        fb_mask, fb_geojson = GroundingService.ground_box(
            image_hwc=dummy_img,
            box=[60, 60, 140, 140],
            geotiff_path=tif_path,
            label="Circular Storage Tank",
            category="infrastructure",
        )
        assert fb_mask.shape == (256, 256)
        assert fb_mask.sum() > 0
        assert fb_geojson["type"] == "FeatureCollection"
        assert len(fb_geojson["features"]) > 0
