"""Phase 7 - Siamese Temporal & Multi-Spectral Tensor Integration Tests.

Validates:
1. PyTorch Siamese bi-temporal feature differencing with SiameseChangeNet.
2. Cross-modal Optical + SAR 4-band tensor stacking and BigEarthNetClassifierNet multi-label forward pass.
3. GeoJSON polygonization with geodesic areas and end-to-end analytical workflow execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
import torch

from app.services.analytical.cross_modal import (
    CrossModalAnalysisTool,
    CrossModalResult,
)
from app.services.analytical.temporal_change import (
    SiameseChangeNet,
    SiameseChangeResult,
    TemporalChangeTool,
)
from app.services.models.bigearthnet import (
    CORINE_19_CLASSES,
    BigEarthNetClassifierNet,
    BigEarthNetLandCoverClassifier,
)


def _create_synthetic_geotiff(
    path: Path,
    width: int = 128,
    height: int = 128,
    count: int = 3,
    dtype: str = "float32",
    fill_value: float = 0.5,
) -> Path:
    """Helper creating small georeferenced GeoTIFF for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(77.0, 12.0, 77.1, 12.1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        data = np.full((count, height, width), fill_value, dtype=dtype)
        dst.write(data)
    return path


# ============================================================================
# 1. Siamese Temporal Differencing Tests
# ============================================================================

def test_siamese_temporal_forward_pass() -> None:
    """Assert passing two different (256, 256, 3) arrays through SiameseChangeNet produces a valid (256, 256) change mask and valid GeoJSON."""
    net = SiameseChangeNet(in_channels=3, feature_dim=32)
    net.eval()

    # Create two different (256, 256, 3) image arrays
    arr1 = np.zeros((256, 256, 3), dtype=np.float32)
    arr2 = np.zeros((256, 256, 3), dtype=np.float32)
    # Introduce surface change in central region
    arr2[64:192, 64:192, :] = 1.0

    # Execute forward pass through SiameseChangeNet
    res = net(arr1, arr2)

    # Validate unpacking contract: mask, geojson = net(arr1, arr2)
    assert isinstance(res, (tuple, SiameseChangeResult))
    mask, geojson = res

    # 1. Assert valid (256, 256) change mask
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    # Assert change was detected in the altered quadrant
    assert int(mask.sum()) > 0

    # 2. Assert valid GeoJSON FeatureCollection
    assert isinstance(geojson, dict)
    assert geojson.get("type") == "FeatureCollection"
    assert "features" in geojson
    features = geojson["features"]
    assert len(features) >= 1

    # 3. Assert polygon geometry and geodesic area properties
    feat0 = features[0]
    assert feat0["type"] == "Feature"
    assert "geometry" in feat0
    assert feat0["geometry"]["type"] in ("Polygon", "MultiPolygon")

    props = feat0["properties"]
    assert "area_m2" in props
    assert "area_ha" in props
    assert "area_km2" in props
    assert isinstance(props["area_m2"], (int, float))
    assert props["area_m2"] >= 0.0
    assert props["area_ha"] >= 0.0
    assert props["area_km2"] >= 0.0


def test_siamese_temporal_tensor_forward_pass() -> None:
    """Assert passing PyTorch tensors directly returns [B, 1, H, W] change probability map."""
    net = SiameseChangeNet(in_channels=3, feature_dim=32)
    net.eval()

    t1 = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    t2 = torch.ones((1, 3, 128, 128), dtype=torch.float32)

    with torch.no_grad():
        prob = net(t1, t2)

    assert isinstance(prob, torch.Tensor)
    assert prob.shape == (1, 1, 128, 128)
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
    # Dissimilar inputs should yield high change probability
    assert prob.mean().item() > 0.5


def test_siamese_temporal_identical_arrays_no_change() -> None:
    """Assert passing two identical arrays yields zero detected change."""
    net = SiameseChangeNet(in_channels=3, feature_dim=32)
    net.eval()

    arr = np.full((256, 256, 3), 0.5, dtype=np.float32)
    mask, geojson = net(arr, arr)

    assert mask.shape == (256, 256)
    assert int(mask.sum()) == 0
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 0


# ============================================================================
# 2. Cross-Modal SAR + Optical Tensor Stacking Tests
# ============================================================================

def test_cross_modal_tensor_stacking() -> None:
    """Assert BigEarthNetClassifierNet accepts a [1, 4, 256, 256] tensor and outputs a [1, 19] logit array."""
    net = BigEarthNetClassifierNet(in_channels=4, num_classes=19)
    net.eval()

    # Pass 4-band tensor: [B, C, H, W] = [1, 4, 256, 256]
    tensor = torch.randn(1, 4, 256, 256, dtype=torch.float32)

    with torch.no_grad():
        logits = net(tensor)

    # 1. Assert logit output shape
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 19)

    # 2. Assert multi-label sigmoid probabilities
    probs = torch.sigmoid(logits)
    assert probs.shape == (1, 19)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Convert to numpy logit array
    logits_np = logits.squeeze(0).cpu().numpy()
    assert logits_np.shape == (19,)


def test_cross_modal_analysis_tool_tensor_stacking(tmp_path: Path) -> None:
    """Verify CrossModalAnalysisTool dynamically stacks 3-band Optical + 1-band SAR and predicts CORINE-19."""
    opt_file = _create_synthetic_geotiff(tmp_path / "cartosat_opt.tif", width=128, height=128, count=3, fill_value=0.4)
    sar_file = _create_synthetic_geotiff(tmp_path / "risat_sar.tif", width=128, height=128, count=1, fill_value=-22.0)

    tool = CrossModalAnalysisTool()

    # 1. Test dynamic tensor stacking method
    stacked = tool.stack_tensors(opt_file, sar_file)
    assert isinstance(stacked, torch.Tensor)
    assert stacked.shape == (1, 4, 128, 128)

    # 2. Test full end-to-end analysis
    result = tool.analyze(
        optical_path=opt_file,
        sar_path=sar_file,
        query="Map flooded regions and built-up infrastructure",
    )

    assert isinstance(result, CrossModalResult)
    assert "Multi-spectral tensor fusion" in result.answer
    assert "corine_classes" in result.params
    assert len(result.params["corine_classes"]) >= 1
    assert result.params["stacked_tensor_shape"] == [1, 4, 128, 128]
    assert result.geojson["type"] == "FeatureCollection"
    assert len(result.geojson["features"]) >= 1


def test_temporal_change_tool_siamese_execution(tmp_path: Path) -> None:
    """Verify TemporalChangeTool runs PyTorch Siamese differencing and updates scratchpad."""
    t1_file = _create_synthetic_geotiff(tmp_path / "pre_event.tif", width=128, height=128, count=3, fill_value=0.1)
    t2_file = _create_synthetic_geotiff(tmp_path / "post_event.tif", width=128, height=128, count=3, fill_value=0.9)

    tool = TemporalChangeTool()
    scratchpad: Dict[str, Any] = {"filepaths": [str(t1_file), str(t2_file)]}

    res = tool.run(scratchpad, t1_path=str(t1_file), t2_path=str(t2_file), threshold=0.5)

    assert res["status"] == "success"
    assert res["tool"] == "TemporalChangeTool"
    assert res["mode"] == "siamese_pytorch_tensor"
    assert isinstance(res["change_mask"], np.ndarray)
    assert res["change_mask"].shape == (128, 128)
    assert res["change_pixel_count"] > 0
    assert "[INCREASED]" in res["directional_verdict"]
    assert res["geojson"]["type"] == "FeatureCollection"
    assert len(res["geojson"]["features"]) >= 1

    # Assert scratchpad was properly populated
    assert scratchpad["change_pixel_count"] == res["change_pixel_count"]
    assert scratchpad["temporal_mode"] == "siamese_pytorch_tensor"
