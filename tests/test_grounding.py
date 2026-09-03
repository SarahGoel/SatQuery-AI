"""Phase 4 Step 8 — zero-shot grounding and raster-to-GeoJSON."""

from __future__ import annotations

import numpy as np
import torch

from app.services.geospatial.vector import convert_raster_mask_to_geojson
from app.services.models.grounding import ZeroShotSAMGrounder


def test_convert_raster_mask_to_geojson_multipolygon() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    transform = [10.0, 0.0, 450000.0, 0.0, -10.0, 5300000.0]
    geojson = convert_raster_mask_to_geojson(mask, transform, "EPSG:32643")
    assert geojson["geometry"]["type"] == "MultiPolygon"
    assert geojson["properties"]["crs"] == "EPSG:32643"
    coords = geojson["geometry"]["coordinates"]
    assert isinstance(coords, list) and coords
    xs = [pt[0] for ring in coords[0] for pt in ring]
    ys = [pt[1] for ring in coords[0] for pt in ring]
    assert min(xs) >= 450000.0
    assert max(ys) <= 5300000.0


def test_convert_empty_mask_is_feature_collection() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    transform = [1.0, 0.0, 0.0, 0.0, -1.0, 16.0]
    geojson = convert_raster_mask_to_geojson(mask, transform, "EPSG:4326")
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"] == []


def test_zero_shot_sam_grounder_box_and_mask() -> None:
    grounder = ZeroShotSAMGrounder(checkpoint_path="local_models/sam/mobile_sam.pt")
    image = torch.rand(3, 64, 80)
    box = grounder.predict_bounding_box("delineate the reservoir", image)
    assert len(box) == 4
    xmin, ymin, xmax, ymax = box
    assert 0 <= xmin < xmax <= 80
    assert 0 <= ymin < ymax <= 64
    rgb = np.random.rand(64, 80, 3).astype(np.float32)
    mask = grounder.generate_sam_mask(box, rgb)
    assert mask.shape == (64, 80)
    assert mask.dtype == np.uint8
    assert mask.max() == 1
