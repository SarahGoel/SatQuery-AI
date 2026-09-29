"""Tests for Geospatial API Endpoints (/composite, /inspect-point, /export) and Spatial AOI Clipping.

Validates:
1. POST /api/v1/geospatial/composite (RGB, CIR, SAR dB, NDVI, NDWI, data_uri).
2. POST /api/v1/geospatial/inspect-point (coordinate to pixel sampling, indices, terrain classification).
3. GET /api/v1/reports/{trace_id}/export (GeoJSON and KML downloads).
4. Spatial AOI clipping and context injection in POST /api/v1/query.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

import backend.api.routes as routes_mod
from app.utils.trace_store import remember_trace
from backend.main import app


def _create_synthetic_geotiff(
    path: Path,
    width: int = 100,
    height: int = 100,
    bands: int = 4,
    bounds: tuple[float, float, float, float] = (78.0, 20.0, 79.0, 21.0),
) -> Path:
    """Creates a synthetic multi-band GeoTIFF with valid georeferencing."""
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    # Band 1: Red, Band 2: Green, Band 3: Blue, Band 4: NIR
    data = np.zeros((bands, height, width), dtype=np.float32)
    # Center water pond (high green, low NIR)
    data[0, :, :] = 100.0   # Red
    data[1, :, :] = 150.0   # Green
    data[2, :, :] = 80.0    # Blue
    data[3, :, :] = 500.0   # NIR (vegetation dominant)
    data[3, 40:60, 40:60] = 20.0  # Center water pond

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ============================================================================
# 1. Composite Generation Tests
# ============================================================================

def test_geospatial_composite_multipart_upload(client: TestClient, tmp_path: Path):
    """Verify POST /api/v1/geospatial/composite accepts multipart GeoTIFF and renders composites."""
    tif_path = _create_synthetic_geotiff(tmp_path / "test_comp.tif")
    tif_bytes = tif_path.read_bytes()

    # Test RGB composite
    res_rgb = client.post(
        "/api/v1/geospatial/composite",
        files={"file": ("scene.tif", tif_bytes, "image/tiff")},
        data={"composite_type": "rgb"},
    )
    assert res_rgb.status_code == 200
    assert res_rgb.headers["content-type"] == "image/png"
    assert res_rgb.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Test CIR composite
    res_cir = client.post(
        "/api/v1/geospatial/composite",
        files={"file": ("scene.tif", tif_bytes, "image/tiff")},
        data={"composite_type": "cir"},
    )
    assert res_cir.status_code == 200
    assert res_cir.headers["content-type"] == "image/png"
    assert res_cir.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Test SAR dB composite
    res_sar = client.post(
        "/api/v1/geospatial/composite",
        files={"file": ("scene.tif", tif_bytes, "image/tiff")},
        data={"composite_type": "sar_db"},
    )
    assert res_sar.status_code == 200
    assert res_sar.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Test NDVI colormap composite
    res_ndvi = client.post(
        "/api/v1/geospatial/composite",
        files={"file": ("scene.tif", tif_bytes, "image/tiff")},
        data={"composite_type": "ndvi"},
    )
    assert res_ndvi.status_code == 200
    assert res_ndvi.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Test NDWI colormap composite
    res_ndwi = client.post(
        "/api/v1/geospatial/composite",
        files={"file": ("scene.tif", tif_bytes, "image/tiff")},
        data={"composite_type": "ndwi"},
    )
    assert res_ndwi.status_code == 200
    assert res_ndwi.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Test as_data_uri=True
    res_b64 = client.post(
        "/api/v1/geospatial/composite",
        files={"file": ("scene.tif", tif_bytes, "image/tiff")},
        data={"composite_type": "rgb", "as_data_uri": "true"},
    )
    assert res_b64.status_code == 200
    json_b64 = res_b64.json()
    assert json_b64["status"] == "ok"
    assert json_b64["composite_type"] == "rgb"
    assert json_b64["data_uri"].startswith("data:image/png;base64,")


def test_geospatial_composite_by_trace_id(client: TestClient, tmp_path: Path):
    """Verify POST /api/v1/geospatial/composite resolves raster via stored trace_id."""
    tif_path = _create_synthetic_geotiff(tmp_path / "trace_scene.tif")
    trace_id = "ISRO-SQ-TEST-COMPOSITE"
    remember_trace(trace_id, {"trace_id": trace_id, "filepaths": [str(tif_path)]})

    response = client.post(
        "/api/v1/geospatial/composite",
        data={"trace_id": trace_id, "composite_type": "cir"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


# ============================================================================
# 2. Inspect Point Tests
# ============================================================================

def test_geospatial_inspect_point_sampling(client: TestClient, tmp_path: Path):
    """Verify POST /api/v1/geospatial/inspect-point samples exact pixel coordinates & spectral indices."""
    tif_path = _create_synthetic_geotiff(
        tmp_path / "inspect_scene.tif",
        width=100,
        height=100,
        bounds=(78.0, 20.0, 79.0, 21.0),
    )
    trace_id = "ISRO-SQ-TEST-INSPECT"
    remember_trace(trace_id, {"trace_id": trace_id, "filepaths": [str(tif_path)]})

    # Sample Center Point (lon=78.5, lat=20.5 -> row=50, col=50, which has low NIR / center pond)
    payload = {
        "trace_id": trace_id,
        "latitude": 20.5,
        "longitude": 78.5,
    }
    response = client.post("/api/v1/geospatial/inspect-point", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["coordinate"]["lat"] == 20.5
    assert data["coordinate"]["lon"] == 78.5
    assert data["pixel"]["row"] == 50
    assert data["pixel"]["col"] == 50

    # Bands inspection
    assert "red" in data["bands"]
    assert "green" in data["bands"]
    assert "nir" in data["bands"]

    # Indices inspection
    assert "ndvi" in data["indices"]
    assert "ndwi" in data["indices"]
    assert "terrain_classification" in data
    assert isinstance(data["terrain_classification"], str)

    # Available composites inspection
    assert "available_composites" in data
    assert "cir" in data["available_composites"]
    assert "ndvi" in data["available_composites"]
    assert "ndwi" in data["available_composites"]


def test_geospatial_inspect_point_out_of_bounds_and_missing(client: TestClient, tmp_path: Path):
    """Verify inspect-point returns 400 for out-of-bounds coordinates and 404 for unknown trace_id."""
    tif_path = _create_synthetic_geotiff(tmp_path / "bounds_scene.tif", bounds=(78.0, 20.0, 79.0, 21.0))
    trace_id = "ISRO-SQ-BOUNDS-TEST"
    remember_trace(trace_id, {"trace_id": trace_id, "filepaths": [str(tif_path)]})

    # Out-of-bounds request
    oob_res = client.post(
        "/api/v1/geospatial/inspect-point",
        json={"trace_id": trace_id, "latitude": 5.0, "longitude": 10.0},
    )
    assert oob_res.status_code == 400
    assert "outside raster extent" in oob_res.json()["detail"]

    # Non-existent trace_id
    notfound_res = client.post(
        "/api/v1/geospatial/inspect-point",
        json={"trace_id": "NON-EXISTENT-TRACE", "latitude": 20.5, "longitude": 78.5},
    )
    assert notfound_res.status_code == 404


# ============================================================================
# 3. Vector Export Tests (GeoJSON and KML)
# ============================================================================

def test_export_vector_report_geojson_and_kml(client: TestClient):
    """Verify GET /api/v1/reports/{trace_id}/export streams valid GeoJSON and KML files."""
    trace_id = "ISRO-SQ-EXPORT-TEST-99"
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Water Reservoir Alpha",
                    "category": "water",
                    "confidence": 0.95,
                    "area_ha": 42.5,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [78.2, 20.2],
                            [78.6, 20.2],
                            [78.6, 20.6],
                            [78.2, 20.6],
                            [78.2, 20.2],
                        ]
                    ],
                },
            }
        ],
    }
    remember_trace(trace_id, {"trace_id": trace_id, "geojson": feature_collection})

    # Test GeoJSON Export
    res_geojson = client.get(f"/api/v1/reports/{trace_id}/export?format=geojson")
    assert res_geojson.status_code == 200
    assert res_geojson.headers["content-type"] == "application/geo+json"
    assert f'filename="{trace_id}.geojson"' in res_geojson.headers["content-disposition"]
    parsed_json = res_geojson.json()
    assert parsed_json["type"] == "FeatureCollection"
    assert len(parsed_json["features"]) == 1
    assert parsed_json["features"][0]["properties"]["name"] == "Water Reservoir Alpha"

    # Test KML Export
    res_kml = client.get(f"/api/v1/reports/{trace_id}/export?format=kml")
    assert res_kml.status_code == 200
    assert res_kml.headers["content-type"] == "application/vnd.google-earth.kml+xml"
    assert f'filename="{trace_id}.kml"' in res_kml.headers["content-disposition"]
    kml_text = res_kml.text
    assert "<kml" in kml_text
    assert "Water Reservoir Alpha" in kml_text
    assert "<Polygon" in kml_text
    assert "<coordinates>" in kml_text

    # Test Unsupported Format
    res_bad = client.get(f"/api/v1/reports/{trace_id}/export?format=shapefile")
    assert res_bad.status_code == 400


# ============================================================================
# 4. Spatial AOI Clipping in Query Pipeline Tests
# ============================================================================

def test_parse_spatial_aoi_helpers():
    """Verify parse_spatial_aoi handles bbox lists, dicts, and GeoJSON geometries."""
    # Bbox list
    geom1, bounds1 = routes_mod.parse_spatial_aoi("[78.2, 20.2, 78.6, 20.6]")
    assert bounds1 == (78.2, 20.2, 78.6, 20.6)
    assert geom1["type"] == "Polygon"

    # Comma-separated string
    geom2, bounds2 = routes_mod.parse_spatial_aoi("78.1, 20.1, 78.9, 20.9")
    assert bounds2 == (78.1, 20.1, 78.9, 20.9)

    # GeoJSON Polygon dict
    poly_dict = {
        "type": "Polygon",
        "coordinates": [[[78.3, 20.3], [78.7, 20.3], [78.7, 20.7], [78.3, 20.7], [78.3, 20.3]]],
    }
    geom3, bounds3 = routes_mod.parse_spatial_aoi(json.dumps(poly_dict))
    assert geom3["type"] == "Polygon"
    assert bounds3 == (78.3, 20.3, 78.7, 20.7)


def test_clip_raster_to_aoi(tmp_path: Path):
    """Verify clip_raster_to_aoi crops the GeoTIFF to the exact AOI bounds."""
    src_tif = _create_synthetic_geotiff(
        tmp_path / "full_scene.tif",
        width=100,
        height=100,
        bounds=(78.0, 20.0, 79.0, 21.0),
    )
    aoi_geom = {
        "type": "Polygon",
        "coordinates": [[[78.2, 20.2], [78.6, 20.2], [78.6, 20.6], [78.2, 20.6], [78.2, 20.2]]],
    }
    cropped_out = tmp_path / "cropped_scene.tif"

    result_path = routes_mod.clip_raster_to_aoi(str(src_tif), aoi_geom, str(cropped_out))
    assert Path(result_path).exists()

    with rasterio.open(result_path) as dst:
        # Cropped raster should be smaller than full 100x100 raster
        assert dst.width < 100
        assert dst.height < 100
        assert dst.count == 4
        # Bounds should approximate the AOI [78.2, 20.2, 78.6, 20.6]
        b = dst.bounds
        assert 78.19 <= b.left <= 78.21
        assert 20.19 <= b.bottom <= 20.21
        assert 78.59 <= b.right <= 78.61
        assert 20.59 <= b.top <= 20.61


def test_query_pipeline_with_spatial_aoi_and_export_links(client: TestClient, tmp_path: Path):
    """Verify POST /api/v1/query accepts spatial_aoi, crops raster, and includes vector export links."""
    src_tif = _create_synthetic_geotiff(
        tmp_path / "aoi_query_scene.tif",
        width=100,
        height=100,
        bounds=(78.0, 20.0, 79.0, 21.0),
    )
    tif_bytes = src_tif.read_bytes()

    res = client.post(
        "/api/v1/query",
        files={"file": ("aoi_query_scene.tif", tif_bytes, "image/tiff")},
        data={
            "query": "Highlight and delineate water bodies in this region",
            "spatial_aoi": "[78.2, 20.2, 78.6, 20.6]",
            "use_mobilesam": "false",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "report" in data
    # Check that geojson and kml download links are present in report envelope
    assert "geojson" in data["report"]
    assert "kml" in data["report"]
    assert "export?format=geojson" in data["report"]["geojson"]
    assert "export?format=kml" in data["report"]["kml"]
    # Check that available_composites is present in response and input_metadata
    assert "available_composites" in data
    assert "cir" in data["available_composites"]
    assert "ndvi" in data["available_composites"]
    assert "available_composites" in data["input_metadata"]
    assert "cir" in data["input_metadata"]["available_composites"]

