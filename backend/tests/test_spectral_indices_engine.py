"""Comprehensive unit tests for SpectralIndicesEngine & Dynamic Sensor Band Profiles."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.geospatial.spectral import (
    BandMapping,
    SpectralIndicesEngine,
    SpectralIndicesResult,
    compute_ndvi,
    compute_ndwi,
    compute_mndwi,
    compute_ndbi,
    generate_cir_composite,
    generate_sar_db,
    render_colormap,
    detect_sensor_profile,
    calculate_spectral_indices,
    generate_n_channel_tensor,
    SpectralPack,
    PROFILE_SENTINEL2_12B,
    PROFILE_LANDSAT89_7B,
    PROFILE_CARTOSAT_4B,
    PROFILE_GENERIC_RGB,
)


# ---------------------------------------------------------------------------
# 1. Core Index Calculation Tests (NDVI, NDWI, MNDWI, NDBI)
# ---------------------------------------------------------------------------

def test_compute_ndvi_formula_and_zero_division() -> None:
    red = np.array([[0.1, 0.2], [0.0, 0.5]], dtype=np.float32)
    nir = np.array([[0.6, 0.4], [0.0, 0.5]], dtype=np.float32)

    ndvi = compute_ndvi(red, nir)
    assert ndvi.shape == (2, 2)
    assert ndvi.dtype == np.float32

    # Expected: (nir - red) / (nir + red + 1e-7)
    expected_00 = (0.6 - 0.1) / (0.6 + 0.1 + 1e-7)
    expected_01 = (0.4 - 0.2) / (0.4 + 0.2 + 1e-7)
    expected_10 = 0.0  # Zero-division protected
    expected_11 = (0.5 - 0.5) / (0.5 + 0.5 + 1e-7)

    assert pytest.approx(ndvi[0, 0], rel=1e-4) == expected_00
    assert pytest.approx(ndvi[0, 1], rel=1e-4) == expected_01
    assert pytest.approx(ndvi[1, 0], abs=1e-4) == expected_10
    assert pytest.approx(ndvi[1, 1], abs=1e-4) == expected_11
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)


def test_compute_ndwi_formula_and_zero_division() -> None:
    green = np.array([[0.4, 0.1], [0.0, 0.0]], dtype=np.float32)
    nir = np.array([[0.1, 0.5], [0.0, 0.0]], dtype=np.float32)

    ndwi = compute_ndwi(green, nir)
    assert ndwi.shape == (2, 2)
    assert ndwi.dtype == np.float32

    # Water has positive NDWI (Green > NIR)
    assert ndwi[0, 0] > 0.0
    # Vegetation has negative NDWI (NIR > Green)
    assert ndwi[0, 1] < 0.0
    # Zero division returns 0.0
    assert pytest.approx(ndwi[1, 0], abs=1e-4) == 0.0


def test_compute_mndwi_with_swir_and_fallback() -> None:
    green = np.array([[0.5, 0.3], [0.1, 0.0]], dtype=np.float32)
    swir = np.array([[0.1, 0.6], [0.1, 0.0]], dtype=np.float32)
    nir = np.array([[0.2, 0.5], [0.1, 0.0]], dtype=np.float32)

    # Standard MNDWI using SWIR
    mndwi = compute_mndwi(green, swir=swir)
    assert mndwi.shape == (2, 2)
    expected_00 = (0.5 - 0.1) / (0.5 + 0.1 + 1e-7)
    assert pytest.approx(mndwi[0, 0], rel=1e-4) == expected_00

    # Fallback to NIR when SWIR is None
    mndwi_fallback = compute_mndwi(green, swir=None, nir=nir)
    expected_fallback = compute_ndwi(green, nir)
    np.testing.assert_allclose(mndwi_fallback, expected_fallback, rtol=1e-5)

    # Fallback to zeros when neither is available
    mndwi_zeros = compute_mndwi(green, swir=None, nir=None)
    np.testing.assert_array_equal(mndwi_zeros, np.zeros_like(green))


def test_compute_ndbi_with_swir_and_fallback() -> None:
    swir = np.array([[0.6, 0.2], [0.0, 0.0]], dtype=np.float32)
    nir = np.array([[0.3, 0.5], [0.0, 0.0]], dtype=np.float32)

    ndbi = compute_ndbi(swir, nir)
    assert ndbi.shape == (2, 2)
    assert ndbi[0, 0] > 0.0  # Urban/built-up (SWIR > NIR)
    assert ndbi[0, 1] < 0.0  # Non-builtup (NIR > SWIR)
    assert pytest.approx(ndbi[1, 0], abs=1e-4) == 0.0

    # Fallback when SWIR is None
    ndbi_none = compute_ndbi(None, nir)
    np.testing.assert_array_equal(ndbi_none, np.zeros_like(nir))


# ---------------------------------------------------------------------------
# 2. Dynamic Sensor Band Profile Detection Tests
# ---------------------------------------------------------------------------

def test_detect_sentinel2_profile_from_tags_and_count() -> None:
    # 12-band Sentinel-2 L2A
    bmap_12 = detect_sensor_profile(12, tags={"SENSOR": "Sentinel-2", "PLATFORM": "S2A"})
    assert bmap_12.sensor_name == "Sentinel-2"
    assert bmap_12.blue == 1   # B2
    assert bmap_12.green == 2  # B3
    assert bmap_12.red == 3    # B4
    assert bmap_12.nir == 7    # B8
    assert bmap_12.swir == 10  # B11

    # 6-band Sentinel-2 subset
    bmap_6 = detect_sensor_profile(6, tags={"SATELLITE": "SENTINEL-2"})
    assert bmap_6.sensor_name == "Sentinel-2"
    assert bmap_6.blue == 0
    assert bmap_6.nir == 3
    assert bmap_6.swir == 4

    # 4-band Sentinel-2 VNIR
    bmap_4 = detect_sensor_profile(4, tags={"SENSOR": "Sentinel-2A MSI"})
    assert bmap_4.sensor_name == "Sentinel-2"
    assert bmap_4.nir == 3


def test_detect_landsat_profile_from_tags_and_count() -> None:
    # 7-band Landsat-8/9 OLI
    bmap_7 = detect_sensor_profile(7, tags={"SPACECRAFT_NAME": "Landsat-8"})
    assert bmap_7.sensor_name == "Landsat-8/9"
    assert bmap_7.blue == 1
    assert bmap_7.green == 2
    assert bmap_7.red == 3
    assert bmap_7.nir == 4
    assert bmap_7.swir == 5


def test_detect_cartosat_profile() -> None:
    bmap = detect_sensor_profile(4, tags={"SENSOR": "Cartosat-2S", "MODALITY": "Optical/Multispectral"})
    assert bmap.sensor_name == "Cartosat-2S"
    assert bmap.blue == 0
    assert bmap.green == 1
    assert bmap.red == 2
    assert bmap.nir == 3


def test_detect_from_band_descriptions() -> None:
    descs = ["Blue channel", "Green channel", "Red channel", "Near-Infrared B8", "Short-Wave IR B11"]
    bmap = detect_sensor_profile(5, descriptions=descs)
    assert bmap.blue == 0
    assert bmap.green == 1
    assert bmap.red == 2
    assert bmap.nir == 3
    assert bmap.swir == 4


def test_detect_generic_fallbacks_from_counts() -> None:
    # 3-band true-color RGB
    bmap_3 = detect_sensor_profile(3)
    assert bmap_3.red == 0
    assert bmap_3.green == 1
    assert bmap_3.blue == 2
    assert bmap_3.nir is None

    # 4-band generic
    bmap_4 = detect_sensor_profile(4)
    assert bmap_4.nir == 3

    # 1-band grayscale/SAR
    bmap_1 = detect_sensor_profile(1)
    assert bmap_1.sensor_name == "Grayscale-SAR"


# ---------------------------------------------------------------------------
# 3. Composite Generation Tests (CIR, SAR dB, Colormap Rendering)
# ---------------------------------------------------------------------------

def test_generate_cir_composite_shapes_and_values() -> None:
    # 4-band raster: (4, 32, 32) -> B, G, R, NIR
    rng = np.random.default_rng(42)
    b = rng.uniform(0.1, 0.4, (32, 32)).astype(np.float32)
    g = rng.uniform(0.1, 0.5, (32, 32)).astype(np.float32)
    r = rng.uniform(0.1, 0.6, (32, 32)).astype(np.float32)
    nir = rng.uniform(0.4, 0.9, (32, 32)).astype(np.float32)

    raster_4ch = np.stack([b, g, r, nir], axis=0)

    cir = generate_cir_composite(raster_4ch)
    assert cir.shape == (32, 32, 3)
    assert cir.dtype == np.uint8
    assert cir.min() >= 0
    assert cir.max() <= 255


def test_generate_sar_db_conversion() -> None:
    # Linear amplitude array
    sar_linear = np.array([[0.0, 0.1], [1.0, 10.0]], dtype=np.float32)

    sar_8bit = generate_sar_db(sar_linear)
    assert sar_8bit.shape == (2, 2)
    assert sar_8bit.dtype == np.uint8
    # High amplitude has higher brightness in calibrated dB display
    assert sar_8bit[1, 1] >= sar_8bit[0, 0]


def test_render_colormap_png_bytes() -> None:
    index_map = np.linspace(-1.0, 1.0, 100, dtype=np.float32).reshape(10, 10)

    png_buf = render_colormap(index_map, colormap="RdYlGn")
    assert isinstance(png_buf, (bytes, io.BytesIO))
    png_bytes = png_buf.getvalue() if isinstance(png_buf, io.BytesIO) else png_buf
    assert len(png_bytes) > 50

    # Verify valid PNG header
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # Verify PIL can open and verify dimensions
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (10, 10)

    # Verify SpectralIndicesEngine staticmethod returns io.BytesIO buffer directly
    engine_buf = SpectralIndicesEngine.render_colormap(index_map, colormap="RdYlGn")
    assert isinstance(engine_buf, io.BytesIO)
    assert engine_buf.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert img.size == (10, 10)


# ---------------------------------------------------------------------------
# 4. Unified SpectralIndicesEngine Class Tests
# ---------------------------------------------------------------------------

def test_spectral_indices_engine_process_raster() -> None:
    engine = SpectralIndicesEngine()

    # Synthetic 5-band raster: (5, 24, 24) -> Blue, Green, Red, NIR, SWIR
    rng = np.random.default_rng(101)
    raster_5ch = rng.uniform(0.05, 0.85, (5, 24, 24)).astype(np.float32)

    res = engine.process_raster(raster_5ch, generate_cir=True)

    assert isinstance(res, SpectralIndicesResult)
    assert res.ndvi.shape == (24, 24)
    assert res.ndwi.shape == (24, 24)
    assert res.mndwi.shape == (24, 24)
    assert res.ndbi.shape == (24, 24)
    assert res.cir_composite is not None
    assert res.cir_composite.shape == (24, 24, 3)

    # Check metrics
    metrics = res.metrics
    assert "ndvi_mean" in metrics
    assert "ndwi_mean" in metrics
    assert "mndwi_mean" in metrics
    assert "ndbi_mean" in metrics


def test_legacy_calculate_spectral_indices_backward_compat() -> None:
    r = np.array([[0.2, 0.4]], dtype=np.float32)
    g = np.array([[0.3, 0.5]], dtype=np.float32)
    nir = np.array([[0.7, 0.8]], dtype=np.float32)

    ndvi, ndwi = calculate_spectral_indices(r, g, nir)
    assert ndvi.shape == (1, 2)
    assert ndwi.shape == (1, 2)
    np.testing.assert_allclose(ndvi, (nir - r) / (nir + r), rtol=1e-5)
    np.testing.assert_allclose(ndwi, (g - nir) / (g + nir), rtol=1e-5)


def test_dynamic_available_composites_detection() -> None:
    """Verify available_composites flags correctly for 4-band, 5-band, SAR, and RGB sensors."""
    engine = SpectralIndicesEngine()

    # 1. 4-Band Optical (RGB + NIR) -> RGB, CIR, NDVI, NDWI
    profile_4b = detect_sensor_profile(4)
    assert profile_4b.nir is not None and profile_4b.red is not None and profile_4b.green is not None
    assert profile_4b.available_composites == ["rgb", "cir", "ndvi", "ndwi"]
    assert profile_4b.to_dict()["available_composites"] == ["rgb", "cir", "ndvi", "ndwi"]

    # 2. 5-Band / SWIR Optical -> RGB, CIR, NDVI, NDWI, MNDWI, NDBI
    profile_5b = detect_sensor_profile(5)
    assert profile_5b.swir is not None
    assert profile_5b.available_composites == ["rgb", "cir", "ndvi", "ndwi", "mndwi", "ndbi"]

    # 3. Sentinel-2 12-Band -> All optical indices
    profile_s2 = PROFILE_SENTINEL2_12B
    assert profile_s2.available_composites == ["rgb", "cir", "ndvi", "ndwi", "mndwi", "ndbi"]

    # 4. Single-Band / 2-Band SAR -> SAR dB backscatter
    profile_sar = detect_sensor_profile(1, tags={"SENSOR": "Sentinel-1 C-SAR"})
    assert profile_sar.available_composites == ["sar_db"]

    # 5. Standard 3-Band RGB -> RGB only
    profile_rgb = PROFILE_GENERIC_RGB
    assert profile_rgb.available_composites == ["rgb"]

    # 6. Static helper method on SpectralIndicesEngine
    comps_4b = engine.get_available_composites(band_map=profile_4b, band_count=4)
    assert comps_4b == ["rgb", "cir", "ndvi", "ndwi"]

    comps_sar = engine.get_available_composites(band_map=profile_sar, band_count=1, is_sar=True)
    assert comps_sar == ["sar_db"]


def test_3band_rgb_visible_spectral_fallbacks() -> None:
    """Verify 3-band RGB imagery generates valid visible-spectrum approximations for NDVI, NDWI, and CIR."""
    engine = SpectralIndicesEngine()
    red = np.array([[100, 50], [20, 200]], dtype=np.float32)
    green = np.array([[180, 150], [30, 50]], dtype=np.float32)
    blue = np.array([[80, 40], [100, 30]], dtype=np.float32)

    # 1. NDVI without NIR -> Visible Vegetation Index (GLI/GRVI)
    ndvi_approx = engine.compute_ndvi(red, nir=None, green=green)
    assert ndvi_approx.shape == (2, 2)
    # At (0, 0): green(180) > red(100) -> positive vegetation signal
    assert ndvi_approx[0, 0] > 0.0
    # At (1, 1): red(200) > green(50) -> negative
    assert ndvi_approx[1, 1] < 0.0
    assert np.all(ndvi_approx >= -1.0) and np.all(ndvi_approx <= 1.0)

    # 2. NDWI without NIR -> Visible Water Contrast (Blue - Red)
    ndwi_approx = engine.compute_ndwi(green, nir=None, blue=blue, red=red)
    assert ndwi_approx.shape == (2, 2)
    # At (1, 0): blue(100) > red(20) -> positive water signal
    assert ndwi_approx[1, 0] > 0.0
    assert np.all(ndwi_approx >= -1.0) and np.all(ndwi_approx <= 1.0)

    # 3. CIR false-color composite on 3-band RGB
    rgb_stack = np.stack([red, green, blue], axis=0)  # (3, 2, 2)
    cir = engine.generate_cir_composite(rgb_stack)
    assert cir.shape == (2, 2, 3)
    assert cir.dtype == np.uint8

    # 4. Processing 3-band raster through process_raster
    res = engine.process_raster(rgb_stack)
    assert res.ndvi.shape == (2, 2)
    assert res.ndwi.shape == (2, 2)
    assert res.cir_composite is not None
    assert res.cir_composite.shape == (2, 2, 3)

