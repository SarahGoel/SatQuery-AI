"""Phase 3 — spectral indices and Optical-SAR cross-attention fusion."""

from __future__ import annotations

import numpy as np
import torch

from app.services.geospatial.spectral import calculate_spectral_indices, generate_n_channel_tensor
from app.services.models.fusion import CrossAttentionOpticalSAR


def test_calculate_spectral_indices_shapes_and_formulas() -> None:
    rng = np.random.default_rng(0)
    r = rng.random((100, 100)).astype(np.float32)
    g = rng.random((100, 100)).astype(np.float32)
    nir = rng.random((100, 100)).astype(np.float32)
    ndvi, ndwi = calculate_spectral_indices(r, g, nir)
    assert ndvi.shape == (100, 100)
    assert ndwi.shape == (100, 100)
    assert ndvi.dtype == np.float32
    assert ndwi.dtype == np.float32
    expected_ndvi = (nir - r) / (nir + r)
    expected_ndwi = (g - nir) / (g + nir)
    np.testing.assert_allclose(ndvi, expected_ndvi, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ndwi, expected_ndwi, rtol=1e-5, atol=1e-6)


def test_ndvi_ndwi_zero_division() -> None:
    red = np.zeros((2, 2), dtype=np.float32)
    green = np.zeros((2, 2), dtype=np.float32)
    nir = np.zeros((2, 2), dtype=np.float32)
    ndvi, ndwi = calculate_spectral_indices(red, green, nir)
    np.testing.assert_array_equal(ndvi, np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(ndwi, np.zeros((2, 2), dtype=np.float32))


def test_generate_n_channel_tensor_is_5ch() -> None:
    rng = np.random.default_rng(1)
    r, g, nir = rng.random((100, 100)), rng.random((100, 100)), rng.random((100, 100))
    ndvi, ndwi = calculate_spectral_indices(r, g, nir)
    tensor = generate_n_channel_tensor(np.stack([r, g, r]), ndvi, ndwi)
    assert tensor.shape[0] == 5
    assert tensor.shape == (5, 100, 100)
    assert tensor.dtype == torch.float32


def test_cross_attention_optical_sar_shapes() -> None:
    fusion = CrossAttentionOpticalSAR(d_model=256, num_heads=8)
    opt = torch.randn(1, 64, 256)
    sar = torch.randn(1, 64, 256)
    fused = fusion(opt, sar)
    assert fused.shape == (1, 64, 256)
