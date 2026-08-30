"""Phase 2 GIS preprocessing tests — synthetic arrays, no GPU or database."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from backend.preprocessing.alignment import AlignmentError, SubPixelAligner
from backend.preprocessing.core import (
    SAR_DB_CEILING,
    SAR_DB_FLOOR,
    RasterProcessingError,
    RasterProcessor,
)
from backend.preprocessing.spectral import SpectralIndexError, calculate_ndvi, calculate_ndwi

from scripts.download_datasets import DATASET_LAYOUT, DatasetDownloadManager


def test_ndvi_formula_and_zero_division() -> None:
    nir = np.array([[1.0, 0.8], [0.0, 0.5]], dtype=np.float32)
    red = np.array([[0.0, 0.2], [0.0, 0.5]], dtype=np.float32)
    ndvi = calculate_ndvi(nir, red)
    assert ndvi.shape == nir.shape
    assert ndvi.dtype == np.float32
    np.testing.assert_allclose(ndvi[0, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(ndvi[0, 1], 0.6, atol=1e-5)
    np.testing.assert_allclose(ndvi[1, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(ndvi[1, 1], 0.0, atol=1e-5)


def test_ndwi_formula() -> None:
    green = np.array([[0.9, 0.1], [0.0, 0.4]], dtype=np.float32)
    nir = np.array([[0.1, 0.9], [0.0, 0.4]], dtype=np.float32)
    ndwi = calculate_ndwi(green, nir)
    np.testing.assert_allclose(ndwi[0, 0], 0.8, atol=1e-5)
    np.testing.assert_allclose(ndwi[0, 1], -0.8, atol=1e-5)
    np.testing.assert_allclose(ndwi[1, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(ndwi[1, 1], 0.0, atol=1e-5)


def test_spectral_shape_mismatch_raises() -> None:
    with pytest.raises(SpectralIndexError):
        calculate_ndvi(np.ones((2, 2)), np.ones((3, 3)))


def _write_geotiff(path: Path) -> None:
    height, width = 16, 20
    transform = from_bounds(77.0, 28.0, 77.2, 28.2, width, height)
    data = np.linspace(100, 400, 3 * height * width, dtype=np.float32).reshape(3, height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.update_tags(TIFFTAG_DATETIME="2023:06:15 08:30:00", ACQUISITION_DATE="2023-06-15")


def test_extract_metadata(tmp_path: Path) -> None:
    path = tmp_path / "scene.tif"
    _write_geotiff(path)
    meta = RasterProcessor().extract_metadata(str(path))
    assert meta["crs"] in {"EPSG:4326", "OGC:CRS84"} or "4326" in str(meta["crs"])
    assert meta["band_count"] == 3
    assert meta["width"] == 20
    assert meta["height"] == 16
    assert len(meta["transform"]) == 6
    bounds = meta["bounds"]
    assert bounds["left"] == pytest.approx(77.0, abs=1e-6)
    assert bounds["bottom"] == pytest.approx(28.0, abs=1e-6)
    assert bounds["right"] == pytest.approx(77.2, abs=1e-6)
    assert bounds["top"] == pytest.approx(28.2, abs=1e-6)
    assert meta["resolution"]["x"] > 0
    assert meta["resolution"]["y"] > 0
    assert meta["acquisition_date"] == "2023-06-15"


def test_extract_metadata_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RasterProcessor().extract_metadata(str(tmp_path / "missing.tif"))


def test_optical_normalization_range() -> None:
    rng = np.random.default_rng(0)
    stack = rng.uniform(50, 4000, size=(4, 32, 32)).astype(np.float32)
    stack[0, 0, 0] = 1e5
    stack[0, 0, 1] = -50
    out = RasterProcessor().normalize_bands(stack, "optical")
    assert out.shape == stack.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_sar_decibel_clipping() -> None:
    intensity = np.array([[0.0, 1.0], [1e-8, 100.0]], dtype=np.float32)
    out = RasterProcessor().normalize_bands(intensity, "sar")
    assert out.shape == intensity.shape
    assert float(out.min()) >= SAR_DB_FLOOR
    assert float(out.max()) <= SAR_DB_CEILING
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-4)
    expected = 10.0 * np.log10(100.0 + 1e-12)
    np.testing.assert_allclose(out[1, 1], np.clip(expected, SAR_DB_FLOOR, SAR_DB_CEILING), atol=1e-4)


def test_unknown_modality_raises() -> None:
    with pytest.raises(RasterProcessingError):
        RasterProcessor().normalize_bands(np.ones((4, 4)), "lidar")


def test_reproject_raster_lanczos(tmp_path: Path) -> None:
    path = tmp_path / "wgs84.tif"
    _write_geotiff(path)
    array = RasterProcessor().reproject_raster(str(path), "EPSG:3857")
    assert array.ndim == 3
    assert array.shape[0] == 3
    assert array.dtype == np.float32
    assert array.shape[1] > 0 and array.shape[2] > 0


def _feature_rich_scene(size: int = 256) -> np.ndarray:
    """High-contrast synthetic grid so SIFT has stable corners."""
    image = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.indices((size, size))
    image += ((xx // 16 + yy // 16) % 2).astype(np.float32)
    rng = np.random.default_rng(7)
    for _ in range(18):
        cy = int(rng.integers(24, size - 24))
        cx = int(rng.integers(24, size - 24))
        radius = int(rng.integers(5, 12))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        image[mask] = 1.5 + rng.random()
    return image


def test_sift_ransac_alignment_recovers_translation() -> None:
    t1 = _feature_rich_scene()
    dx, dy = 6.0, -4.0
    matrix_fwd = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    import cv2

    t2 = cv2.warpAffine(t1, matrix_fwd, (t1.shape[1], t1.shape[0]), flags=cv2.INTER_CUBIC)
    aligner = SubPixelAligner(min_matches=8, prefer_affine=True)
    aligned, transform = aligner.align_pair(t1, t2)
    assert aligned.shape[:2] == t1.shape
    assert transform.shape in {(2, 3), (3, 3)}
    if transform.shape == (2, 3):
        recovered_dx, recovered_dy = float(transform[0, 2]), float(transform[1, 2])
    else:
        recovered_dx, recovered_dy = float(transform[0, 2]), float(transform[1, 2])
    # Inverse of the forward shift: T2 -> T1 ≈ (-dx, -dy)
    assert recovered_dx == pytest.approx(-dx, abs=1.5)
    assert recovered_dy == pytest.approx(-dy, abs=1.5)


def test_align_pair_rejects_empty() -> None:
    blank = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(AlignmentError):
        SubPixelAligner(min_matches=8).align_pair(blank, blank)


def test_dataset_scaffold_and_integrity(tmp_path: Path) -> None:
    manager = DatasetDownloadManager(tmp_path)
    created = manager.scaffold()
    assert set(created) == set(DATASET_LAYOUT)
    for name, children in DATASET_LAYOUT.items():
        root = tmp_path / name
        assert root.is_dir()
        for child in children:
            assert (root / child).is_dir()
        splits = manager.parse_metadata_splits(name)
        assert set(splits) == {"train", "val", "test"}

    ben = tmp_path / "bigearthnet"
    good = ben / "sentinel2" / "ok.tif"
    _write_geotiff(good)
    empty = ben / "sentinel2" / "empty.tif"
    empty.write_bytes(b"")
    garbage = ben / "sentinel2" / "bad.tif"
    garbage.write_bytes(b"not-a-tiff")
    (ben / "splits" / "splits.json").write_text(
        json.dumps({"train": ["ok"], "val": [], "test": []}),
        encoding="utf-8",
    )
    splits = manager.parse_metadata_splits("bigearthnet")
    assert splits["train"] == ["ok"]

    report = manager.verify_and_drop_corrupted("bigearthnet", quarantine=True)
    assert report.ok >= 1
    assert any("empty" in item for item in report.dropped)
    assert any("bad" in item for item in report.dropped)
    assert good.exists()
    assert not empty.exists()
    assert not garbage.exists()
    assert (tmp_path / "_quarantine" / "bigearthnet" / "sentinel2" / "empty.tif").exists()
