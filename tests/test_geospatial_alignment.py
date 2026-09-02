"""Phase 2 Step 5 — reprojection and SIFT/RANSAC sub-pixel alignment."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from app.services.geospatial.alignment import align_subpixel_images, reproject_to_match


def _feature_rich_scene(size: int = 256) -> np.ndarray:
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


def _write_geotiff(path: Path, crs: str) -> None:
    height, width = 32, 32
    transform = from_bounds(77.0, 28.0, 77.2, 28.2, width, height)
    data = np.linspace(100, 400, height * width, dtype=np.float32).reshape(1, height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


def test_reproject_to_match_changes_crs(tmp_path: Path) -> None:
    src = tmp_path / "wgs84.tif"
    out = tmp_path / "webmerc.tif"
    _write_geotiff(src, "EPSG:4326")
    result = reproject_to_match(str(src), "EPSG:3857", str(out))
    assert Path(result).exists()
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 3857


def test_align_subpixel_images_recovers_shift() -> None:
    ref_img = _feature_rich_scene()
    dx, dy = 3.0, -2.0
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    target_img = cv2.warpAffine(
        ref_img, matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_CUBIC
    )
    warped_target, homography = align_subpixel_images(ref_img, target_img)
    assert warped_target.shape[:2] == ref_img.shape
    assert homography.shape == (3, 3)
    overlap = (ref_img > 0) & (warped_target > 0)
    if overlap.any():
        error = float(
            np.mean((ref_img[overlap] - warped_target[overlap].astype(np.float32) / 255.0) ** 2)
        )
        assert error < 0.5
