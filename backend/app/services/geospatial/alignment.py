"""OpenCV sub-pixel co-registration — CRS match, SIFT, RANSAC warp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def reproject_to_match(src_path: str, match_crs: str, out_path: str) -> str:
    """
    Aligns Coordinate Reference System projections via Lanczos4 resampling [39, 42, 43].
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, match_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": match_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=match_crs,
                    resampling=Resampling.lanczos,
                )
    return out_path


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim == 3:
        if array.shape[0] <= 16 and array.shape[0] < min(array.shape[1], array.shape[2]):
            gray = array.mean(axis=0)
        else:
            gray = array.mean(axis=2)
    else:
        gray = array
    gray = gray.astype(np.float32)
    gmin, gmax = float(gray.min()), float(gray.max())
    if gmax > gmin:
        gray = (gray - gmin) / (gmax - gmin)
    else:
        gray = np.zeros_like(gray)
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


def align_subpixel_images(ref_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    SIFT visual keypoint detection, descriptor matching, and RANSAC outlier warping [36, 39, 44].
    """
    ref_gray = _to_uint8_gray(ref_img)
    target_gray = _to_uint8_gray(target_img)

    # Initialize SIFT Keypoint Detector [36, 44]
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("Insufficient spatial feature correlation matches found between sensors.")

    # Fast Library for Approximate Nearest Neighbors (FLANN) Matcher [39]
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.asarray(des1, dtype=np.float32), np.asarray(des2, dtype=np.float32), k=2)

    # Lowe's Ratio Test to preserve distinct points
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Insufficient spatial feature correlation matches found between sensors.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography with robust RANSAC mapping [36, 39, 44]
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Insufficient spatial feature correlation matches found between sensors.")

    h, w = ref_gray.shape[:2]
    warped_target = cv2.warpPerspective(target_gray, H, (w, h))
    align_subpixel_images.last_inliers = int(mask.sum()) if mask is not None else 0  # type: ignore[attr-defined]

    return warped_target, H


@dataclass
class AlignmentResult:
    reference_path: Path
    moving_path: Path
    crs: str
    inliers: int
    homography: list[float] | None


class SpatialAligner:
    """Warp a moving raster onto the reference GeoTIFF grid (zero-pixel offset base)."""

    def reproject_to_crs(self, src_path: Path, dst_crs: str, dest: Path) -> Path:
        return Path(reproject_to_match(str(src_path), dst_crs, str(dest)))

    def resample_to_grid(self, src_path: Path, ref_path: Path, dest: Path) -> Path:
        with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
            profile = ref.profile.copy()
            profile.update({"count": src.count, "dtype": src.dtypes[0]})
            dest.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dest, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref.transform,
                        dst_crs=ref.crs,
                        resampling=Resampling.lanczos,
                    )
        return dest

    def sift_ransac(self, reference_gray: np.ndarray, moving_gray: np.ndarray) -> tuple[np.ndarray | None, int]:
        try:
            warped, homography = align_subpixel_images(reference_gray, moving_gray)
        except ValueError:
            return None, 0
        del warped
        inliers = int(getattr(align_subpixel_images, "last_inliers", 0))
        return homography, inliers

    def align_pair(self, reference: Path, moving: Path) -> AlignmentResult:
        work = settings.ARTIFACT_DIR / "alignment" / reference.stem
        work.mkdir(parents=True, exist_ok=True)
        with rasterio.open(reference) as ref:
            target_crs = ref.crs.to_string() if ref.crs else "EPSG:4326"
        reprojected = self.reproject_to_crs(moving, target_crs, work / f"{moving.stem}_reproj.tif")
        gridded = self.resample_to_grid(reprojected, reference, work / f"{moving.stem}_grid.tif")

        ref_gray = self._preview_gray(reference)
        mov_gray = self._preview_gray(gridded)
        homography, inliers = self.sift_ransac(ref_gray, mov_gray)
        if homography is not None:
            h, w = ref_gray.shape[:2]
            warped = cv2.warpPerspective(mov_gray, homography, (w, h))
            with rasterio.open(gridded) as src:
                profile = src.profile.copy()
                warped_path = work / f"{moving.stem}_warped.tif"
                profile.update({"count": 1, "dtype": "float32", "height": h, "width": w})
                with rasterio.open(warped_path, "w", **profile) as dst:
                    dst.write(warped.astype(np.float32), 1)
                gridded = warped_path
        logger.info(
            "alignment_complete",
            extra={"reference": str(reference), "moving": str(moving), "inliers": inliers},
        )
        coeffs = homography.flatten().tolist() if homography is not None else None
        if inliers < settings.SIFT_MIN_INLIERS:
            logger.warning("sift_inliers_low_using_grid_only", extra={"inliers": inliers})
        return AlignmentResult(
            reference_path=reference,
            moving_path=gridded,
            crs=target_crs,
            inliers=inliers,
            homography=coeffs,
        )

    @staticmethod
    def _preview_gray(path: Path, size: int = 512) -> np.ndarray:
        with rasterio.open(path) as src:
            band = src.read(1, out_shape=(size, size), resampling=Resampling.average)
        band = np.nan_to_num(band).astype(np.float32)
        if band.max() > band.min():
            band = (band - band.min()) / (band.max() - band.min())
        return (band * 255).astype(np.uint8)

    @staticmethod
    def bounds_wgs84(path: Path) -> tuple[float, float, float, float]:
        with rasterio.open(path) as src:
            return transform_bounds(src.crs, "EPSG:4326", *src.bounds)
