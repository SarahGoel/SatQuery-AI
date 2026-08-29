"""Feature 2.1 — reprojection, grid resampling, SIFT, RANSAC registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlignmentResult:
    reference_path: Path
    moving_path: Path
    crs: str
    inliers: int
    homography: list[float] | None


class SpatialAligner:
    """Warp a moving raster onto the reference GeoTIFF grid."""

    def reproject_to_crs(self, src_path: Path, dst_crs: str, dest: Path) -> Path:
        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(
                {
                    "crs": dst_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dest, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
        return dest

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
                        resampling=Resampling.bilinear,
                    )
        return dest

    def sift_ransac(self, reference_gray: np.ndarray, moving_gray: np.ndarray) -> tuple[np.ndarray | None, int]:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(reference_gray, None)
        kp2, des2 = sift.detectAndCompute(moving_gray, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, 0
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < settings.SIFT_MIN_INLIERS:
            return None, 0
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        return homography, inliers

    def align_pair(self, reference: Path, moving: Path) -> AlignmentResult:
        work = settings.ARTIFACT_DIR / "alignment" / reference.stem
        work.mkdir(parents=True, exist_ok=True)
        with rasterio.open(reference) as ref:
            target_crs = ref.crs.to_string()
        reprojected = self.reproject_to_crs(moving, target_crs, work / f"{moving.stem}_reproj.tif")
        gridded = self.resample_to_grid(reprojected, reference, work / f"{moving.stem}_grid.tif")

        ref_gray = self._preview_gray(reference)
        mov_gray = self._preview_gray(gridded)
        homography, inliers = self.sift_ransac(ref_gray, mov_gray)
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
