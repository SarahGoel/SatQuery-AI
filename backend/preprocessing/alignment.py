"""Sub-pixel co-registration of bi-temporal raster grids via SIFT + RANSAC.

Warps T2 onto the T1 pixel grid. No database sessions or inference models.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 3.0
MIN_MATCH_COUNT = 8
FLANN_INDEX_KDTREE = 1


class AlignmentError(RuntimeError):
    """Raised when SIFT/RANSAC cannot estimate a stable warp from T2 to T1."""


class SubPixelAligner:
    """SIFT feature matching, Lowe ratio test, RANSAC affine/homography warp."""

    def __init__(
        self,
        *,
        ratio_threshold: float = LOWE_RATIO,
        ransac_reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
        min_matches: int = MIN_MATCH_COUNT,
        prefer_affine: bool = True,
    ) -> None:
        self.ratio_threshold = ratio_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_matches = min_matches
        self.prefer_affine = prefer_affine
        try:
            self._sift = cv2.SIFT_create()
        except Exception as exc:  # noqa: BLE001
            raise AlignmentError(
                "OpenCV SIFT is unavailable. Install opencv-python-headless>=4.4."
            ) from exc

    def align_pair(
        self, image_t1: np.ndarray, image_t2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Register ``image_t2`` onto the ``image_t1`` grid.

        Pipeline
            1. Convert both grids to uint8 grayscale.
            2. Extract SIFT keypoints/descriptors.
            3. FLANN kNN match with BF-Matcher fallback; Lowe's ratio test.
            4. RANSAC affine (default) or homography to reject outliers.
            5. Cubic ``warpAffine`` / ``warpPerspective`` of T2.

        Returns
        -------
        aligned_t2, transform_matrix
            Warped T2 (same spatial size as T1) and the 2x3 affine or 3x3
            homography that maps T2 pixel coordinates onto T1.
        """
        gray_t1 = _to_uint8_gray(image_t1)
        gray_t2 = _to_uint8_gray(image_t2)
        height, width = gray_t1.shape

        kp1, des1 = self._sift.detectAndCompute(gray_t1, None)
        kp2, des2 = self._sift.detectAndCompute(gray_t2, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            raise AlignmentError("Insufficient SIFT keypoints on T1/T2 for co-registration.")

        good = self._match_descriptors(des1, des2)
        if len(good) < self.min_matches:
            raise AlignmentError(
                f"Only {len(good)} Lowe-inlier matches (need {self.min_matches})."
            )

        # Matches are (query=T1, train=T2). Warp maps T2 -> T1.
        pts_t1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_t2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, warp_kind, inliers = self._estimate_transform(pts_t2, pts_t1)
        if matrix is None or inliers < self.min_matches:
            raise AlignmentError(
                f"RANSAC failed (inliers={inliers}, kind={warp_kind})."
            )

        aligned = self._warp(image_t2, matrix, (width, height), warp_kind)
        logger.info(
            "align_pair matches=%d inliers=%d kind=%s shape=%s",
            len(good),
            inliers,
            warp_kind,
            aligned.shape,
        )
        return aligned, matrix

    def _match_descriptors(self, des1: np.ndarray, des2: np.ndarray) -> list[cv2.DMatch]:
        descriptors_t1 = np.asarray(des1, dtype=np.float32)
        descriptors_t2 = np.asarray(des2, dtype=np.float32)
        knn = self._knn_match(descriptors_t1, descriptors_t2)
        good: list[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < self.ratio_threshold * second.distance:
                good.append(best)
        return good

    def _knn_match(self, des1: np.ndarray, des2: np.ndarray) -> list:
        try:
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=64)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            return matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            logger.warning("FLANN matcher failed; falling back to BFMatcher(L2).")
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            return matcher.knnMatch(des1, des2, k=2)

    def _estimate_transform(
        self, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> tuple[np.ndarray | None, str, int]:
        if self.prefer_affine:
            affine, mask = cv2.estimateAffine2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
                maxIters=5000,
                confidence=0.99,
            )
            inliers = int(mask.sum()) if mask is not None else 0
            if affine is not None and inliers >= self.min_matches:
                return affine.astype(np.float64), "affine", inliers

        homography, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None:
            return None, "homography", inliers
        return homography.astype(np.float64), "homography", inliers

    def _warp(
        self,
        image_t2: np.ndarray,
        matrix: np.ndarray,
        dsize: tuple[int, int],
        warp_kind: str,
    ) -> np.ndarray:
        array = np.asarray(image_t2)
        flags = cv2.INTER_CUBIC
        border = cv2.BORDER_CONSTANT

        def _warp_plane(plane: np.ndarray) -> np.ndarray:
            if warp_kind == "affine":
                return cv2.warpAffine(plane, matrix, dsize, flags=flags, borderMode=border)
            return cv2.warpPerspective(plane, matrix, dsize, flags=flags, borderMode=border)

        if array.ndim == 2:
            return _warp_plane(array)
        if array.ndim != 3:
            raise AlignmentError(f"Unsupported T2 rank {array.ndim}.")

        # Channel-first (C, H, W) vs channel-last (H, W, C).
        if array.shape[0] <= 16 and array.shape[0] < min(array.shape[1], array.shape[2]):
            planes = [_warp_plane(array[i]) for i in range(array.shape[0])]
            return np.stack(planes, axis=0)
        planes = [_warp_plane(array[:, :, i]) for i in range(array.shape[2])]
        return np.stack(planes, axis=2)


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim == 3:
        if array.shape[0] <= 16 and array.shape[0] < min(array.shape[1], array.shape[2]):
            gray = array.mean(axis=0)
        else:
            gray = array.mean(axis=2)
    elif array.ndim == 2:
        gray = array
    else:
        raise AlignmentError(f"Cannot convert shape {array.shape} to grayscale.")
    gmin, gmax = float(gray.min()), float(gray.max())
    if gmax > gmin:
        gray = (gray - gmin) / (gmax - gmin)
    else:
        gray = np.zeros_like(gray)
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)
