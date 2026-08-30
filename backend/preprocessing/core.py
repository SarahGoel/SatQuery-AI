"""GeoTIFF ingest, metadata parsing, reprojection, and modality normalization.

Phase 2 GIS core for SatQuery AI (SIH26167). No database or ML inference.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

logger = logging.getLogger(__name__)

SAR_DB_FLOOR = -40.0
SAR_DB_CEILING = 10.0
OPTICAL_LOW_PERCENTILE = 2.0
OPTICAL_HIGH_PERCENTILE = 98.0
_EPS = 1e-12

_ACQUISITION_TAG_KEYS: tuple[str, ...] = (
    "TIFFTAG_DATETIME",
    "ACQUISITION_DATE",
    "acquisition_date",
    "ACQUISITIONDATETIME",
    "SENSING_TIME",
    "sensing_time",
    "PRODUCT_START_TIME",
    "DATE_ACQUIRED",
    "TIMEOFCOLLECTION",
    "Img_Acquisition_Date",
    "IMAGING_DATE",
)


class RasterProcessingError(RuntimeError):
    """Raised when a GeoTIFF cannot be read, reprojected, or normalized."""


class RasterProcessor:
    """Raster I/O and physical preprocessing for optical and SAR GeoTIFFs."""

    def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Parse CRS, affine transform, bounds, bands, resolution, and date tags.

        Parameters
        ----------
        file_path:
            Path to a GeoTIFF (or GDAL-readable raster).

        Returns
        -------
        dict
            Serialisable metadata dictionary.

        Raises
        ------
        FileNotFoundError
            If ``file_path`` does not exist.
        RasterProcessingError
            If GDAL/Rasterio cannot open the file or CRS is missing.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Raster not found: {path}")

        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise RasterProcessingError(f"Raster has no CRS: {path}")
                transform: Affine = src.transform
                tags = dict(src.tags())
                band_tags = {idx: dict(src.tags(idx)) for idx in range(1, src.count + 1)}
                acquisition = _parse_acquisition_date(tags, band_tags.values())
                bounds = src.bounds
                res_x, res_y = src.res
                metadata: dict[str, Any] = {
                    "path": str(path.resolve()),
                    "driver": src.driver,
                    "crs": src.crs.to_string(),
                    "epsg": src.crs.to_epsg(),
                    "transform": list(transform)[:6],
                    "affine": {
                        "a": float(transform.a),
                        "b": float(transform.b),
                        "c": float(transform.c),
                        "d": float(transform.d),
                        "e": float(transform.e),
                        "f": float(transform.f),
                    },
                    "bounds": {
                        "left": float(bounds.left),
                        "bottom": float(bounds.bottom),
                        "right": float(bounds.right),
                        "top": float(bounds.top),
                    },
                    "width": int(src.width),
                    "height": int(src.height),
                    "band_count": int(src.count),
                    "dtype": src.dtypes[0] if src.dtypes else None,
                    "nodata": src.nodata,
                    "resolution": {
                        "x": float(res_x),
                        "y": float(res_y),
                    },
                    "acquisition_date": acquisition,
                    "tags": tags,
                }
        except RasterProcessingError:
            raise
        except RasterioIOError as exc:
            raise RasterProcessingError(f"Failed to open raster {path}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RasterProcessingError(f"Metadata extraction failed for {path}: {exc}") from exc

        logger.info("extract_metadata path=%s crs=%s", path, metadata["crs"])
        return metadata

    def reproject_raster(
        self,
        source_path: str,
        target_crs: str,
        target_transform: Affine | Iterable[float] | None = None,
        target_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Reproject a GeoTIFF onto ``target_crs`` with Lanczos (window-4) resampling.

        Parameters
        ----------
        source_path:
            Source GeoTIFF path.
        target_crs:
            Destination CRS (e.g. ``EPSG:4326`` or a WKT/PROJ string).
        target_transform:
            Optional destination affine. When omitted, a default transform is
            computed from the source footprint.
        target_shape:
            Optional ``(height, width)``. Required when ``target_transform`` is
            supplied; otherwise derived from ``calculate_default_transform``.

        Returns
        -------
        np.ndarray
            Reprojected array with shape ``(bands, height, width)``, float32.
        """
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Raster not found: {path}")

        try:
            dst_crs = CRS.from_user_input(target_crs)
        except Exception as exc:  # noqa: BLE001
            raise RasterProcessingError(f"Invalid target CRS {target_crs!r}: {exc}") from exc

        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise RasterProcessingError(f"Source raster has no CRS: {path}")
                dst_transform, dst_width, dst_height = _resolve_destination_grid(
                    src, dst_crs, target_transform, target_shape
                )
                destination = np.zeros((src.count, dst_height, dst_width), dtype=np.float32)
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=destination[band_idx - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.lanczos,
                        src_nodata=src.nodata,
                        dst_nodata=0.0,
                    )
        except RasterProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RasterProcessingError(f"Reprojection failed for {path}: {exc}") from exc

        logger.info(
            "reproject_raster path=%s target_crs=%s shape=%s",
            path,
            dst_crs.to_string(),
            destination.shape,
        )
        return destination

    def normalize_bands(self, raster_array: np.ndarray, modality: str) -> np.ndarray:
        """Normalize a raster stack for optical (percentile) or SAR (dB) physics.

        Optical
            Independent 2%–98% percentile stretch per band, clipped to ``[0, 1]``.
        SAR
            Intensity to decibels: ``10 * log10(I + eps)``, clipped to
            ``[SAR_DB_FLOOR, SAR_DB_CEILING]``.

        Accepts ``(H, W)`` or ``(C, H, W)``. Returns the same rank, float32.
        """
        if raster_array is None or raster_array.size == 0:
            raise RasterProcessingError("Cannot normalize an empty raster array.")

        kind = modality.strip().lower()
        array = np.asarray(raster_array, dtype=np.float64)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

        if kind in {"optical", "opt", "s2", "sentinel-2", "sentinel2", "cartosat"}:
            stretched = _percentile_stretch(array)
        elif kind in {"sar", "s1", "sentinel-1", "sentinel1", "risat"}:
            stretched = _sar_decibel(array)
        else:
            raise RasterProcessingError(
                f"Unknown modality {modality!r}. Expected 'optical' or 'sar'."
            )
        return stretched.astype(np.float32)


def _resolve_destination_grid(
    src: rasterio.DatasetReader,
    dst_crs: CRS,
    target_transform: Affine | Iterable[float] | None,
    target_shape: tuple[int, int] | None,
) -> tuple[Affine, int, int]:
    if target_transform is not None:
        transform = (
            target_transform
            if isinstance(target_transform, Affine)
            else Affine(*list(target_transform)[:6])
        )
        if target_shape is None:
            raise RasterProcessingError(
                "target_shape=(height, width) is required when target_transform is set."
            )
        height, width = int(target_shape[0]), int(target_shape[1])
        if height <= 0 or width <= 0:
            raise RasterProcessingError(f"Invalid target_shape {target_shape!r}.")
        return transform, width, height

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    return transform, int(width), int(height)


def _percentile_stretch(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return _stretch_band(array)
    if array.ndim != 3:
        raise RasterProcessingError(f"Expected 2-D or 3-D array, got shape {array.shape}.")
    bands = [_stretch_band(array[i]) for i in range(array.shape[0])]
    return np.stack(bands, axis=0)


def _stretch_band(band: np.ndarray) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float64)
    low, high = np.percentile(finite, [OPTICAL_LOW_PERCENTILE, OPTICAL_HIGH_PERCENTILE])
    span = float(high - low)
    if span < _EPS:
        return np.zeros_like(band, dtype=np.float64)
    stretched = (band - low) / span
    return np.clip(stretched, 0.0, 1.0)


def _sar_decibel(array: np.ndarray) -> np.ndarray:
    intensity = np.maximum(array, 0.0)
    db = 10.0 * np.log10(intensity + _EPS)
    return np.clip(db, SAR_DB_FLOOR, SAR_DB_CEILING)


def _parse_acquisition_date(
    tags: Mapping[str, Any],
    band_tag_groups: Iterable[Mapping[str, Any]],
) -> str | None:
    candidates: list[str] = []
    for key in _ACQUISITION_TAG_KEYS:
        value = tags.get(key)
        if value:
            candidates.append(str(value))
    for group in band_tag_groups:
        for key in _ACQUISITION_TAG_KEYS:
            value = group.get(key)
            if value:
                candidates.append(str(value))
    for raw in candidates:
        parsed = _coerce_datetime(raw)
        if parsed is not None:
            return parsed
    return candidates[0] if candidates else None


def _coerce_datetime(raw: str) -> str | None:
    text = raw.strip()
    layouts = (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d",
    )
    for fmt in layouts:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10]
    return None
