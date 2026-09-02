"""Phase 2 Step 4 — GeoTIFF spatial metadata parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.services.geospatial.parser import parse_geotiff_metadata


def _write_raster(
    path: Path,
    *,
    count: int,
    crs: str = "EPSG:4326",
    tags: dict[str, str] | None = None,
    descriptions: tuple[str, ...] | None = None,
) -> None:
    height, width = 16, 20
    transform = from_bounds(77.0, 28.0, 77.2, 28.2, width, height)
    data = np.linspace(1, 100, count * height * width, dtype=np.float32).reshape(count, height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
        if descriptions:
            dst.descriptions = descriptions


def test_parse_geotiff_metadata_crs_and_affine(tmp_path: Path) -> None:
    path = tmp_path / "test.tif"
    _write_raster(path, count=3)
    meta = parse_geotiff_metadata(str(path))
    assert "crs" in meta
    assert len(meta["affine_transform"]) == 6
    assert meta["band_count"] == 3
    assert meta["modality"] == "RGB"
    assert len(meta["bounds"]) == 4


def test_parse_optical_multispectral_cartosat(tmp_path: Path) -> None:
    path = tmp_path / "cartosat.tif"
    _write_raster(path, count=4)
    meta = parse_geotiff_metadata(str(path))
    assert meta["modality"] == "Optical/Multispectral"
    assert meta["modalities"] == ["Red", "Green", "Blue", "NIR"]


def test_parse_sar_risat_with_backscatter_tags(tmp_path: Path) -> None:
    path = tmp_path / "risat.tif"
    _write_raster(
        path,
        count=2,
        tags={"SENSOR": "RISAT", "PRODUCT": "sigma0", "POLARIZATION": "VV VH"},
        descriptions=("VV", "VH"),
    )
    meta = parse_geotiff_metadata(str(path))
    assert meta["modality"] == "SAR"
    assert "VV" in meta["modalities"]


def test_parse_sar_single_band(tmp_path: Path) -> None:
    path = tmp_path / "sar_vv.tif"
    _write_raster(path, count=1, tags={"backscatter": "sigma0", "POLARIZATION": "VV"})
    meta = parse_geotiff_metadata(str(path))
    assert meta["modality"] == "SAR"
    assert meta["modalities"] == ["VV"]


def test_parse_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_geotiff_metadata(str(tmp_path / "missing.tif"))
