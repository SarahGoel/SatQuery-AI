"""Tests for Synthetic Benchmark Data Staging Script (scripts/stage_mock_datasets.py).

Verifies programmatic generation of GeoTIFF patches, binary change masks,
directory structures, and compliant benchmark JSON splits for:
- VRSBench
- RSVQA
- CDVQA
- BigEarthNet
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

try:
    import backend.scripts.stage_mock_datasets as stage_mod
except ModuleNotFoundError:
    try:
        import stage_mock_datasets as stage_mod
    except ModuleNotFoundError:
        import scripts.stage_mock_datasets as stage_mod


def test_create_synthetic_geotiff(tmp_path: Path) -> None:
    """Verifies that synthetic GeoTIFFs are written with valid CRS, shape, and transform."""
    out_tif = tmp_path / "test_tile.tif"
    stage_mod.create_synthetic_geotiff(
        filepath=out_tif,
        shape=(256, 256),
        bands=3,
        dtype="uint8",
        pattern="urban",
        bounds=(77.0, 28.0, 77.05, 28.05),
    )

    assert out_tif.exists()
    with rasterio.open(out_tif) as src:
        assert src.width == 256
        assert src.height == 256
        assert src.count == 3
        assert src.crs.to_string() == "EPSG:4326"
        assert src.bounds.left == pytest.approx(77.0, abs=1e-4)
        assert src.bounds.top == pytest.approx(28.05, abs=1e-4)
        arr = src.read()
        assert arr.shape == (3, 256, 256)
        assert arr.dtype == np.uint8


def test_create_synthetic_mask(tmp_path: Path) -> None:
    """Verifies that binary change masks are written correctly as single-band GeoTIFF."""
    out_mask = tmp_path / "test_mask.tif"
    stage_mod.create_synthetic_mask(
        filepath=out_mask,
        shape=(256, 256),
        bbox=(64, 64, 192, 192),
    )

    assert out_mask.exists()
    with rasterio.open(out_mask) as src:
        assert src.width == 256
        assert src.height == 256
        assert src.count == 1
        mask_arr = src.read(1)
        assert mask_arr.dtype == np.uint8
        # Check that center has positive values and border has 0
        assert mask_arr[100, 100] == 255
        assert mask_arr[10, 10] == 0


def test_stage_all_benchmarks_structure_and_schema(tmp_path: Path) -> None:
    """Verifies that stage_all_benchmarks builds compliant folders and annotation splits."""
    base_dir = tmp_path / "mock_data"
    results = stage_mod.stage_all_benchmarks(base_dir=base_dir, num_samples=5, overwrite=True)

    assert "vrsbench" in results
    assert "rsvqa" in results
    assert "cdvqa" in results
    assert "bigearthnet" in results

    # 1. VRSBench inspection
    vrs_dir = base_dir / "vrsbench"
    vrs_split = vrs_dir / "splits" / "test_benchmark.json"
    assert vrs_split.exists()
    with open(vrs_split, "r", encoding="utf-8") as f:
        vrs_records = json.load(f)
    assert len(vrs_records) == 5
    for rec in vrs_records:
        assert "image" in rec
        assert "question" in rec
        assert "answer" in rec
        assert "bbox" in rec
        assert len(rec["bbox"]) == 4
        assert (vrs_dir / rec["image"]).exists()

    # 2. RSVQA inspection
    rsvqa_dir = base_dir / "rsvqa"
    rsvqa_split = rsvqa_dir / "splits" / "test_benchmark.json"
    assert rsvqa_split.exists()
    with open(rsvqa_split, "r", encoding="utf-8") as f:
        rsvqa_records = json.load(f)
    assert len(rsvqa_records) == 5
    for rec in rsvqa_records:
        assert "image" in rec
        assert "question" in rec
        assert "answer" in rec
        assert "type" in rec
        assert (rsvqa_dir / rec["image"]).exists()

    # 3. CDVQA inspection
    cdvqa_dir = base_dir / "cdvqa"
    cdvqa_split = cdvqa_dir / "splits" / "test_benchmark.json"
    assert cdvqa_split.exists()
    with open(cdvqa_split, "r", encoding="utf-8") as f:
        cdvqa_records = json.load(f)
    assert len(cdvqa_records) == 5
    for rec in cdvqa_records:
        assert "image_t1" in rec
        assert "image_t2" in rec
        assert "question" in rec
        assert "answer" in rec
        assert "mask" in rec
        assert (cdvqa_dir / rec["image_t1"]).exists()
        assert (cdvqa_dir / rec["image_t2"]).exists()
        assert (cdvqa_dir / rec["mask"]).exists()

    # 4. BigEarthNet inspection
    ben_dir = base_dir / "bigearthnet"
    ben_split = ben_dir / "splits" / "test_benchmark.json"
    assert ben_split.exists()
    with open(ben_split, "r", encoding="utf-8") as f:
        ben_records = json.load(f)
    assert len(ben_records) == 5
    for rec in ben_records:
        assert "image_opt" in rec
        assert "image_sar" in rec
        assert "labels" in rec
        assert isinstance(rec["labels"], list) and len(rec["labels"]) > 0
        assert (ben_dir / rec["image_opt"]).exists()
        assert (ben_dir / rec["image_sar"]).exists()


def test_cli_execution_standalone(tmp_path: Path) -> None:
    """Verifies CLI execution of scripts/stage_mock_datasets.py."""
    script_path = Path(stage_mod.__file__).resolve()
    target_dir = tmp_path / "cli_stage"

    cmd = [
        sys.executable,
        str(script_path),
        "--base-dir",
        str(target_dir),
        "--num-samples",
        "5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI staging failed: {res.stderr}"
    assert "SYNTHETIC BENCHMARK DATA STAGING COMPLETE" in res.stdout
    assert (target_dir / "vrsbench" / "splits" / "test_benchmark.json").exists()
    assert (target_dir / "rsvqa" / "splits" / "test_benchmark.json").exists()
    assert (target_dir / "cdvqa" / "splits" / "test_benchmark.json").exists()
    assert (target_dir / "bigearthnet" / "splits" / "test_benchmark.json").exists()
