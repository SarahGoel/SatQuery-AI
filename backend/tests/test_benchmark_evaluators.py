"""Tests for the 4 ISRO Public Benchmark Evaluators.

Verifies end-to-end execution, CLI argument parsing, metric computation,
and graceful exit on missing data for:
- scripts/evaluate_vrsbench.py
- scripts/evaluate_rsvqa.py
- scripts/evaluate_cdvqa.py
- scripts/evaluate_bigearthnet.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

try:
    import backend.scripts.evaluate_bigearthnet as eval_bigearthnet
    import backend.scripts.evaluate_cdvqa as eval_cdvqa
    import backend.scripts.evaluate_rsvqa as eval_rsvqa
    import backend.scripts.evaluate_vrsbench as eval_vrsbench
except ModuleNotFoundError:
    try:
        import evaluate_bigearthnet as eval_bigearthnet
        import evaluate_cdvqa as eval_cdvqa
        import evaluate_rsvqa as eval_rsvqa
        import evaluate_vrsbench as eval_vrsbench
    except ModuleNotFoundError:
        import scripts.evaluate_bigearthnet as eval_bigearthnet
        import scripts.evaluate_cdvqa as eval_cdvqa
        import scripts.evaluate_rsvqa as eval_rsvqa
        import scripts.evaluate_vrsbench as eval_vrsbench
from app.schemas.trace import AuditableTraceLogSchema, InputMetadataSchema
from app.services.models.base import VLMResult


def _create_synthetic_tile(path: Path, shape=(32, 32), bands=3, value=120) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = shape
    transform = from_bounds(77.0, 28.0, 77.1, 28.1, w, h)
    data = np.ones((bands, h, w), dtype=np.float32) * float(value)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as ds:
        ds.write(data)
    return path


def test_evaluators_graceful_exit_when_missing_dir(tmp_path: Path) -> None:
    """Verifies that all 4 scripts exit with code 0 when benchmark directory does not exist."""
    missing_dir = str(tmp_path / "non_existent_benchmark_dir")

    for script_name in [
        "evaluate_vrsbench.py",
        "evaluate_rsvqa.py",
        "evaluate_cdvqa.py",
        "evaluate_bigearthnet.py",
    ]:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / script_name
        res = subprocess.run(
            [sys.executable, str(script_path), "--data-dir", missing_dir],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"{script_name} failed with code {res.returncode}: {res.stderr}"
        assert "Warning" in res.stdout or "not found" in res.stdout


def test_vrsbench_evaluator_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies VRSBench evaluation run, BLEU-4, ROUGE-L, CIDEr, and mIoU computation."""
    data_dir = tmp_path / "vrsbench"
    img_path = _create_synthetic_tile(data_dir / "images" / "scene_001.tif")
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "image": "scene_001.tif",
            "question": "Locate industrial warehouse rooftops",
            "answer": "Industrial warehouse rooftops located in the northern quadrant.",
            "bbox": [50.0, 50.0, 150.0, 150.0],
        }
    ]
    with open(splits_dir / "test_benchmark.json", "w") as f:
        json.dump(samples, f)

    out_file = tmp_path / "vrsbench_report.json"

    # Mock controller workflow execution
    class _MockController:
        last_bbox = [45.0, 48.0, 155.0, 148.0]
        def execute_workflow(self, query: str, filepaths, **kwargs):
            return AuditableTraceLogSchema(
                trace_id="VRS-001",
                task="single_grounding",
                query=query,
                input_metadata=InputMetadataSchema(crs="EPSG:4326", bounds=[0,0,1,1], affine_transform=[1,0,0,0,1,0], modalities=["RGB"]),
                output="Industrial warehouse rooftops located in the northern quadrant with clear boundaries.",
                confidence_score=0.91,
            )

    monkeypatch.setattr(eval_vrsbench, "SatQueryController", _MockController)

    test_args = ["evaluate_vrsbench.py", "--data-dir", str(data_dir), "--output", str(out_file)]
    with patch.object(sys, "argv", test_args):
        eval_vrsbench.main()

    assert out_file.exists()
    with open(out_file, "r") as f:
        rep = json.load(f)

    assert rep["benchmark"] == "VRSBench"
    assert rep["evaluated_samples"] == 1
    assert rep["metrics"]["mean_bleu4"] > 0.0
    assert rep["metrics"]["mean_rouge_l_f1"] > 0.0
    assert rep["metrics"]["mean_cider"] > 0.0
    assert rep["metrics"]["mean_bbox_miou"] is not None


def test_rsvqa_evaluator_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies RSVQA evaluation, accuracy grouping by question type."""
    data_dir = tmp_path / "rsvqa"
    _create_synthetic_tile(data_dir / "images" / "tile_001.tif")
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {"image": "tile_001.tif", "question": "Is there a water body?", "answer": "yes", "type": "presence"},
        {"image": "tile_001.tif", "question": "How many fuel tanks are visible?", "answer": "3", "type": "count"},
    ]
    with open(splits_dir / "test_benchmark.json", "w") as f:
        json.dump(samples, f)

    out_file = tmp_path / "rsvqa_report.json"

    class _MockController:
        def execute_workflow(self, query: str, filepaths, **kwargs):
            ans = "Yes, a water reservoir is present." if "water" in query else "There are 3 fuel tanks detected."
            return AuditableTraceLogSchema(
                trace_id="RSVQA-001",
                task="single_vqa",
                query=query,
                input_metadata=InputMetadataSchema(crs="EPSG:4326", bounds=[0,0,1,1], affine_transform=[1,0,0,0,1,0], modalities=["RGB"]),
                output=ans,
                confidence_score=0.93,
            )

    monkeypatch.setattr(eval_rsvqa, "SatQueryController", _MockController)

    test_args = ["evaluate_rsvqa.py", "--data-dir", str(data_dir), "--output", str(out_file)]
    with patch.object(sys, "argv", test_args):
        eval_rsvqa.main()

    assert out_file.exists()
    with open(out_file, "r") as f:
        rep = json.load(f)

    assert rep["benchmark"] == "RSVQA"
    assert rep["evaluated_samples"] == 2
    assert "presence" in rep["by_type"]
    assert "count" in rep["by_type"]
    assert rep["metrics"]["overall_accuracy"] > 0.0


def test_cdvqa_evaluator_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies CDVQA bitemporal change detection evaluation and mIoU score."""
    data_dir = tmp_path / "cdvqa"
    _create_synthetic_tile(data_dir / "t1" / "t1_001.tif")
    _create_synthetic_tile(data_dir / "t2" / "t2_001.tif")
    mask_path = _create_synthetic_tile(data_dir / "masks" / "mask_001.tif", bands=1, value=1)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "image_t1": "t1_001.tif",
            "image_t2": "t2_001.tif",
            "question": "Has the built-up area increased, decreased, or remained unchanged?",
            "answer": "[INCREASED] Built-up expanded by 15%.",
            "mask": "mask_001.tif",
        }
    ]
    with open(splits_dir / "test_benchmark.json", "w") as f:
        json.dump(samples, f)

    out_file = tmp_path / "cdvqa_report.json"

    class _MockController:
        last_change_mask = np.ones((32, 32), dtype=np.uint8)
        def execute_workflow(self, query: str, filepaths, **kwargs):
            return AuditableTraceLogSchema(
                trace_id="CDVQA-001",
                task="bitemporal_change",
                query=query,
                input_metadata=InputMetadataSchema(crs="EPSG:4326", bounds=[0,0,1,1], affine_transform=[1,0,0,0,1,0], modalities=["RGB"]),
                output="[INCREASED] Significant new built-up construction detected across observation date.",
                confidence_score=0.92,
            )

    monkeypatch.setattr(eval_cdvqa, "SatQueryController", _MockController)

    test_args = ["evaluate_cdvqa.py", "--data-dir", str(data_dir), "--output", str(out_file)]
    with patch.object(sys, "argv", test_args):
        eval_cdvqa.main()

    assert out_file.exists()
    with open(out_file, "r") as f:
        rep = json.load(f)

    assert rep["benchmark"] == "CDVQA"
    assert rep["evaluated_samples"] == 1
    assert rep["metrics"]["directional_accuracy"] == 1.0
    assert rep["metrics"]["mean_change_miou"] == 1.0


def test_bigearthnet_evaluator_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies BigEarthNet multi-label evaluation and Macro-F1 computation."""
    data_dir = tmp_path / "bigearthnet"
    _create_synthetic_tile(data_dir / "sentinel2" / "opt_001.tif", bands=4)
    _create_synthetic_tile(data_dir / "sentinel1" / "sar_001.tif", bands=2)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "image_opt": "opt_001.tif",
            "image_sar": "sar_001.tif",
            "labels": ["Urban fabric", "Inland waters"],
        }
    ]
    with open(splits_dir / "test_benchmark.json", "w") as f:
        json.dump(samples, f)

    out_file = tmp_path / "bigearthnet_report.json"

    # Mock CrossModalAnalysisTool
    class _MockResult:
        answer = "Joint optical and radar analysis mapped urban fabric and inland waters."
        params = {"builtup_features_count": 5, "water_features_count": 2}

    class _MockTool:
        def analyze(self, *args, **kwargs):
            return _MockResult()

    monkeypatch.setattr(eval_bigearthnet, "CrossModalAnalysisTool", _MockTool)

    test_args = ["evaluate_bigearthnet.py", "--data-dir", str(data_dir), "--output", str(out_file)]
    with patch.object(sys, "argv", test_args):
        eval_bigearthnet.main()

    assert out_file.exists()
    with open(out_file, "r") as f:
        rep = json.load(f)

    assert rep["benchmark"] == "BigEarthNet"
    assert rep["evaluated_samples"] == 1
    assert rep["metrics"]["macro_f1"] > 0.0
    assert rep["metrics"]["subset_accuracy"] == 1.0
    assert "Urban fabric" in rep["class_metrics"]
    assert "Inland waters" in rep["class_metrics"]
