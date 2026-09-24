#!/usr/bin/env python3
"""ISRO SIH Public Benchmark Evaluator — CDVQA (Change Detection VQA).

Evaluates Change-based Visual Question Answering (CDVQA) on bi-temporal satellite
pairs (T1 baseline vs T2 observation).
Computes quantitative evaluation metrics:
- BLEU-4: Text similarity between reference change explanation and generated output.
- ROUGE-L: Longest Common Subsequence F1 for structural change description accuracy.
- mIoU: Mean Intersection over Union between predicted change segmentation masks
        and ground-truth raster change masks.
- Directional Accuracy: Correctness of trend verdicts ([INCREASED] / [DECREASED] / [REMAINED UNCHANGED]).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure backend directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
repo_root = backend_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.evaluation.metrics import calculate_bleu4, calculate_miou, calculate_rouge_l
from app.services.agent import SatQueryController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_cdvqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SatQuery on CDVQA benchmark")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/cdvqa",
        help="Path to CDVQA root directory containing splits/test_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/cdvqa_eval.json",
        help="Destination path for evaluation metric report JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of test samples to evaluate",
    )
    return parser.parse_args()


def resolve_image_path(data_dir: Path, image_name: str, subfolder: str = "") -> Path:
    p = Path(image_name)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        data_dir / p,
        data_dir / subfolder / p if subfolder else data_dir / p,
        data_dir / p.name,
        data_dir / subfolder / p.name if subfolder else data_dir / p.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return data_dir / subfolder / p.name if subfolder else data_dir / p.name


def load_mask(mask_path: Path, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Loads a ground-truth raster change mask from disk as a binary 2D array."""
    if not mask_path.exists():
        return None
    try:
        import rasterio
        with rasterio.open(mask_path) as ds:
            arr = ds.read(1)
    except Exception:
        try:
            from PIL import Image
            with Image.open(mask_path) as pimg:
                arr = np.array(pimg.convert("L"))
        except Exception as err:
            logger.debug("Failed to load mask '%s': %s", mask_path, err)
            return None

    # Binary thresholding: change pixels > 0
    bin_mask = (arr > 0).astype(np.uint8)
    if target_shape and bin_mask.shape != target_shape:
        bin_mask = cv2.resize(bin_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return bin_mask


def check_directional_accuracy(gt_answer: str, pred_answer: str) -> Optional[bool]:
    """Evaluates whether directional change classification ([INCREASED] / [DECREASED]) matches."""
    verdicts = ["[INCREASED]", "[DECREASED]", "[REMAINED UNCHANGED]"]
    gt_upper = gt_answer.upper()
    pred_upper = pred_answer.upper()

    gt_verdict = next((v for v in verdicts if v in gt_upper), None)
    if not gt_verdict:
        # Check loose keywords if formal bracketed tag is absent
        if "INCREAS" in gt_upper:
            gt_verdict = "[INCREASED]"
        elif "DECREAS" in gt_upper:
            gt_verdict = "[DECREASED]"
        elif "UNCHANGED" in gt_upper or "NO CHANGE" in gt_upper:
            gt_verdict = "[REMAINED UNCHANGED]"

    if not gt_verdict:
        return None

    return gt_verdict in pred_upper


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    split_file = data_dir / "splits" / "test_benchmark.json"

    # Graceful exit if benchmark data does not exist
    if not data_dir.exists() or not split_file.exists():
        print(f"Warning: CDVQA benchmark split not found at: {split_file}")
        print("Staging placeholder directory: please stage CDVQA test samples to run evaluation.")
        sys.exit(0)

    try:
        with open(split_file, "r", encoding="utf-8") as f:
            samples: List[Dict[str, Any]] = json.load(f)
    except Exception as err:
        print(f"Warning: Failed to parse CDVQA JSON split: {err}")
        sys.exit(0)

    if not isinstance(samples, list) or not samples:
        print(f"Warning: No valid samples found in {split_file}")
        sys.exit(0)

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    logger.info("Starting CDVQA evaluation on %d bi-temporal test samples...", len(samples))

    controller = SatQueryController()

    bleu4_scores: List[float] = []
    rouge_l_f1s: List[float] = []
    miou_scores: List[float] = []
    directional_checks: List[bool] = []
    eval_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(samples, 1):
        t1_ref = item.get("image_t1", "")
        t2_ref = item.get("image_t2", "")
        question = item.get("question", "What changed between these two observation dates?")
        gt_answer = str(item.get("answer", ""))
        gt_mask_ref = item.get("mask", "")

        t1_path = resolve_image_path(data_dir, t1_ref, subfolder="t1")
        t2_path = resolve_image_path(data_dir, t2_ref, subfolder="t2")

        if not t1_path.exists() or not t2_path.exists():
            logger.warning("Sample %d: T1 (%s) or T2 (%s) missing on disk. Skipping.", idx, t1_path, t2_path)
            continue

        try:
            trace = controller.execute_workflow(
                query=question,
                filepaths=[str(t1_path), str(t2_path)],
                force_task="bitemporal_change",
            )
            pred_answer = trace.output

            # 1. Text Metrics
            b4 = calculate_bleu4(gt_answer, pred_answer)
            rl = calculate_rouge_l(gt_answer, pred_answer)
            bleu4_scores.append(b4)
            rouge_l_f1s.append(rl["fmeasure"])

            # 2. Directional Accuracy
            dir_correct = check_directional_accuracy(gt_answer, pred_answer)
            if dir_correct is not None:
                directional_checks.append(dir_correct)

            # 3. Spatial Change Mask mIoU
            miou_val = None
            pred_mask = getattr(controller, "last_change_mask", None)
            if pred_mask is None and trace.scratchpad:
                pred_mask = trace.scratchpad.get("change_mask")

            if pred_mask is not None and gt_mask_ref:
                gt_mask_path = resolve_image_path(data_dir, gt_mask_ref, subfolder="masks")
                gt_mask = load_mask(gt_mask_path, target_shape=pred_mask.shape)
                if gt_mask is not None:
                    miou_val = calculate_miou(pred_mask, gt_mask, num_classes=2)
                    miou_scores.append(miou_val)

            eval_records.append({
                "sample_id": idx,
                "t1": str(t1_path.name),
                "t2": str(t2_path.name),
                "question": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "directional_correct": dir_correct,
                "bleu4": round(b4, 4),
                "rouge_l_f1": round(rl["fmeasure"], 4),
                "change_miou": round(miou_val, 4) if miou_val is not None else None,
            })

            logger.info("Sample %d/%d evaluated: BLEU-4=%.4f, mIoU=%s", idx, len(samples), b4, miou_val)

        except Exception as exc:
            logger.error("Sample %d failed during bitemporal change evaluation: %s", idx, exc)

    total_evaluated = len(eval_records)
    if total_evaluated == 0:
        print("Warning: No samples were successfully evaluated.")
        sys.exit(0)

    summary = {
        "benchmark": "CDVQA",
        "total_samples": len(samples),
        "evaluated_samples": total_evaluated,
        "metrics": {
            "mean_bleu4": round(float(np.mean(bleu4_scores)), 4) if bleu4_scores else 0.0,
            "mean_rouge_l_f1": round(float(np.mean(rouge_l_f1s)), 4) if rouge_l_f1s else 0.0,
            "mean_change_miou": round(float(np.mean(miou_scores)), 4) if miou_scores else None,
            "directional_accuracy": round(float(np.mean(directional_checks)), 4) if directional_checks else None,
        },
        "records": eval_records,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CDVQA CHANGE DETECTION BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated:        {total_evaluated}/{len(samples)}")
    print(f"BLEU-4:                 {summary['metrics']['mean_bleu4']:.4f}")
    print(f"ROUGE-L (F1):           {summary['metrics']['mean_rouge_l_f1']:.4f}")
    if summary["metrics"]["mean_change_miou"] is not None:
        print(f"Change Mask mIoU:       {summary['metrics']['mean_change_miou']:.4f}")
    if summary["metrics"]["directional_accuracy"] is not None:
        print(f"Directional Accuracy:   {summary['metrics']['directional_accuracy'] * 100:.2f}%")
    print(f"Report saved to: {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
