#!/usr/bin/env python3
"""ISRO SIH Public Benchmark Evaluator — VRSBench.

Evaluates single-image visual question answering, captioning, and visual grounding
against ground-truth annotations from the VRSBench remote-sensing benchmark.
Computes quantitative evaluation metrics:
- BLEU-4: Smoothed 4-gram bilingual evaluation understudy.
- ROUGE-L: Longest Common Subsequence precision, recall, and F1.
- CIDEr: Consensus-based Image Description Evaluation.
- mIoU: Mean Intersection over Union for visual grounding bounding boxes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure backend directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
repo_root = backend_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.evaluation.metrics import (
    calculate_bleu4,
    calculate_cider,
    calculate_miou,
    calculate_rouge_l,
)
from app.services.agent import SatQueryController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_vrsbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SatQuery on VRSBench benchmark")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/vrsbench",
        help="Path to VRSBench root directory containing splits/test_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/vrsbench_eval.json",
        help="Destination path for evaluation metric report JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of test samples to evaluate",
    )
    return parser.parse_args()


def resolve_image_path(data_dir: Path, image_name: str) -> Path:
    p = Path(image_name)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        data_dir / p,
        data_dir / "images" / p,
        data_dir / p.name,
        data_dir / "images" / p.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return data_dir / "images" / p.name


def bbox_to_mask(bbox: Optional[List[float]], shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Converts a bounding box [minx, miny, maxx, maxy] into a 2D binary segmentation mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if not bbox or len(bbox) < 4:
        return mask

    h, w = shape
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Normalize coordinates if expressed in [0..1] or [0..1000] space
    max_val = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_val <= 1.0:
        px1, px2 = int(x1 * w), int(x2 * w)
        py1, py2 = int(y1 * h), int(y2 * h)
    elif max_val <= 1000.0:
        px1, px2 = int((x1 / 1000.0) * w), int((x2 / 1000.0) * w)
        py1, py2 = int((y1 / 1000.0) * h), int((y2 / 1000.0) * h)
    else:
        px1, px2 = int(min(max(x1, 0), w)), int(min(max(x2, 0), w))
        py1, py2 = int(min(max(y1, 0), h)), int(min(max(y2, 0), h))

    xmin, xmax = min(px1, px2), max(px1, px2)
    ymin, ymax = min(py1, py2), max(py1, py2)

    mask[ymin:ymax, xmin:xmax] = 1
    return mask


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    split_file = data_dir / "splits" / "test_benchmark.json"

    # Graceful exit if benchmark data does not exist
    if not data_dir.exists() or not split_file.exists():
        print(f"Warning: VRSBench benchmark split not found at: {split_file}")
        print("Staging placeholder directory: please stage VRSBench test samples to run evaluation.")
        sys.exit(0)

    try:
        with open(split_file, "r", encoding="utf-8") as f:
            samples: List[Dict[str, Any]] = json.load(f)
    except Exception as err:
        print(f"Warning: Failed to parse VRSBench JSON split: {err}")
        sys.exit(0)

    if not isinstance(samples, list) or not samples:
        print(f"Warning: No valid samples found in {split_file}")
        sys.exit(0)

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    logger.info("Starting VRSBench evaluation on %d test samples...", len(samples))

    controller = SatQueryController()

    bleu4_scores: List[float] = []
    rouge_l_f1s: List[float] = []
    cider_scores: List[float] = []
    miou_scores: List[float] = []
    eval_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(samples, 1):
        image_ref = item.get("image", "")
        question = item.get("question", "Describe visible features in this image.")
        gt_answer = item.get("answer", "")
        gt_bbox = item.get("bbox")

        img_path = resolve_image_path(data_dir, image_ref)
        if not img_path.exists():
            logger.warning("Sample %d: image file '%s' not found on disk. Skipping.", idx, img_path)
            continue

        try:
            trace = controller.execute_workflow(query=question, filepaths=[str(img_path)])
            pred_answer = trace.output
            pred_bbox = controller.last_bbox

            # 1. Text Metrics
            b4 = calculate_bleu4(gt_answer, pred_answer)
            rl = calculate_rouge_l(gt_answer, pred_answer)
            cid = calculate_cider([gt_answer], pred_answer)

            bleu4_scores.append(b4)
            rouge_l_f1s.append(rl["fmeasure"])
            cider_scores.append(cid)

            # 2. Visual Grounding Bounding Box mIoU
            miou_val = None
            if gt_bbox is not None:
                gt_mask = bbox_to_mask(gt_bbox)
                pred_mask = bbox_to_mask(pred_bbox)
                miou_val = calculate_miou(pred_mask, gt_mask, num_classes=2)
                miou_scores.append(miou_val)

            eval_records.append({
                "sample_id": idx,
                "image": str(img_path.name),
                "question": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "gt_bbox": gt_bbox,
                "pred_bbox": pred_bbox,
                "bleu4": round(b4, 4),
                "rouge_l": {k: round(v, 4) for k, v in rl.items()},
                "cider": round(cid, 4),
                "bbox_miou": round(miou_val, 4) if miou_val is not None else None,
            })

            logger.info("Sample %d/%d evaluated: BLEU-4=%.4f, CIDEr=%.4f", idx, len(samples), b4, cid)

        except Exception as exc:
            logger.error("Sample %d failed during pipeline inference: %s", idx, exc)

    total_evaluated = len(eval_records)
    if total_evaluated == 0:
        print("Warning: No samples were successfully evaluated.")
        sys.exit(0)

    summary = {
        "benchmark": "VRSBench",
        "total_samples": len(samples),
        "evaluated_samples": total_evaluated,
        "metrics": {
            "mean_bleu4": round(float(np.mean(bleu4_scores)), 4) if bleu4_scores else 0.0,
            "mean_rouge_l_f1": round(float(np.mean(rouge_l_f1s)), 4) if rouge_l_f1s else 0.0,
            "mean_cider": round(float(np.mean(cider_scores)), 4) if cider_scores else 0.0,
            "mean_bbox_miou": round(float(np.mean(miou_scores)), 4) if miou_scores else None,
        },
        "records": eval_records,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("VRSBENCH BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated: {total_evaluated}/{len(samples)}")
    print(f"BLEU-4:          {summary['metrics']['mean_bleu4']:.4f}")
    print(f"ROUGE-L (F1):    {summary['metrics']['mean_rouge_l_f1']:.4f}")
    print(f"CIDEr:           {summary['metrics']['mean_cider']:.4f}")
    if summary["metrics"]["mean_bbox_miou"] is not None:
        print(f"BBox mIoU:       {summary['metrics']['mean_bbox_miou']:.4f}")
    print(f"Report saved to: {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
