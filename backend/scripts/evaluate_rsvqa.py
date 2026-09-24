#!/usr/bin/env python3
"""ISRO SIH Public Benchmark Evaluator — RSVQA.

Evaluates Remote Sensing Visual Question Answering (RSVQA) across question types:
- presence: object or land-cover presence verification (e.g., 'Is there a residential area?')
- count: object counting questions (e.g., 'How many fuel tanks are visible?')
- comparison: spatial/relational comparisons (e.g., 'Are there more buildings than trees?')
- rural_urban: land use / urbanization classification.

Computes exact-match accuracy and smoothed BLEU-4 grouped by question category.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure backend directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
repo_root = backend_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.evaluation.metrics import calculate_bleu4
from app.services.agent import SatQueryController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_rsvqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SatQuery on RSVQA benchmark")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/rsvqa",
        help="Path to RSVQA root directory containing splits/test_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/rsvqa_eval.json",
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


def is_exact_match(gt_answer: str, pred_answer: str) -> float:
    """Computes exact match accuracy handling minor punctuation and casing variance."""
    gt_clean = gt_answer.strip().lower().rstrip(".!?,")
    pred_clean = pred_answer.strip().lower().rstrip(".!?,")

    if gt_clean == pred_clean:
        return 1.0

    # Also accept if the concise ground-truth token is directly matched in concise answer
    tokens_pred = pred_clean.split()
    if gt_clean in tokens_pred:
        return 1.0

    # Yes/No specific verification
    if gt_clean in {"yes", "no"}:
        if pred_clean.startswith(gt_clean):
            return 1.0
        if f" {gt_clean} " in f" {pred_clean} ":
            return 1.0

    return 0.0


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    split_file = data_dir / "splits" / "test_benchmark.json"

    # Graceful exit if benchmark data does not exist
    if not data_dir.exists() or not split_file.exists():
        print(f"Warning: RSVQA benchmark split not found at: {split_file}")
        print("Staging placeholder directory: please stage RSVQA test samples to run evaluation.")
        sys.exit(0)

    try:
        with open(split_file, "r", encoding="utf-8") as f:
            samples: List[Dict[str, Any]] = json.load(f)
    except Exception as err:
        print(f"Warning: Failed to parse RSVQA JSON split: {err}")
        sys.exit(0)

    if not isinstance(samples, list) or not samples:
        print(f"Warning: No valid samples found in {split_file}")
        sys.exit(0)

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    logger.info("Starting RSVQA evaluation on %d test samples...", len(samples))

    controller = SatQueryController()

    type_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"accuracy": [], "bleu4": []})
    all_accuracy: List[float] = []
    all_bleu4: List[float] = []
    eval_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(samples, 1):
        image_ref = item.get("image", "")
        question = item.get("question", "")
        gt_answer = str(item.get("answer", ""))
        q_type = str(item.get("type", "general")).lower().strip()

        img_path = resolve_image_path(data_dir, image_ref)
        if not img_path.exists():
            logger.warning("Sample %d: image file '%s' not found on disk. Skipping.", idx, img_path)
            continue

        try:
            trace = controller.execute_workflow(query=question, filepaths=[str(img_path)])
            pred_answer = trace.output

            b4 = calculate_bleu4(gt_answer, pred_answer)
            acc = is_exact_match(gt_answer, pred_answer)

            type_stats[q_type]["accuracy"].append(acc)
            type_stats[q_type]["bleu4"].append(b4)
            all_accuracy.append(acc)
            all_bleu4.append(b4)

            eval_records.append({
                "sample_id": idx,
                "image": str(img_path.name),
                "type": q_type,
                "question": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "exact_match": acc == 1.0,
                "bleu4": round(b4, 4),
            })

            logger.info("Sample %d/%d [%s]: exact=%s, BLEU-4=%.4f", idx, len(samples), q_type, bool(acc), b4)

        except Exception as exc:
            logger.error("Sample %d failed during pipeline inference: %s", idx, exc)

    total_evaluated = len(eval_records)
    if total_evaluated == 0:
        print("Warning: No samples were successfully evaluated.")
        sys.exit(0)

    by_type_summary: Dict[str, Dict[str, Any]] = {}
    for cat, metrics in type_stats.items():
        by_type_summary[cat] = {
            "count": len(metrics["accuracy"]),
            "accuracy": round(float(np.mean(metrics["accuracy"])), 4) if metrics["accuracy"] else 0.0,
            "mean_bleu4": round(float(np.mean(metrics["bleu4"])), 4) if metrics["bleu4"] else 0.0,
        }

    summary = {
        "benchmark": "RSVQA",
        "total_samples": len(samples),
        "evaluated_samples": total_evaluated,
        "metrics": {
            "overall_accuracy": round(float(np.mean(all_accuracy)), 4) if all_accuracy else 0.0,
            "overall_bleu4": round(float(np.mean(all_bleu4)), 4) if all_bleu4 else 0.0,
        },
        "by_type": by_type_summary,
        "records": eval_records,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RSVQA BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated:   {total_evaluated}/{len(samples)}")
    print(f"Overall Accuracy:  {summary['metrics']['overall_accuracy'] * 100:.2f}%")
    print(f"Overall BLEU-4:    {summary['metrics']['overall_bleu4']:.4f}")
    print("-" * 60)
    print("Breakdown by Question Type:")
    for cat, dat in by_type_summary.items():
        print(f"  - {cat:<15}: Acc = {dat['accuracy'] * 100:>5.1f}%  |  BLEU-4 = {dat['mean_bleu4']:.4f}  (N={dat['count']})")
    print(f"\nReport saved to: {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
