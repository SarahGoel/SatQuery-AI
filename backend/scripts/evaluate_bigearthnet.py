#!/usr/bin/env python3
"""ISRO SIH Public Benchmark Evaluator — BigEarthNet.

Evaluates multi-sensor (Optical Sentinel-2 + SAR Sentinel-1) multi-label land-cover
classification against the standard 19-class CORINE Land Cover (CLC) nomenclature.
Computes quantitative evaluation metrics:
- Macro-F1: Unweighted mean of class-wise F1 scores across land-cover categories.
- Subset Accuracy: Proportion of test tiles where predicted label sets match ground truth exactly.
- Class-wise Precision, Recall, and F1 with support counts.
- Mean Jaccard / Hamming Similarity score across multi-label predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

# Ensure backend directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
repo_root = backend_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.services.models.cross_modal import CrossModalAnalysisTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_bigearthnet")

BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SatQuery on BigEarthNet benchmark")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/bigearthnet",
        help="Path to BigEarthNet root directory containing splits/test_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/bigearthnet_eval.json",
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


def predict_multilabel_from_fusion(result: Any, opt_path: Path) -> List[str]:
    """Infers multi-label remote sensing land cover categories from fusion result and spectral features."""
    pred_labels: Set[str] = set()
    params = getattr(result, "params", {}) or {}
    answer_text = str(getattr(result, "answer", "")).lower()

    # 1. Structural Optical Built-up Detection
    if params.get("builtup_features_count", 0) > 0:
        pred_labels.add("Urban fabric")

    # 2. Specular SAR Radar Water Detection
    if params.get("water_features_count", 0) > 0:
        pred_labels.add("Inland waters")

    # 3. Textual mentions in specialist output
    for cls_name in BIGEARTHNET_19_CLASSES:
        if cls_name.lower() in answer_text:
            pred_labels.add(cls_name)

    # 4. Multispectral vegetation index (NDVI) if optical GeoTIFF has >= 4 bands (Red: Band 1/3, NIR: Band 4)
    try:
        import rasterio
        with rasterio.open(opt_path) as src:
            if src.count >= 4:
                red = src.read(3).astype(np.float32)
                nir = src.read(4).astype(np.float32)
                ndvi = (nir - red) / (nir + red + 1e-6)
                mean_ndvi = float(np.nanmean(ndvi))
                if mean_ndvi > 0.45:
                    pred_labels.add("Broad-leaved forest")
                elif mean_ndvi > 0.25:
                    pred_labels.add("Arable land")
    except Exception:
        pass

    return sorted(list(pred_labels))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    split_file = data_dir / "splits" / "test_benchmark.json"

    # Graceful exit if benchmark data does not exist
    if not data_dir.exists() or not split_file.exists():
        print(f"Warning: BigEarthNet benchmark split not found at: {split_file}")
        print("Staging placeholder directory: please stage BigEarthNet test samples to run evaluation.")
        sys.exit(0)

    try:
        with open(split_file, "r", encoding="utf-8") as f:
            samples: List[Dict[str, Any]] = json.load(f)
    except Exception as err:
        print(f"Warning: Failed to parse BigEarthNet JSON split: {err}")
        sys.exit(0)

    if not isinstance(samples, list) or not samples:
        print(f"Warning: No valid samples found in {split_file}")
        sys.exit(0)

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    logger.info("Starting BigEarthNet multi-label evaluation on %d test samples...", len(samples))

    tool = CrossModalAnalysisTool()

    eval_records: List[Dict[str, Any]] = []
    subset_matches: List[float] = []
    jaccard_scores: List[float] = []

    # Confusion statistics per class: TP, FP, FN, Support
    class_stats: Dict[str, Dict[str, int]] = {
        c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in BIGEARTHNET_19_CLASSES
    }

    for idx, item in enumerate(samples, 1):
        opt_ref = item.get("image_opt", "")
        sar_ref = item.get("image_sar", "")
        gt_labels: List[str] = item.get("labels", [])

        opt_path = resolve_image_path(data_dir, opt_ref, subfolder="sentinel2")
        sar_path = resolve_image_path(data_dir, sar_ref, subfolder="sentinel1")

        if not opt_path.exists() or not sar_path.exists():
            logger.warning("Sample %d: Optical (%s) or SAR (%s) missing. Skipping.", idx, opt_path, sar_path)
            continue

        try:
            res = tool.analyze(
                optical_path=opt_path,
                sar_path=sar_path,
                query="Identify built-up urban infrastructure and water surfaces",
            )
            pred_labels = predict_multilabel_from_fusion(res, opt_path)

            set_gt = set(gt_labels)
            set_pred = set(pred_labels)

            # Subset exact match
            exact_subset = 1.0 if set_gt == set_pred else 0.0
            subset_matches.append(exact_subset)

            # Jaccard / Hamming similarity
            union_len = len(set_gt | set_pred)
            jaccard = (len(set_gt & set_pred) / union_len) if union_len > 0 else 1.0
            jaccard_scores.append(jaccard)

            # Track class stats
            for c in BIGEARTHNET_19_CLASSES:
                in_gt = c in set_gt
                in_pred = c in set_pred
                if in_gt:
                    class_stats[c]["support"] += 1
                if in_gt and in_pred:
                    class_stats[c]["tp"] += 1
                elif in_pred and not in_gt:
                    class_stats[c]["fp"] += 1
                elif in_gt and not in_pred:
                    class_stats[c]["fn"] += 1

            eval_records.append({
                "sample_id": idx,
                "image_opt": str(opt_path.name),
                "image_sar": str(sar_path.name),
                "gt_labels": gt_labels,
                "pred_labels": pred_labels,
                "subset_accuracy": exact_subset,
                "jaccard_similarity": round(jaccard, 4),
            })

            logger.info("Sample %d/%d: GT=%s | Pred=%s (Jaccard=%.2f)", idx, len(samples), gt_labels, pred_labels, jaccard)

        except Exception as exc:
            logger.error("Sample %d failed during BigEarthNet inference: %s", idx, exc)

    total_evaluated = len(eval_records)
    if total_evaluated == 0:
        print("Warning: No samples were successfully evaluated.")
        sys.exit(0)

    # Compute Macro-F1 across active classes (support > 0 or predicted)
    f1_list: List[float] = []
    class_results: Dict[str, Dict[str, Any]] = {}

    for c, stats in class_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        support = stats["support"]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if support > 0 or (tp + fp) > 0:
            f1_list.append(f1)

        class_results[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    mean_subset_acc = float(np.mean(subset_matches)) if subset_matches else 0.0
    mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

    summary = {
        "benchmark": "BigEarthNet",
        "total_samples": len(samples),
        "evaluated_samples": total_evaluated,
        "metrics": {
            "macro_f1": round(macro_f1, 4),
            "subset_accuracy": round(mean_subset_acc, 4),
            "mean_jaccard_similarity": round(mean_jaccard, 4),
        },
        "class_metrics": class_results,
        "records": eval_records,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("BIGEARTHNET MULTI-LABEL EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total Evaluated:         {total_evaluated}/{len(samples)}")
    print(f"Macro-F1:                {macro_f1:.4f}")
    print(f"Subset Accuracy:         {mean_subset_acc * 100:.2f}%")
    print(f"Mean Jaccard Similarity: {mean_jaccard:.4f}")
    print("-" * 65)
    print(f"{'Class Name':<35} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Supp'}")
    print("-" * 65)
    for c, r in class_results.items():
        if r["support"] > 0 or r["f1"] > 0:
            print(f"{c[:33]:<35} | {r['precision']:>5.2f}  | {r['recall']:>5.2f}  | {r['f1']:>5.2f}  | {r['support']}")
    print(f"\nReport saved to: {out_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
