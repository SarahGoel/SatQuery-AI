"""Unit test suite for ISRO SIH Evaluation Metrics Engine.

Verifies mathematical correctness and boundary handling for:
1. BLEU-4 (Bilingual Evaluation Understudy with Smoothing Method 4)
2. ROUGE-L (Longest Common Subsequence precision, recall, and F1)
3. CIDEr (Consensus-based Image Description Evaluation across 1-4 grams)
4. mIoU (Mean Intersection over Union across binary and multi-class masks)
"""

from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import (
    calculate_bleu4,
    calculate_cider,
    calculate_miou,
    calculate_rouge_l,
)


def test_calculate_bleu4() -> None:
    """Verifies BLEU-4 computation with Chen & Cherry smoothing method 4."""
    reference = "Cartosat-2S optical imagery shows dense urban residential rooftops and asphalt roads."
    hypothesis_identical = "Cartosat-2S optical imagery shows dense urban residential rooftops and asphalt roads."
    hypothesis_partial = "Cartosat-2S satellite photo displays urban residential houses and road infrastructure."
    hypothesis_disjoint = "Completely unrelated text about marine biology in the deep ocean."

    # 1. Identical strings yield maximum score
    score_identical = calculate_bleu4(reference, hypothesis_identical)
    assert pytest.approx(1.0, abs=1e-3) == score_identical

    # 2. Similar remote-sensing captions yield valid positive score
    score_partial = calculate_bleu4(reference, hypothesis_partial)
    assert 0.0 < score_partial < 1.0
    assert isinstance(score_partial, float)

    # 3. Disjoint strings yield lower score
    score_disjoint = calculate_bleu4(reference, hypothesis_disjoint)
    assert score_disjoint < score_partial

    # 4. Boundary cases
    assert calculate_bleu4("", "") == 1.0
    assert calculate_bleu4(reference, "") == 0.0


def test_calculate_rouge_l() -> None:
    """Verifies ROUGE-L precision, recall, and F-measure extraction."""
    reference = "Sentinel-1 C-band SAR reveals severe river inundation and agricultural floodwaters."
    hypothesis = "Sentinel-1 SAR shows severe river inundation and floodwaters."

    res = calculate_rouge_l(reference, hypothesis)

    # Assert standard return contract
    assert isinstance(res, dict)
    assert "precision" in res
    assert "recall" in res
    assert "fmeasure" in res

    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["fmeasure"] <= 1.0

    # Shorter hypothesis with full token containment has high precision
    assert res["precision"] > 0.8
    assert res["fmeasure"] > 0.6

    # Identical string verification
    identical_res = calculate_rouge_l(reference, reference)
    assert pytest.approx(1.0, abs=1e-4) == identical_res["precision"]
    assert pytest.approx(1.0, abs=1e-4) == identical_res["recall"]
    assert pytest.approx(1.0, abs=1e-4) == identical_res["fmeasure"]


def test_calculate_cider() -> None:
    """Verifies CIDEr n-gram consensus score against multiple reference captions."""
    references = [
        "A high resolution optical satellite scene showing urban built-up area and road network.",
        "Optical satellite imagery displaying buildings and road infrastructure in a city.",
        "Aerial view of an urban residential district with roads and structures.",
    ]
    hypothesis_good = "A high resolution optical satellite scene with urban buildings and road network."
    hypothesis_poor = "Clouds over an empty ocean desert surface."

    # 1. Consensus against multiple reference captions
    score_good = calculate_cider(references, hypothesis_good)
    assert score_good > 1.5
    assert score_good <= 10.0

    # 2. Poor hypothesis produces substantially lower score
    score_poor = calculate_cider(references, hypothesis_poor)
    assert score_poor < score_good

    # 3. Identical reference reproduction approaches maximum scale (~10.0)
    score_identical = calculate_cider([references[0]], references[0])
    assert pytest.approx(10.0, rel=1e-2) == score_identical

    # 4. Boundary cases
    assert calculate_cider([], hypothesis_good) == 0.0
    assert calculate_cider(references, "") == 0.0


def test_calculate_miou() -> None:
    """Verifies Mean Intersection over Union across binary, float, and multi-class masks."""
    # 1. Identical binary mask yields 1.0
    pred_exact = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    true_exact = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert calculate_miou(pred_exact, true_exact, num_classes=2) == 1.0

    # 2. Controlled partial overlap
    # Class 0: pred {0, 1}, true {0, 3} -> inter=1, union=3 -> 1/3
    # Class 1: pred {2, 3}, true {1, 2} -> inter=1, union=3 -> 1/3
    # Mean: 1/3 ~ 0.3333
    pred_part = np.array([0, 0, 1, 1], dtype=np.uint8)
    true_part = np.array([0, 1, 1, 0], dtype=np.uint8)
    assert pytest.approx(1.0 / 3.0, abs=1e-4) == calculate_miou(pred_part, true_part, num_classes=2)

    # 3. Continuous float probability masks (thresholded at 0.5)
    pred_prob = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    true_bin = np.array([[1, 0], [0, 1]], dtype=np.int64)
    assert calculate_miou(pred_prob, true_bin, num_classes=2) == 1.0

    # 4. Multi-class semantic segmentation (num_classes = 3)
    pred_multi = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    true_multi = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    assert calculate_miou(pred_multi, true_multi, num_classes=3) == 1.0

    # 5. Dimension mismatch raises ValueError
    with pytest.raises(ValueError):
        calculate_miou(np.zeros((2, 2)), np.zeros((3, 3)))
