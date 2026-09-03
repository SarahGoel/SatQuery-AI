"""Phase 4 Step 9 — multi-temporal attention Change-VQA."""

from __future__ import annotations

import torch

from app.services.models.change_vqa import (
    CHANGE_VOCAB,
    ChangeVQATextDecoder,
    TemporalDifferenceAttention,
)


def test_temporal_difference_attention_shape() -> None:
    tda = TemporalDifferenceAttention(channels=64)
    t1 = torch.randn(1, 64, 128, 128)
    t2 = torch.randn(1, 64, 128, 128)
    gated_diff = tda(t1, t2)
    assert gated_diff.shape == (1, 64, 128, 128)


def test_change_vqa_text_decoder_logits() -> None:
    decoder = ChangeVQATextDecoder(vocab_size=len(CHANGE_VOCAB), embed_dim=64)
    features = torch.randn(2, 64, 32, 32)
    logits = decoder(features)
    assert logits.shape == (2, len(CHANGE_VOCAB))
