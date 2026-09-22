"""CD-VQA Siamese change-detection training loop."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

try:
    from train_cdvqa import (
        CHANGE_VOCAB,
        SiameseChangeNet,
        resolve_device,
        run_cdvqa_training,
    )
except ImportError:
    from training.train_cdvqa import (
        CHANGE_VOCAB,
        SiameseChangeNet,
        resolve_device,
        run_cdvqa_training,
    )


def test_siamese_shared_encoder_and_heads() -> None:
    model = SiameseChangeNet(channels=16)
    t1 = torch.randn(2, 3, 32, 32)
    t2 = torch.randn(2, 3, 32, 32)
    mask_logits, token_logits = model(t1, t2)
    assert mask_logits.shape == (2, 1, 32, 32)
    assert token_logits.shape == (2, len(CHANGE_VOCAB))
    encoder_names = [name for name, _ in model.named_modules() if name == "encoder"]
    assert encoder_names == ["encoder"]


def test_resolve_device_cpu_or_cuda() -> None:
    device = resolve_device()
    assert device.type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_test_mode_saves_cdvqa_weights(tmp_path: Path) -> None:
    data = tmp_path / "data" / "raw" / "cdvqa"
    out = tmp_path / "local_models" / "cdvqa"
    run_cdvqa_training(data_dir=data, output_dir=out, test_mode=True, batch_size=1)
    assert (out / "checkpoint.pt").is_file()
    assert (out / "temporal_attn.pt").is_file()
    payload = torch.load(out / "checkpoint.pt", map_location="cpu")
    assert "encoder" in payload
    assert "tda" in payload
    assert "text_decoder" in payload
