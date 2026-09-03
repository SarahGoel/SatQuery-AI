"""Phase 6 — BigEarthNet 19-class conversion and LoRA adapter compilation."""

from __future__ import annotations

from pathlib import Path

import torch

from train_lora import (
    LLAVA_ADAPTER_INVENTORY,
    LORA_TARGET_MODULES,
    NUM_BEN19_CLASSES,
    BigEarthNetDataset,
    ProjectionAdapterHost,
    build_lora_config,
    corine_labels_to_ben19,
    parse_corine_43_to_19,
    print_trainable_parameter_inventory,
)
from peft import inject_adapter_in_model


def test_corine_43_to_19_urban_and_arable() -> None:
    mapping = parse_corine_43_to_19()
    assert mapping[111] == 0
    assert mapping[112] == 0
    assert mapping[211] == 2
    assert mapping[122] == -1
    assert len({v for v in mapping.values() if v >= 0}) == NUM_BEN19_CLASSES


def test_multi_hot_19_class_labels() -> None:
    labels = corine_labels_to_ben19([111, 211, 311, 512, 122])
    assert labels.shape == (19,)
    assert labels.dtype == torch.float32
    assert float(labels[0]) == 1.0  # urban fabric
    assert float(labels[2]) == 1.0  # arable land
    assert float(labels[8]) == 1.0  # broad-leaved forest
    assert float(labels[17]) == 1.0  # inland waters
    assert float(labels.sum()) == 4.0  # dropped 122


def test_dataset_returns_s1_s2_and_19_class_labels(tmp_path: Path) -> None:
    ds = BigEarthNetDataset(str(tmp_path / "missing-catalog"))
    assert len(ds) >= 1
    item = ds[0]
    assert item["optical_pixel_values"].shape == (5, 224, 224)
    assert item["sar_pixel_values"].shape == (2, 224, 224)
    assert item["labels"].shape == (19,)
    assert item["coordinates"].shape == (2,)


def test_lora_config_targets_projection_matrices() -> None:
    cfg = build_lora_config()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert list(cfg.target_modules) == list(LORA_TARGET_MODULES)
    assert cfg.lora_dropout == 0.05
    assert cfg.bias == "none"
    assert cfg.task_type == "CAUSAL_LM"


def test_inject_lora_compiles_on_projection_host() -> None:
    host = ProjectionAdapterHost()
    adapted = inject_adapter_in_model(build_lora_config(), host)
    line = print_trainable_parameter_inventory(adapted)
    assert "trainable params:" in line
    trainable = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    assert trainable > 0
    q = adapted.q_proj
    assert any("lora" in n.lower() for n, _ in q.named_parameters())


def test_published_llava_adapter_inventory_string() -> None:
    assert LLAVA_ADAPTER_INVENTORY == (
        "trainable params: 14,155,776 || all params: 7,010,123,776 || trainable%: 0.2019"
    )
