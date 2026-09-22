#!/usr/bin/env python3
"""Compatibility entry point — canonical loop is training/train_bigearthnet_lora.py."""

from __future__ import annotations

from train_bigearthnet_lora import (  # noqa: F401
    DEFAULT_CATALOG,
    DEFAULT_SCHEME_PATH,
    LLAVA_ADAPTER_INVENTORY,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    NUM_BEN19_CLASSES,
    OPTICAL_CHANNELS,
    PATCH_SIZE,
    SAR_CHANNELS,
    BigEarthNetDataset,
    ProjectionAdapterHost,
    build_lora_config,
    corine_labels_to_ben19,
    parse_corine_43_to_19,
    print_trainable_parameter_inventory,
    resolve_device,
    run_peft_domain_adaptation,
)


if __name__ == "__main__":
    from train_bigearthnet_lora import main

    raise SystemExit(main())
