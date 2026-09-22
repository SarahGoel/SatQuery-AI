#!/usr/bin/env python3
"""CD-VQA Siamese bi-temporal change-detection training (air-gapped).

Reads T1/T2 image pairs exclusively from ``data/raw/cdvqa/`` (``t1/`` and
``t2/`` with matching stems) and writes fine-tuned weights to
``local_models/cdvqa/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("satquery.train_cdvqa")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "cdvqa"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_models" / "cdvqa"

TEST_MODE_STEPS = 5
PATCH_SIZE = 128
IN_CHANNELS = 3
FEATURE_CHANNELS = 64

CHANGE_VOCAB = [
    "<pad>",
    "no-change",
    "increase",
    "decrease",
    "urban-growth",
    "vegetation-loss",
    "water-expansion",
    "bare-soil",
    "seasonal-noise",
    "illumination-shift",
    "new-construction",
    "deforestation",
]

RASTER_SUFFIXES = {".tif", ".tiff", ".gtiff", ".jp2", ".png", ".jpg", ".jpeg", ".npy"}


def resolve_device() -> torch.device:
    """Prefer CUDA when present; otherwise fall back to CPU without error."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TemporalDifferenceAttention(nn.Module):
    """Gated absolute difference between shared-encoder T1/T2 features."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(feat_t1 - feat_t2)
        attn_weights = self.attn_layer(diff)
        return feat_t2 * attn_weights


class ChangeVQATextDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, change_features: torch.Tensor) -> torch.Tensor:
        pooled = self.global_pool(change_features).squeeze(-1).squeeze(-1)
        return self.fc(pooled)


class SiameseChangeNet(nn.Module):
    """Weight-tied Siamese encoder + temporal difference attention + VQA head.

    T1 and T2 share ``encoder`` parameters. Change is scored from the
    attention-gated difference map (mask + token logits).
    """

    def __init__(self, channels: int = FEATURE_CHANNELS, in_ch: int = IN_CHANNELS) -> None:
        super().__init__()
        # Weight-tied Siamese stem: the same Conv2d is applied to T1 and T2.
        self.encoder = nn.Conv2d(in_ch, channels, kernel_size=3, padding=1)
        self.tda = TemporalDifferenceAttention(channels=channels)
        self.text_decoder = ChangeVQATextDecoder(vocab_size=len(CHANGE_VOCAB), embed_dim=channels)
        self.mask_head = nn.Conv2d(channels, 1, kernel_size=1)

    def encode_pair(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        feat_t1 = self.encoder(t1)
        feat_t2 = self.encoder(t2)
        return self.tda(feat_t1, feat_t2)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gated = self.encode_pair(t1, t2)
        mask_logits = self.mask_head(gated)
        token_logits = self.text_decoder(gated)
        return mask_logits, token_logits


def _load_rgb(path: Path, size: int = PATCH_SIZE) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = np.transpose(arr[..., :3], (2, 0, 1))
        tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32)[:3])
    else:
        tensor = None
        try:
            import rasterio

            with rasterio.open(path) as src:
                count = min(IN_CHANNELS, src.count)
                arr = src.read(list(range(1, count + 1))).astype(np.float32)
            if arr.shape[0] < IN_CHANNELS:
                arr = np.repeat(arr[:1], IN_CHANNELS, axis=0)
            tensor = torch.from_numpy(arr[:IN_CHANNELS])
        except Exception:
            try:
                import cv2

                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
            except Exception:
                tensor = None
        if tensor is None:
            logger.warning("Could not decode %s; using random placeholder.", path)
            tensor = torch.rand(IN_CHANNELS, size, size)

    tensor = tensor.unsqueeze(0)
    tensor = nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


class CDVQAPairDataset(Dataset):
    """Pairs files in ``t1/`` and ``t2/`` that share the same stem."""

    def __init__(self, data_root: str | Path, allow_synthetic: bool = False, patch_size: int = PATCH_SIZE):
        self.root = Path(data_root)
        self.allow_synthetic = allow_synthetic
        self.patch_size = patch_size
        self.qa_labels = self._load_qa()
        self.pairs: List[Dict[str, Any]] = self._discover_pairs()

    def _load_qa(self) -> Dict[str, Any]:
        qa_dir = self.root / "qa"
        labels: Dict[str, Any] = {}
        if not qa_dir.is_dir():
            return labels
        for path in qa_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and "pairs" in payload:
                for row in payload["pairs"]:
                    stem = str(row.get("stem") or row.get("id") or "")
                    if stem:
                        labels[stem] = row
            elif isinstance(payload, list):
                for row in payload:
                    stem = str(row.get("stem") or row.get("id") or path.stem)
                    labels[stem] = row
            else:
                labels[path.stem] = payload
        return labels

    def _discover_pairs(self) -> List[Dict[str, Any]]:
        t1_dir = self.root / "t1"
        t2_dir = self.root / "t2"
        pairs: List[Dict[str, Any]] = []
        if t1_dir.is_dir() and t2_dir.is_dir():
            t2_by_stem = {p.stem: p for p in t2_dir.iterdir() if p.is_file() and p.suffix.lower() in RASTER_SUFFIXES}
            for t1 in sorted(t1_dir.iterdir()):
                if not t1.is_file() or t1.suffix.lower() not in RASTER_SUFFIXES:
                    continue
                t2 = t2_by_stem.get(t1.stem)
                if t2 is None:
                    continue
                pairs.append({"stem": t1.stem, "t1": t1, "t2": t2, "qa": self.qa_labels.get(t1.stem)})

        if pairs:
            return pairs
        if self.allow_synthetic:
            logger.warning(
                "No T1/T2 pairs under %s; using in-memory placeholders (--test_mode).",
                self.root,
            )
            return [{"stem": "synthetic-pair-0000", "t1": None, "t2": None, "qa": None}]
        raise FileNotFoundError(
            f"No matching T1/T2 pairs found under {self.root} (expected t1/ and t2/)."
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.pairs[idx]
        if item["t1"] is None or item["t2"] is None:
            t1 = torch.rand(IN_CHANNELS, self.patch_size, self.patch_size)
            t2 = t1 + 0.15 * torch.rand(IN_CHANNELS, self.patch_size, self.patch_size)
        else:
            t1 = _load_rgb(Path(item["t1"]), self.patch_size)
            t2 = _load_rgb(Path(item["t2"]), self.patch_size)

        change_frac = float((t1 - t2).abs().mean())
        changed = 1.0 if change_frac > 0.05 else 0.0
        qa = item.get("qa") or {}
        if isinstance(qa, dict):
            if "changed" in qa:
                changed = float(qa["changed"])
            token_name = str(qa.get("label") or qa.get("token") or "")
        else:
            token_name = ""
        token_idx = CHANGE_VOCAB.index(token_name) if token_name in CHANGE_VOCAB else (
            2 if changed else 1
        )
        return {
            "t1": t1,
            "t2": t2,
            "change_target": torch.tensor([changed], dtype=torch.float32),
            "token_target": torch.tensor(token_idx, dtype=torch.long),
        }


def save_cdvqa_weights(model: SiameseChangeNet, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": model.encoder.state_dict(),
        "tda": model.tda.state_dict(),
        "text_decoder": model.text_decoder.state_dict(),
        "mask_head": model.mask_head.state_dict(),
        "vocab": CHANGE_VOCAB,
    }
    ckpt = output_dir / "checkpoint.pt"
    torch.save(payload, ckpt)
    torch.save(model.tda.state_dict(), output_dir / "temporal_attn.pt")
    logger.info("Saved CD-VQA weights to %s", output_dir)
    return ckpt


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CD-VQA Siamese change-detection training")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Local CD-VQA drop zone (default: data/raw/cdvqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Checkpoint output directory (default: local_models/cdvqa)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help=f"Limit training to {TEST_MODE_STEPS} steps for local verification",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args(argv)


def run_cdvqa_training(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    test_mode: bool = False,
    batch_size: int = 2,
    epochs: int = 1,
    lr: float = 1e-4,
    max_steps: int | None = None,
) -> SiameseChangeNet:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    catalog = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    out = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    device = resolve_device()
    logger.info("Using device=%s (CUDA available=%s)", device, torch.cuda.is_available())

    if not catalog.exists() and not test_mode:
        raise FileNotFoundError(
            f"Training data must live under {DEFAULT_DATA_DIR} (got missing path {catalog})."
        )

    steps = TEST_MODE_STEPS if test_mode else max_steps
    dataset = CDVQAPairDataset(catalog, allow_synthetic=test_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SiameseChangeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    model.train()
    logger.info("Starting CD-VQA Siamese training (test_mode=%s, pairs=%s)", test_mode, len(dataset))

    global_step = 0
    stop = False
    for epoch in range(max(1, epochs)):
        for batch in loader:
            if steps is not None and global_step >= steps:
                stop = True
                break
            optimizer.zero_grad()
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            change_target = batch["change_target"].to(device)
            token_target = batch["token_target"].to(device)
            mask_logits, token_logits = model(t1, t2)
            pooled = mask_logits.mean(dim=(2, 3))
            loss = bce(pooled, change_target) + ce(token_logits, token_target)
            loss.backward()
            optimizer.step()
            logger.info("epoch=%s step=%s loss=%.6f", epoch, global_step, float(loss.detach()))
            global_step += 1
        if stop:
            break

    save_cdvqa_weights(model, out)
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_cdvqa_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_mode=args.test_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
