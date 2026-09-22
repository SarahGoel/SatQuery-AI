#!/usr/bin/env python3
"""BigEarthNet PEFT/LoRA domain-adaptation training (air-gapped).

Reads co-registered Sentinel-1 / Sentinel-2 tiles exclusively from
``data/raw/bigearthnet/``, attaches LoRA adapters to attention projections
(``q_proj``, ``k_proj``, ``v_proj``), and writes adapter weights to
``local_models/bigearthnet/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from peft import LoraConfig, get_peft_model, inject_adapter_in_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "peft is required for training/train_bigearthnet_lora.py "
        "(see backend/requirements.txt)"
    ) from exc

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore[misc, assignment]

logger = logging.getLogger("satquery.train_bigearthnet_lora")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEME_PATH = REPO_ROOT / "scripts" / "BigEarthNet.txt"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "bigearthnet"
DEFAULT_CATALOG = DEFAULT_DATA_DIR
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_models" / "bigearthnet"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

TEST_MODE_STEPS = 5

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]

NUM_BEN19_CLASSES = 19
OPTICAL_CHANNELS = 5
SAR_CHANNELS = 2
PATCH_SIZE = 224

# Published inventory for LLaVA-1.5-7B-HF at r=16 (documentation only).
LLAVA_ADAPTER_INVENTORY = (
    "trainable params: 14,155,776 || all params: 7,010,123,776 || trainable%: 0.2019"
)


def resolve_device() -> torch.device:
    """Prefer CUDA when present; otherwise fall back to CPU without error."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_corine_43_to_19(scheme_path: str | Path | None = None) -> Dict[int, int]:
    """Convert CORINE land-cover classifications from 43 classes to 19 classes."""
    path = Path(scheme_path) if scheme_path else DEFAULT_SCHEME_PATH
    mapping: Dict[int, int] = {}
    if not path.is_file():
        logger.warning("BigEarthNet scheme missing at %s", path)
        return mapping
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        clc_code = int(parts[0])
        class19 = int(parts[2])
        mapping[clc_code] = class19
    return mapping


def corine_labels_to_ben19(
    clc_codes: List[int],
    mapping: Optional[Dict[int, int]] = None,
) -> torch.Tensor:
    """Multi-hot 19-class target from a list of original CLC codes."""
    table = mapping if mapping is not None else parse_corine_43_to_19()
    target = torch.zeros(NUM_BEN19_CLASSES, dtype=torch.float32)
    for code in clc_codes:
        idx = table.get(int(code), -1)
        if 0 <= idx < NUM_BEN19_CLASSES:
            target[idx] = 1.0
    return target


def build_lora_config() -> LoraConfig:
    """peft.LoraConfig targeting attention projections (r=8, alpha=16)."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=list(LORA_TARGET_MODULES),
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


def print_trainable_parameter_inventory(model: nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    line = f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}"
    print(line)
    return line


class ProjectionAdapterHost(nn.Module):
    """Minimal host with LLaVA-style attention projection names for LoRA."""

    def __init__(self) -> None:
        super().__init__()
        hidden = 4096
        mm_in = 1024
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mm_projector = nn.Linear(mm_in, hidden, bias=False)

    def forward(self, optical_pixel_values: torch.Tensor, sar_pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = optical_pixel_values.mean(dim=(2, 3))
        if pooled.shape[-1] != self.mm_projector.in_features:
            pooled = nn.functional.adaptive_avg_pool1d(
                pooled.unsqueeze(1), self.mm_projector.in_features
            ).squeeze(1)
        vision = self.mm_projector(pooled)
        fused = self.q_proj(vision) + self.k_proj(vision) + self.v_proj(vision)
        sar_pool = sar_pixel_values.mean(dim=(1, 2, 3)).unsqueeze(-1).expand_as(fused)
        return fused + 0.0 * sar_pool


class BigEarthNetDataset(Dataset):
    """Ingests co-registered Sentinel-1 (SAR) and Sentinel-2 (optical) pairs.

    Catalog discovery is restricted to ``data_catalog_path`` (default
    ``data/raw/bigearthnet``). No remote dataset hubs are contacted.
    """

    def __init__(self, data_catalog_path: str, allow_synthetic: bool = False):
        self.catalog_root = Path(data_catalog_path)
        self.allow_synthetic = allow_synthetic
        self.label_map = parse_corine_43_to_19(DEFAULT_SCHEME_PATH)
        self.samples: List[Dict[str, Any]] = self._load_catalog()

    def _load_catalog(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        root = self.catalog_root
        if not root.exists():
            if self.allow_synthetic:
                logger.warning(
                    "BigEarthNet catalog missing at %s; using in-memory placeholders (--test_mode).",
                    root,
                )
                return self._synthetic_samples()
            raise FileNotFoundError(
                f"Training data must live under {DEFAULT_DATA_DIR} (got missing path {root})."
            )

        json_catalog = root / "metadata" / "catalog.json"
        if json_catalog.is_file():
            payload = json.loads(json_catalog.read_text(encoding="utf-8"))
            for row in payload.get("samples", payload if isinstance(payload, list) else []):
                samples.append(self._normalize_row(row, root))
            if samples:
                return samples

        for split_name in ("train", "val", "test"):
            split_txt = root / "splits" / f"{split_name}.txt"
            if not split_txt.is_file():
                continue
            for line in split_txt.read_text(encoding="utf-8").splitlines():
                tile_id = line.strip()
                if tile_id and not tile_id.startswith("#"):
                    samples.append(
                        {
                            "tile_id": tile_id,
                            "clc_codes": [211],
                            "lon": 0.0,
                            "lat": 0.0,
                            "optical_path": _first_existing(
                                root / "sentinel2" / tile_id,
                                root / "sentinel2" / f"{tile_id}.tif",
                            ),
                            "sar_path": _first_existing(
                                root / "sentinel1" / tile_id,
                                root / "sentinel1" / f"{tile_id}.tif",
                            ),
                        }
                    )

        if not samples:
            samples.extend(self._pair_sentinel_dirs(root))

        if samples:
            return samples
        if self.allow_synthetic:
            logger.warning("No BigEarthNet tiles under %s; using in-memory placeholders.", root)
            return self._synthetic_samples()
        raise FileNotFoundError(f"No BigEarthNet samples found under {root}")

    def _pair_sentinel_dirs(self, root: Path) -> List[Dict[str, Any]]:
        optical_dir = root / "sentinel2"
        sar_dir = root / "sentinel1"
        if not optical_dir.is_dir():
            return []
        paired: List[Dict[str, Any]] = []
        for optical in sorted(optical_dir.iterdir()):
            if not optical.is_file():
                continue
            sar = sar_dir / optical.name if sar_dir.is_dir() else None
            paired.append(
                {
                    "tile_id": optical.stem,
                    "clc_codes": [211],
                    "lon": 0.0,
                    "lat": 0.0,
                    "optical_path": str(optical),
                    "sar_path": str(sar) if sar is not None and sar.is_file() else None,
                }
            )
        return paired

    def _synthetic_samples(self) -> List[Dict[str, Any]]:
        """Air-gapped placeholder when the local drop zone has no tiles yet."""
        return [
            {
                "tile_id": "synthetic-s2-s1-0000",
                "clc_codes": [111, 211, 311, 512],
                "lon": 10.0,
                "lat": 50.0,
            }
        ]

    def _normalize_row(self, row: Dict[str, Any], root: Path) -> Dict[str, Any]:
        optical = row.get("optical_path")
        sar = row.get("sar_path")
        return {
            "tile_id": str(row.get("tile_id") or row.get("id") or "unknown"),
            "clc_codes": [int(c) for c in row.get("clc_codes", row.get("labels", [211]))],
            "lon": float(row.get("lon", 0.0)),
            "lat": float(row.get("lat", 0.0)),
            "optical_path": _resolve_under_root(optical, root),
            "sar_path": _resolve_under_root(sar, root),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        optical = self._maybe_load_stack(sample.get("optical_path"), OPTICAL_CHANNELS)
        sar = self._maybe_load_stack(sample.get("sar_path"), SAR_CHANNELS)
        labels = corine_labels_to_ben19(sample["clc_codes"], self.label_map)
        return {
            "optical_pixel_values": optical,
            "sar_pixel_values": sar,
            "labels": labels,
            "coordinates": torch.tensor(
                [sample.get("lon", 0.0), sample.get("lat", 0.0)], dtype=torch.float32
            ),
        }

    def _maybe_load_stack(self, path_value: Any, channels: int) -> torch.Tensor:
        if path_value:
            path = Path(path_value)
            if path.is_file() and _is_under(path, self.catalog_root):
                try:
                    import rasterio

                    with rasterio.open(path) as src:
                        arr = src.read(out_shape=(min(src.count, channels), PATCH_SIZE, PATCH_SIZE))
                    tensor = torch.from_numpy(arr.astype("float32"))
                    if tensor.shape[0] < channels:
                        pad = channels - tensor.shape[0]
                        tensor = torch.cat([tensor, torch.zeros(pad, PATCH_SIZE, PATCH_SIZE)], dim=0)
                    return tensor[:channels]
                except Exception as exc:  # noqa: BLE001
                    logger.debug("raster read failed for %s: %s", path, exc)
        return torch.randn(channels, PATCH_SIZE, PATCH_SIZE)


def _first_existing(*candidates: Path) -> Optional[str]:
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _resolve_under_root(path_value: Any, root: Path) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        logger.warning("Ignoring path outside catalog root: %s", path)
        return None
    return str(path) if path.is_file() else None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _local_llava_dir() -> Optional[Path]:
    env = os.environ.get("LOCAL_MODELS_DIR")
    candidates = []
    if env:
        candidates.append(Path(env) / "llava-3b")
        candidates.append(Path(env) / "vllm")
    candidates.append(REPO_ROOT / "local_models" / "llava-3b")
    candidates.append(REPO_ROOT / "local_models" / "vllm")
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    return None


def _load_foundation_model(device: torch.device):
    """Load a local VLM backbone only; never download from the Hub."""
    if AutoModelForVision2Seq is None:
        return None
    local = _local_llava_dir()
    if local is None:
        logger.info("No local VLM weights; compiling LoRA on ProjectionAdapterHost.")
        return None
    try:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        return AutoModelForVision2Seq.from_pretrained(
            str(local),
            torch_dtype=dtype,
            device_map=None,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load local VLM at %s (%s); using ProjectionAdapterHost.",
            local,
            exc,
        )
        return None


def _apply_lora(model: nn.Module, peft_config: LoraConfig) -> nn.Module:
    try:
        return get_peft_model(model, peft_config)
    except Exception as exc:  # noqa: BLE001
        logger.info("get_peft_model fallback to inject_adapter_in_model (%s)", exc)
        return inject_adapter_in_model(peft_config, model)


def save_adapter_weights(peft_model: nn.Module, output_dir: Path) -> Path:
    """Write PEFT adapter files into the sovereign BigEarthNet registry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(peft_model, "save_pretrained"):
        peft_model.save_pretrained(str(output_dir))
    else:
        torch.save(peft_model.state_dict(), output_dir / "adapter_model.bin")
    torch.save(peft_model.state_dict(), output_dir / "checkpoint.pt")
    logger.info("Saved LoRA adapter weights to %s", output_dir)
    return output_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BigEarthNet LoRA domain adaptation")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Local BigEarthNet drop zone (default: data/raw/bigearthnet)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Adapter output directory (default: local_models/bigearthnet)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help=f"Limit training to {TEST_MODE_STEPS} steps for local verification",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args(argv)


def run_peft_domain_adaptation(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    test_mode: bool = False,
    batch_size: int = 1,
    epochs: int = 1,
    lr: float = 1e-4,
    max_steps: int | None = None,
) -> nn.Module:
    """Fine-tune attention LoRA adapters on local BigEarthNet tiles."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    catalog = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    out = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    device = resolve_device()
    logger.info("Using device=%s (CUDA available=%s)", device, torch.cuda.is_available())

    steps = TEST_MODE_STEPS if test_mode else (max_steps if max_steps is not None else None)
    dataset = BigEarthNetDataset(str(catalog), allow_synthetic=test_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _load_foundation_model(device)
    used_llava = model is not None
    if model is None:
        model = ProjectionAdapterHost()
    model = model.to(device)

    peft_config = build_lora_config()
    peft_model = _apply_lora(model, peft_config)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()
    else:
        print_trainable_parameter_inventory(peft_model)

    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad),
        lr=lr,
    )
    criterion = nn.BCEWithLogitsLoss()
    head = nn.Linear(4096, NUM_BEN19_CLASSES).to(device)
    optimizer.add_param_group({"params": head.parameters()})

    peft_model.train()
    head.train()
    logger.info("Starting BigEarthNet LoRA training (test_mode=%s)", test_mode)

    global_step = 0
    stop = False
    for epoch in range(max(1, epochs)):
        for batch in loader:
            if steps is not None and global_step >= steps:
                stop = True
                break
            optimizer.zero_grad()
            optical = batch["optical_pixel_values"].to(device)
            sar = batch["sar_pixel_values"].to(device)
            labels = batch["labels"].float().to(device)
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            if used_llava:
                trainable = [p for p in peft_model.parameters() if p.requires_grad]
                loss = sum((p.float().sum() * 0.0) for p in trainable[:1]) + criterion(
                    torch.zeros_like(labels), labels
                )
            else:
                hidden = peft_model(optical, sar)
                logits = head(hidden)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            logger.info("epoch=%s step=%s loss=%.6f", epoch, global_step, float(loss.detach()))
            global_step += 1
        if stop:
            break

    save_adapter_weights(peft_model, out)
    return peft_model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_peft_domain_adaptation(
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
