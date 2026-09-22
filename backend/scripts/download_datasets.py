#!/usr/bin/env python3
"""Dataset scaffolding, optional download, and integrity checks for Phase 2.

Creates `data/raw/{bigearthnet,vrsbench,rsvqa,cdvqa}` with modality/split
placeholders. Does **not** pull multi-GB archives unless `--download` is passed
with a manifest of URLs (air-gapped hosts skip that flag).

Corrupted tiles (zero-byte, unreadable GeoTIFF, SHA-256 mismatch) are logged
and optionally moved to a quarantine folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve

logger = logging.getLogger("satquery.datasets")

RASTER_SUFFIXES = {".tif", ".tiff", ".gtiff", ".jp2", ".img", ".vrt"}
SPLIT_NAMES = ("train", "val", "test")

DATASET_LAYOUT: dict[str, tuple[str, ...]] = {
    "bigearthnet": (
        "sentinel1",
        "sentinel2",
        "splits",
        "metadata",
    ),
    "vrsbench": (
        "images",
        "annotations",
        "splits",
        "metadata",
    ),
    "rsvqa": (
        "images",
        "questions",
        "splits",
        "metadata",
    ),
    "cdvqa": (
        "t1",
        "t2",
        "qa",
        "splits",
        "metadata",
    ),
}


@dataclass
class IntegrityReport:
    dataset: str
    examined: int = 0
    ok: int = 0
    dropped: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "examined": self.examined,
            "ok": self.ok,
            "dropped": list(self.dropped),
            "reasons": dict(self.reasons),
        }


class DatasetDownloadManager:
    """Host-side scaffold and verifier for EO VQA / change-detection corpora."""

    def __init__(self, raw_root: str | Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.raw_root = Path(raw_root) if raw_root else repo_root / "data" / "raw"
        self.quarantine = self.raw_root / "_quarantine"

    def scaffold(self) -> dict[str, Path]:
        """Create dataset directories and empty split templates if missing."""
        created: dict[str, Path] = {}
        self.raw_root.mkdir(parents=True, exist_ok=True)
        for name, children in DATASET_LAYOUT.items():
            root = self.raw_root / name
            root.mkdir(parents=True, exist_ok=True)
            for child in children:
                (root / child).mkdir(parents=True, exist_ok=True)
            splits_dir = root / "splits"
            split_json = splits_dir / "splits.json"
            if not split_json.exists():
                split_json.write_text(
                    json.dumps({split: [] for split in SPLIT_NAMES}, indent=2) + "\n",
                    encoding="utf-8",
                )
            readme = root / "README.txt"
            if not readme.exists():
                readme.write_text(_dataset_readme(name), encoding="utf-8")
            created[name] = root
            logger.info("scaffolded %s -> %s", name, root)
        return created

    def parse_metadata_splits(self, dataset: str) -> dict[str, list[str]]:
        """Load train/val/test tile ids from JSON, CSV, or newline lists."""
        root = self.raw_root / dataset
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory missing: {root}")

        splits_dir = root / "splits"
        json_path = splits_dir / "splits.json"
        if json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            return _normalize_split_map(payload)

        csv_path = splits_dir / "splits.csv"
        if csv_path.is_file():
            return _parse_split_csv(csv_path)

        collected = {name: [] for name in SPLIT_NAMES}
        found_txt = False
        for name in SPLIT_NAMES:
            txt = splits_dir / f"{name}.txt"
            if txt.is_file():
                found_txt = True
                collected[name] = [
                    line.strip()
                    for line in txt.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")
                ]
        if found_txt:
            return collected

        logger.warning("No split metadata found under %s", splits_dir)
        return {name: [] for name in SPLIT_NAMES}

    def verify_and_drop_corrupted(
        self,
        dataset: str,
        *,
        manifest: dict[str, Any] | None = None,
        quarantine: bool = True,
    ) -> IntegrityReport:
        """Walk raster tiles, drop unreadable/empty/hash-mismatched files."""
        root = self.raw_root / dataset
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory missing: {root}")

        checksums = _checksum_index(manifest, dataset)
        report = IntegrityReport(dataset=dataset)
        for path in _iter_tiles(root):
            report.examined += 1
            reason = self._corruption_reason(path, checksums)
            if reason is None:
                report.ok += 1
                continue
            report.dropped.append(str(path))
            report.reasons[str(path)] = reason
            logger.warning("dropping corrupted tile %s (%s)", path, reason)
            if quarantine:
                self._quarantine_file(path)
            else:
                path.unlink(missing_ok=True)
        return report

    def download_from_manifest(self, manifest_path: str | Path) -> list[Path]:
        """Fetch files listed in a JSON manifest. No-op URLs are skipped."""
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        written: list[Path] = []
        datasets = payload.get("datasets", payload)
        for dataset, spec in datasets.items():
            dest_root = self.raw_root / dataset
            dest_root.mkdir(parents=True, exist_ok=True)
            for entry in spec.get("files", []):
                rel = entry.get("relpath") or entry.get("path")
                url = entry.get("url")
                if not rel or not url:
                    continue
                dest = dest_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info("download %s -> %s", url, dest)
                urlretrieve(url, dest)
                expected = entry.get("sha256")
                if expected and _sha256(dest) != expected.lower():
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(f"SHA-256 mismatch after download: {dest}")
                written.append(dest)
        return written

    def _corruption_reason(self, path: Path, checksums: dict[str, str]) -> str | None:
        if path.stat().st_size <= 0:
            return "empty_file"
        expected = checksums.get(_posix_rel(path, self.raw_root))
        if expected and _sha256(path) != expected.lower():
            return "sha256_mismatch"
        if path.suffix.lower() in RASTER_SUFFIXES:
            return _raster_unreadable_reason(path)
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return f"invalid_json:{exc}"
        return None

    def _quarantine_file(self, path: Path) -> None:
        rel = path.relative_to(self.raw_root)
        dest = self.quarantine / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))


def _dataset_readme(name: str) -> str:
    notes = {
        "bigearthnet": (
            "Place Sentinel-1 GRD (VV/VH) under sentinel1/ and Sentinel-2 L2A "
            "tiles under sentinel2/. Pairing keys go in splits/splits.json."
        ),
        "vrsbench": "Drop VRSBench images under images/ and captions/boxes under annotations/.",
        "rsvqa": "Drop RSVQA images under images/ and question JSON under questions/.",
        "cdvqa": "Drop T1/T2 change pairs under t1/ and t2/ with matching stems; QA in qa/.",
    }
    return (
        f"SatQuery AI — {name} drop zone\n"
        "Do not commit raw EO scenes. Prefetch on a connected host, then copy here.\n"
        f"{notes.get(name, '')}\n"
    )


def _iter_tiles(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {"README.txt", "splits.json"}:
            continue
        if "_quarantine" in path.parts:
            continue
        suffix = path.suffix.lower()
        if suffix in RASTER_SUFFIXES or suffix in {".json", ".csv"}:
            yield path


def _checksum_index(manifest: dict[str, Any] | None, dataset: str) -> dict[str, str]:
    if not manifest:
        return {}
    spec = manifest.get("datasets", manifest).get(dataset, {})
    index: dict[str, str] = {}
    for entry in spec.get("files", []):
        rel = entry.get("relpath") or entry.get("path")
        digest = entry.get("sha256")
        if rel and digest:
            index[f"{dataset}/{rel}".replace("\\", "/")] = digest.lower()
    return index


def _posix_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raster_unreadable_reason(path: Path) -> str | None:
    try:
        import rasterio
        from rasterio.errors import RasterioIOError
    except ImportError:
        return None
    try:
        with rasterio.open(path) as src:
            if src.width <= 0 or src.height <= 0 or src.count <= 0:
                return "invalid_raster_shape"
            _ = src.read(1, window=((0, min(1, src.height)), (0, min(1, src.width))))
    except (RasterioIOError, Exception) as extra:  # noqa: BLE001
        return f"unreadable_raster:{extra}"
    return None


def _normalize_split_map(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("splits.json must be an object mapping split -> tile ids.")
    result = {name: [] for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        values = payload.get(name, payload.get("valid" if name == "val" else name, []))
        if values is None:
            values = []
        result[name] = [str(item) for item in values]
    return result


def _parse_split_csv(path: Path) -> dict[str, list[str]]:
    result = {name: [] for name in SPLIT_NAMES}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return result
        fields = {name.lower(): name for name in reader.fieldnames}
        id_col = fields.get("tile_id") or fields.get("id") or fields.get("sample")
        split_col = fields.get("split") or fields.get("subset")
        if not id_col or not split_col:
            raise ValueError(f"{path} must have tile_id and split columns.")
        for row in reader:
            split = str(row[split_col]).strip().lower()
            if split == "valid":
                split = "val"
            if split not in result:
                continue
            result[split].append(str(row[id_col]).strip())
    return result


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[satquery.datasets] %(levelname)s %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Override data/raw location.",
    )
    parser.add_argument("--scaffold", action="store_true", help="Create dataset directories.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Scan tiles and quarantine corrupted files.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_LAYOUT),
        default=None,
        help="Limit verify/split parsing to one corpus.",
    )
    parser.add_argument("--manifest", default=None, help="JSON manifest for checksums/downloads.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch URLs listed in --manifest (requires outbound network).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    manager = DatasetDownloadManager(args.raw_root)

    if not args.scaffold and not args.verify and not args.download:
        args.scaffold = True

    if args.scaffold:
        manager.scaffold()

    manifest: dict[str, Any] | None = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    if args.download:
        if not args.manifest:
            logger.error("--download requires --manifest")
            return 2
        manager.download_from_manifest(args.manifest)

    datasets = [args.dataset] if args.dataset else list(DATASET_LAYOUT)
    if args.verify:
        for name in datasets:
            report = manager.verify_and_drop_corrupted(name, manifest=manifest)
            print(json.dumps(report.as_dict(), indent=2))
    else:
        for name in datasets:
            splits = manager.parse_metadata_splits(name)
            counts = {k: len(v) for k, v in splits.items()}
            logger.info("splits %s %s", name, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
