#!/usr/bin/env python3
"""Bypass the UI: POST imagery + NL query to /api/v1/query and verify responses.

Starts no server. Expects the FastAPI gateway at http://localhost:8000.

    python scripts/test_e2e_inference.py
    python scripts/test_e2e_inference.py --suite
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

def _find_artifacts_dir() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = p / "artifacts"
        if candidate.is_dir():
            return candidate
        p = p.parent
    return Path.cwd() / "artifacts"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_dummy_png(path: Path, size: int = 32, pattern_seed: int = 42) -> None:
    """Tiny RGB PNG for API testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            r = (pattern_seed + (x * 7) + (y * 3)) % 256
            g = (pattern_seed * 2 + (x * 5) + (y * 11)) % 256
            b = (pattern_seed * 3 + (x * 13) + (y * 2)) % 256
            row.extend((r, g, b))
        rows.append(bytes(row))
    raw = b"".join(b"\x00" + row for row in rows)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def run_e2e_suite(base_url: str) -> int:
    import requests

    endpoint = base_url.rstrip("/") + "/api/v1/query"
    artifacts_dir = _find_artifacts_dir() / "e2e"
    img1 = artifacts_dir / "scene_t1.png"
    img2 = artifacts_dir / "scene_t2.png"
    write_dummy_png(img1, 32, pattern_seed=10)
    write_dummy_png(img2, 32, pattern_seed=50)

    print("==================================================")
    print("RUNNING SATQUERY AI E2E VERIFICATION SUITE")
    print("Endpoint:", endpoint)
    print("==================================================")

    # Test 1: Single-Image Grounding
    print("\n[E2E Test 1] Single-Image Grounding")
    with img1.open("rb") as h1:
        res = requests.post(
            endpoint,
            data={"query": "Locate circular storage tanks in this scene"},
            files={"files": (img1.name, h1, "image/png")},
            timeout=60,
        )
    assert res.status_code == 200, f"Grounding failed: {res.text}"
    p1 = res.json()
    assert p1.get("task_type") == "single_grounding"
    assert p1.get("confidence") is not None
    assert p1.get("geojson") is not None
    print("  -> Status 200 OK. task_type='single_grounding', confidence=", p1.get("confidence"))
    print("  -> Models executed:", p1.get("models_executed"))
    print("  -> GeoJSON features count:", len((p1.get("geojson") or {}).get("features", [])))

    # Test 2: Single-Image VQA
    print("\n[E2E Test 2] Single-Image VQA")
    with img1.open("rb") as h1:
        res = requests.post(
            endpoint,
            data={"query": "Describe scene characteristics and landcover distribution"},
            files={"files": (img1.name, h1, "image/png")},
            timeout=60,
        )
    assert res.status_code == 200, f"VQA failed: {res.text}"
    p2 = res.json()
    assert p2.get("task_type") == "single_vqa"
    print("  -> Status 200 OK. task_type='single_vqa', answer snippet:", p2.get("answer", "")[:80])

    # Test 3: Intent Mismatch Rejection (1 image + temporal query)
    print("\n[E2E Test 3] Rejection on 1-image temporal change intent")
    with img1.open("rb") as h1:
        res = requests.post(
            endpoint,
            data={"query": "What changed between these two dates?"},
            files={"files": (img1.name, h1, "image/png")},
            timeout=60,
        )
    assert res.status_code == 400, f"Expected HTTP 400 rejection, got {res.status_code}"
    print("  -> Correctly rejected with HTTP 400:", res.json().get("detail", ""))

    # Test 4: Bi-Temporal Change Detection (2 images)
    print("\n[E2E Test 4] Bi-Temporal Change Detection (2 images)")
    with img1.open("rb") as h1, img2.open("rb") as h2:
        res = requests.post(
            endpoint,
            data={"query": "What changed between these two dates?"},
            files=[
                ("files", (img1.name, h1, "image/png")),
                ("files", (img2.name, h2, "image/png")),
            ],
            timeout=60,
        )
    assert res.status_code == 200, f"Change detection failed: {res.text}"
    p4 = res.json()
    assert p4.get("task_type") == "bitemporal_change"
    assert p4.get("change_mask") is not None
    print("  -> Status 200 OK. task_type='bitemporal_change'")
    print("  -> Change overlay URI:", p4.get("change_overlay_uri"))

    # Test 5: Cross-Modal Optical + SAR Analysis (2 images)
    print("\n[E2E Test 5] Cross-Modal Optical + SAR Joint Analysis (2 images)")
    with img1.open("rb") as h1, img2.open("rb") as h2:
        res = requests.post(
            endpoint,
            data={"query": "Perform cross-modal optical and SAR joint analysis for flood"},
            files=[
                ("files", (img1.name, h1, "image/png")),
                ("files", (img2.name, h2, "image/png")),
            ],
            timeout=60,
        )
    assert res.status_code == 200, f"Cross-modal failed: {res.text}"
    p5 = res.json()
    assert p5.get("task_type") == "cross_modal"
    assert "cross_modal_analysis_tool" in p5.get("models_executed", [])
    print("  -> Status 200 OK. task_type='cross_modal'")
    print("  -> Models executed:", p5.get("models_executed"))
    print("  -> Answer snippet:", p5.get("answer", "")[:120])

    print("\n==================================================")
    print("ALL 5 E2E INTEGRATION SUITE CHECKS PASSED (100% OK)")
    print("==================================================")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SatQuery AI e2e inference probe")
    parser.add_argument("--url", default="http://localhost:8000", help="API origin")
    parser.add_argument(
        "--query",
        default="Highlight industrial rooftops in this scene",
        help="Natural-language EO query",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional existing image/GeoTIFF (otherwise a dummy PNG is generated)",
    )
    parser.add_argument("--suite", action="store_true", help="Run the full 5-test e2e suite")
    args = parser.parse_args(argv)

    try:
        import requests
    except ImportError:
        print("This script requires the requests package. Install with: pip install requests", file=sys.stderr)
        return 2

    if args.suite:
        return run_e2e_suite(args.url)

    image_path = args.image
    if image_path is None:
        image_path = _find_artifacts_dir() / "e2e" / "dummy_scene.png"
        write_dummy_png(image_path)
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    endpoint = args.url.rstrip("/") + "/api/v1/query"
    mime = "image/png" if image_path.suffix.lower() == ".png" else "application/octet-stream"
    print(f"POST {endpoint}")
    print(f"image={image_path} query={args.query!r}")

    with image_path.open("rb") as handle:
        response = requests.post(
            endpoint,
            data={"query": args.query},
            files={"files": (image_path.name, handle, mime)},
            timeout=180,
        )

    print(f"status={response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        print(response.text)
        return 1
    print(json.dumps(payload, indent=2, default=str))

    if response.status_code != 200:
        return 1
    missing = [key for key in ("answer", "task_type", "audit_summary", "bbox", "geojson", "change_mask") if key not in payload]
    if missing:
        print(f"response missing keys: {missing}", file=sys.stderr)
        return 1
    print("e2e_ok")
    return 0


def test_dummy_png_generation(tmp_path: Path) -> None:
    target = tmp_path / "test.png"
    write_dummy_png(target, size=16)
    assert target.exists()
    assert target.stat().st_size > 0


def test_e2e_suite_probe() -> None:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_server_up = sock.connect_ex(("127.0.0.1", 8000)) == 0
    sock.close()
    if is_server_up:
        assert run_e2e_suite("http://127.0.0.1:8000") == 0


if __name__ == "__main__":
    raise SystemExit(main())

