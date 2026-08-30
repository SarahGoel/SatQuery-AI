"""Unit tests for air-gapped weights-registry probing (no GPU required)."""

from __future__ import annotations

from pathlib import Path

from backend.api.routes_health import WEIGHT_SUBDIRS, probe_air_gap


def test_air_gap_not_ready_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    result = probe_air_gap(missing)
    assert result["ready"] is False
    assert result["exists"] is False
    assert result["readable"] is False


def test_air_gap_ready_when_registries_present(tmp_path: Path) -> None:
    root = tmp_path / "local_models"
    for name in WEIGHT_SUBDIRS:
        (root / name).mkdir(parents=True)
    result = probe_air_gap(root)
    assert result["exists"] is True
    assert result["readable"] is True
    assert result["ready"] is True
    for name in WEIGHT_SUBDIRS:
        assert result["registries"][name]["exists"] is True
        assert result["registries"][name]["readable"] is True


def test_air_gap_degraded_if_one_registry_missing(tmp_path: Path) -> None:
    root = tmp_path / "local_models"
    for name in WEIGHT_SUBDIRS[:-1]:
        (root / name).mkdir(parents=True)
    result = probe_air_gap(root)
    assert result["ready"] is False
    assert result["registries"][WEIGHT_SUBDIRS[-1]]["exists"] is False
