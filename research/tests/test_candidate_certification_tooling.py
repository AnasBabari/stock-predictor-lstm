"""Candidate acquisition tools must never manufacture certification evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.candidate_ohlcv_acquisition_tool import acquisition_status
from scripts.candidate_pit64_construction_tool import package_candidate
from scripts.candidate_version_candidate_tool import create_candidate_manifest


def test_ohlcv_status_never_exposes_credentials_or_claims_rights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "secret-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "secret-value")
    report = acquisition_status()
    assert report["certification_eligible"] is False
    assert report["permission_status"] == "permission_pending"
    assert "secret-key" not in json.dumps(report)
    assert "secret-value" not in json.dumps(report)


def _pit64() -> dict[str, object]:
    return {
        "securities": [
            {
                "security_id": f"US.TEST{index:03d}",
                "cik": f"{index + 1:010d}",
                "figi": f"BBG00TEST{index:03d}",
                "exchange_mic": "XNAS",
                "membership_intervals": [
                    {
                        "start_date": "2020-01-01",
                        "end_date": "2025-01-01",
                        "source": "operator-supplied-source",
                        "source_digest": "a" * 64,
                    }
                ],
            }
            for index in range(64)
        ]
    }


def test_pit64_tool_packages_exact_operator_inputs(tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    sources = tmp_path / "sources.json"
    output = tmp_path / "review-request.json"
    universe.write_text(json.dumps(_pit64()), encoding="utf-8")
    sources.write_text(json.dumps({"operator-supplied-source": {"sha256": "a" * 64}}))
    request = package_candidate(universe, sources, output)
    assert request["certification_eligible"] is False
    assert request["external_reviewer_verified"] is False
    assert request == json.loads(output.read_text(encoding="utf-8"))


def test_pit64_tool_rejects_missing_interval_evidence(tmp_path: Path) -> None:
    payload = _pit64()
    payload["securities"][0]["membership_intervals"] = []  # type: ignore[index]
    universe = tmp_path / "universe.json"
    sources = tmp_path / "sources.json"
    universe.write_text(json.dumps(payload), encoding="utf-8")
    sources.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="membership interval"):
        package_candidate(universe, sources, tmp_path / "output.json")


def test_pit64_tool_rejects_unverifiable_or_duplicate_identity(tmp_path: Path) -> None:
    payload = _pit64()
    payload["securities"][1]["figi"] = payload["securities"][0]["figi"]  # type: ignore[index]
    universe = tmp_path / "universe.json"
    sources = tmp_path / "sources.json"
    universe.write_text(json.dumps(payload), encoding="utf-8")
    sources.write_text(
        json.dumps({"operator-supplied-source": {"sha256": "a" * 64}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicates figi"):
        package_candidate(universe, sources, tmp_path / "output.json")

    payload = _pit64()
    payload["securities"][0]["membership_intervals"][0]["source_digest"] = "bad"  # type: ignore[index]
    universe.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source digest"):
        package_candidate(universe, sources, tmp_path / "output.json")


def test_pit64_tool_never_overwrites_source_material(tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    sources = tmp_path / "sources.json"
    universe.write_text(json.dumps(_pit64()), encoding="utf-8")
    sources.write_text(
        json.dumps({"operator-supplied-source": {"sha256": "a" * 64}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        package_candidate(universe, sources, universe)


def test_version_tool_is_deterministic_and_has_no_side_effects(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    first = create_candidate_manifest(root, tmp_path / "first.json", "v12")
    second = create_candidate_manifest(root, tmp_path / "second.json", "v12")
    assert first == second
    assert first["certification_eligible"] is False
    with pytest.raises(ValueError, match="outside"):
        create_candidate_manifest(root, root / "manifest.json", "v12")


def test_version_tool_rejects_symlinks_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable for this Windows account")
    with pytest.raises(ValueError, match="symbolic links"):
        create_candidate_manifest(root, tmp_path / "manifest.json", "v12")
