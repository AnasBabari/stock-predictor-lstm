"""Auditable-evidence tests for the experiment ledger and holdout recording."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stock_autoresearch.ledger import (
    LEDGER_SCHEMA_VERSION,
    MULTIPLICITY_POLICY,
    LedgerCorruptionError,
    append_record,
    compute_code_hash,
    export_tsv_summary,
    generate_markdown_report,
    read_records,
    resolve_commit,
)


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_append_record_generates_unique_ids_and_provenance(tmp_path: Path) -> None:
    ledger = tmp_path / "experiments.jsonl"
    first = append_record(ledger, {"candidate_family": "ridge", "decision": "discard"})
    second = append_record(ledger, {"candidate_family": "ridge", "decision": "keep"})

    assert first["experiment_id"] != second["experiment_id"]
    assert first["experiment_id"].startswith("exp_")
    assert len(first["experiment_id"]) > len("exp_") + 8  # timestamp + random suffix
    assert first["schema_version"] == LEDGER_SCHEMA_VERSION
    # Provenance is populated from the actual repository when available.
    assert isinstance(first["commit"], str) and first["commit"]
    assert first["code_hash"].startswith("sha256:") or first["code_hash"] == "unavailable"
    # Sanity: resolve_commit/compute_code_hash agree with the record defaults.
    assert resolve_commit() == first["commit"]
    assert compute_code_hash() == first["code_hash"]


def test_decision_fields_survive_serialization(tmp_path: Path) -> None:
    ledger = tmp_path / "experiments.jsonl"
    append_record(
        ledger,
        {
            "candidate_family": "random_features_ridge",
            "folds_beating_persistence": "4/5",
            "promotable": True,
            "decision": "keep",
            "decision_reason": "majority 4/5; pooled CI upper mae=0.91 rmse=0.93",
            "protocol_version": "multi-window-block-bootstrap-v1",
            "multiplicity_policy": MULTIPLICITY_POLICY,
            "evidence": {"per_window": [{"window_index": 0}]},
        },
    )
    stored = read_records(ledger)[0][0]
    assert stored["folds_beating_persistence"] == "4/5"
    assert stored["promotable"] is True
    assert stored["decision_reason"].startswith("majority 4/5")
    assert stored["protocol_version"] == "multi-window-block-bootstrap-v1"
    assert stored["multiplicity_policy"] == MULTIPLICITY_POLICY
    assert stored["evidence"]["per_window"][0]["window_index"] == 0


def test_corrupt_lines_fail_export_by_default(tmp_path: Path) -> None:
    ledger = _write_jsonl(
        tmp_path / "experiments.jsonl",
        [json.dumps({"candidate_family": "ridge", "decision": "keep"}), "{corrupt oops"],
    )
    tsv = tmp_path / "summary.tsv"
    report = tmp_path / "REPORT.md"

    with pytest.raises(LedgerCorruptionError) as excinfo:
        export_tsv_summary(ledger, tsv)
    assert any(line_no == 2 for line_no, _ in excinfo.value.corrupt)

    with pytest.raises(LedgerCorruptionError):
        generate_markdown_report(ledger, report)

    # Skipping requires an explicit opt-in.
    export_tsv_summary(ledger, tsv, skip_corrupt=True)
    generate_markdown_report(ledger, report, skip_corrupt=True)
    assert tsv.exists() and report.exists()


def test_report_is_deterministic_and_marks_unaudited_keeps(tmp_path: Path) -> None:
    legacy_keep = {
        "schema_version": 1,
        "experiment_id": "exp_legacy_0001",
        "candidate_family": "small_tcn",
        "commit": "unknown",
        "snapshot_id": "unknown",
        "status": "success",
        "relative_mae": 0.89,
        "relative_rmse": 0.91,
        "decision": "keep",
        "decision_reason": "",
    }
    audited_keep = {
        "schema_version": 2,
        "experiment_id": "exp_audited_0002",
        "candidate_family": "elastic_net",
        "commit": "a" * 40,
        "snapshot_id": "sha256:" + "b" * 64,
        "status": "success",
        "relative_mae": 0.92,
        "relative_rmse": 0.93,
        "decision": "keep",
        "decision_reason": "majority 3/3; pooled CI upper < 1.0",
    }
    ledger = _write_jsonl(
        tmp_path / "experiments.jsonl",
        [json.dumps(legacy_keep), json.dumps(audited_keep)],
    )
    report_a = tmp_path / "REPORT_A.md"
    report_b = tmp_path / "REPORT_B.md"
    generate_markdown_report(ledger, report_a)
    generate_markdown_report(ledger, report_b)

    content = report_a.read_text(encoding="utf-8")
    assert report_a.read_bytes() == report_b.read_bytes()
    # Exactly one unaudited keep, clearly labelled (IDs render truncated).
    assert content.count("| LEGACY_UNAUDITED |") == 1
    assert "exp_legacy_0" in content
    assert "Kept records without auditable provenance**: 1" in content
    assert MULTIPLICITY_POLICY in content
    # Audited keep carries no warning marker.
    audited_row = next(line for line in content.splitlines() if "exp_audited_" in line)
    assert "audited |" in audited_row


def _synthetic_frame(rows: int = 140, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=rows)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    feature = rng.normal(0, 1, rows)
    return pd.DataFrame({"FeatA": feature, "Close": close}, index=index)


def test_run_holdout_writes_durable_evidence_record(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_holdout as run_holdout

    snapshot = tmp_path / "snap.csv"
    frame = _synthetic_frame()
    frame.to_csv(snapshot)

    ledger = tmp_path / "results" / "experiments.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_holdout.py",
            str(snapshot),
            "--family",
            "random_features_ridge",
            "--horizon",
            "2",
            "--window-count",
            "2",
            "--window-rows",
            "5",
            "--min-train-rows",
            "30",
            "--window",
            "10",
            "--ledger",
            str(ledger),
        ],
    )
    exit_code = run_holdout.main()
    assert exit_code == 0

    records, corrupt = read_records(ledger)
    assert corrupt == []
    entry = next(r for r in records if r["run_tag"] == "holdout-random_features_ridge")

    # Snapshot hash is reproducible from the frozen CSV alone.
    import hashlib

    reloaded = pd.read_csv(snapshot, index_col=0, parse_dates=True)
    digest = hashlib.sha256(reloaded.to_csv(index=True).encode("utf-8")).hexdigest()
    assert entry["snapshot_id"] == "sha256:" + digest
    assert entry["protocol_version"] == "multi-window-block-bootstrap-v1"
    assert entry["multiplicity_policy"] == MULTIPLICITY_POLICY
    assert entry["decision"] in ("keep", "discard")
    assert entry["decision_reason"]
    evidence = entry["evidence"]
    assert evidence["window_count"] >= 1
    assert len(evidence["per_window"]) == evidence["window_count"]
    for window in evidence["per_window"]:
        assert set(window) >= {
            "window_index",
            "test_origin_start",
            "test_origin_end",
            "train_count",
            "relative_mae",
            "relative_rmse",
            "passes_gate",
        }
    # TSV/MD sidecars regenerate from the same ledger.
    assert (ledger.with_suffix(".tsv")).exists()
    assert (ledger.parent / "REPORT.md").exists()


def test_run_holdout_accepts_legacy_family_name(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_holdout as run_holdout

    snapshot = tmp_path / "snap.csv"
    _synthetic_frame().to_csv(snapshot)
    ledger = tmp_path / "results" / "experiments.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_holdout.py",
            str(snapshot),
            "--family",
            "small_tcn",  # legacy alias must resolve at the CLI boundary
            "--horizon",
            "2",
            "--window-count",
            "2",
            "--window-rows",
            "5",
            "--min-train-rows",
            "30",
            "--window",
            "10",
            "--ledger",
            str(ledger),
        ],
    )
    assert run_holdout.main() == 0
    records, corrupt = read_records(ledger)
    assert corrupt == []
    entry = next(r for r in records if r["run_tag"] == "holdout-random_features_ridge")
    assert entry["candidate_family"] == "random_features_ridge"
    assert "legacy name 'small_tcn'" in entry["hypothesis"]
