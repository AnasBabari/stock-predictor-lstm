from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
from volatility_forecasting.data import VolatilityPanelExamples
from volatility_forecasting.folds import VolatilityFold, VolatilityFoldPlan
from volatility_forecasting.model import TorchTrainingConfig
from volatility_forecasting.prospective import OBJECTIVE_PROFILES, prospective_protocol
from volatility_forecasting.resume import (
    REPORT_FILENAME,
    ProspectiveResumeError,
    atomic_write_json,
    build_run_manifest,
    expected_fold_summaries,
    expected_oof_identity,
    protocol_fingerprint,
    record_filename,
    scan_run_directory,
    validate_seed_record,
)

PROTOCOL = prospective_protocol()
TRAINING = TorchTrainingConfig()
PROFILE = OBJECTIVE_PROFILES["multitask_v1"]
PANEL_CHECKSUM = "f" * 64
DEVICE = "cuda"
RESAMPLES = 1000
SEEDS = (41, 42, 43)
PROFILES = ("multitask_v1", "volatility_only_v1")


def _identity() -> dict[str, object]:
    return {"rows": 120, "first_index": 5, "last_index": 400, "sha256": "ab" * 32}


def _fold_summaries(count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "fold": fold,
            "fit_rows": 900 + fold,
            "early_stopping_rows": 63,
            "fit_end": f"2023-0{fold}-01",
            "early_stopping_start": f"2023-0{fold}-10",
            "early_stopping_end": f"2023-0{fold}-20",
            "validation_start": f"2023-0{fold + 2}-01",
            "validation_end": f"2023-0{fold + 2}-28",
            "rows": 60,
        }
        for fold in range(1, count + 1)
    ]


def _fold_record(summary: dict[str, object], *, best_epoch: int = 2) -> dict[str, object]:
    length = min(best_epoch + TRAINING.patience, TRAINING.maximum_epochs)
    return {
        **summary,
        "best_epoch": best_epoch,
        "training_history": [
            {"epoch": float(epoch), "volatility_selection": 0.5} for epoch in range(1, length + 1)
        ],
        "duration_seconds": 100.5,
        "parameter_count": 54321,
        "variance_scale": [1.0] * len(PROTOCOL.horizons),
    }


def _promotion_row(horizon: int, relative_qlike: float = 0.93) -> dict[str, object]:
    return {
        "horizon": horizon,
        "promoted": True,
        "volatility_promoted": True,
        "return_distribution_promoted": False,
        "return_location_promoted": False,
        "direction_promoted": False,
        "holm_significant": True,
        "relative_qlike": relative_qlike,
        "relative_qlike_upper_95": relative_qlike + 0.02,
        "relative_gaussian_crps": 0.97,
        "relative_variance_only_gaussian_crps": 0.96,
        "relative_return_mae": 1.01,
        "relative_return_rmse": 1.02,
        "dm_statistic": -4.2,
        "dm_p_value": 0.0001,
        "coverage_80": 0.80,
        "worst_fold_relative_qlike": 0.99,
        "folds_beating_baseline": 5,
        "reasons": [],
        "return_distribution_reasons": ["coverage outside band"],
        "return_location_reasons": ["not promoted"],
        "direction_reasons": ["diagnostic"],
    }


def _pooled_row(horizon: int, relative_qlike: float = 0.93) -> dict[str, object]:
    return {"horizon": horizon, "relative_qlike": relative_qlike, "rows": 120}


def _manifest(seed: int) -> dict[str, object]:
    return build_run_manifest(
        profile_name=PROFILE.name,
        loss_weights=PROFILE.loss_weights,
        seed=seed,
        quick=False,
        panel_checksum=PANEL_CHECKSUM,
        panel_id="panel-test",
        device=DEVICE,
        resamples=RESAMPLES,
        training=TRAINING,
        architecture_manifest={"encoder_family": "tcn", "channels": 48},
        protocol=PROTOCOL,
    )


def _record(
    seed: int = 41,
    *,
    identity: dict[str, object] | None = None,
    summaries: list[dict[str, object]] | None = None,
    with_manifest: bool = True,
) -> dict[str, object]:
    oof = identity or _identity()
    folds = summaries or _fold_summaries()
    record: dict[str, object] = {
        "protocol_version": PROTOCOL.protocol_version,
        "seed": seed,
        "oof_rows": oof["rows"],
        "oof_identity": {
            "first_index": oof["first_index"],
            "last_index": oof["last_index"],
            "sha256": oof["sha256"],
        },
        "folds": [_fold_record(summary) for summary in folds],
        "promotion": [_promotion_row(horizon) for horizon in PROTOCOL.horizons],
        "pooled_metrics": [_pooled_row(horizon) for horizon in PROTOCOL.horizons],
    }
    if with_manifest:
        record["run_manifest"] = _manifest(seed)
    return record


def _validate(record: object, seed: int = 41, **overrides) -> dict[str, object]:
    parameters: dict[str, object] = {
        "profile_name": PROFILE.name,
        "loss_weights": PROFILE.loss_weights,
        "seed": seed,
        "protocol": PROTOCOL,
        "training": TRAINING,
        "oof_identity": _identity(),
        "fold_summaries": _fold_summaries(),
        "panel_checksum": PANEL_CHECKSUM,
        "device": DEVICE,
        "resamples": RESAMPLES,
        "architecture_manifest": {"encoder_family": "tcn", "channels": 48},
        "accept_missing_manifest": False,
    }
    parameters.update(overrides)
    return validate_seed_record(record, **parameters)


def test_valid_record_round_trips_through_json() -> None:
    record = json.loads(json.dumps(_record()))
    validated = _validate(record)
    assert validated["seed"] == 41
    assert validated["run_manifest"]["panel_checksum"] == PANEL_CHECKSUM


def test_partial_record_fails() -> None:
    record = _record()
    del record["promotion"]
    with pytest.raises(ProspectiveResumeError, match="partial or truncated"):
        _validate(record)


def test_non_object_and_unknown_field_records_fail() -> None:
    with pytest.raises(ProspectiveResumeError, match="JSON object"):
        _validate(["not", "a", "record"])
    record = _record()
    record["extra"] = 1
    with pytest.raises(ProspectiveResumeError, match="unknown fields"):
        _validate(record)


def test_wrong_seed_fails() -> None:
    with pytest.raises(ProspectiveResumeError, match="seed does not match"):
        _validate(_record(seed=42), seed=41)


def test_wrong_protocol_version_fails() -> None:
    record = _record()
    record["protocol_version"] = "global-volatility-distribution-v6"
    with pytest.raises(ProspectiveResumeError, match="protocol_version"):
        _validate(record)


def test_tampered_oof_identity_fails() -> None:
    record = _record()
    record["oof_identity"]["sha256"] = "cd" * 32
    with pytest.raises(ProspectiveResumeError, match="checksum does not match"):
        _validate(record)
    record = _record()
    record["oof_rows"] = 121
    with pytest.raises(ProspectiveResumeError, match="oof_rows"):
        _validate(record)


def test_fold_boundary_mismatch_fails() -> None:
    record = _record()
    record["folds"][0]["validation_end"] = "2031-01-01"
    with pytest.raises(ProspectiveResumeError, match="validation_end"):
        _validate(record)


def test_quick_training_trace_cannot_satisfy_full_run() -> None:
    record = _record()
    # A --quick screen caps training at three epochs with patience two, so its
    # trace can never satisfy min(best_epoch + 8, 60).
    record["folds"][0]["best_epoch"] = 1
    record["folds"][0]["training_history"] = [
        {"epoch": 1.0},
        {"epoch": 2.0},
        {"epoch": 3.0},
    ]
    with pytest.raises(ProspectiveResumeError, match="quick or non-default"):
        _validate(record)


def test_missing_horizon_coverage_fails() -> None:
    record = _record()
    record["promotion"] = record["promotion"][:-1]
    with pytest.raises(ProspectiveResumeError, match="every preregistered horizon"):
        _validate(record)
    record = _record()
    record["promotion"][2]["horizon"] = 6
    with pytest.raises(ProspectiveResumeError, match="horizon order"):
        _validate(record)


def test_pooled_and_promotion_disagreement_fails() -> None:
    record = _record()
    record["pooled_metrics"][0]["relative_qlike"] = 0.80
    with pytest.raises(ProspectiveResumeError, match="disagree"):
        _validate(record)


def test_manifest_mismatches_fail() -> None:
    record = _record()
    record["run_manifest"]["panel_checksum"] = "0" * 64
    with pytest.raises(ProspectiveResumeError, match="panel_checksum"):
        _validate(record)
    record = _record()
    record["run_manifest"]["quick"] = True
    with pytest.raises(ProspectiveResumeError, match="quick"):
        _validate(record)
    record = _record()
    record["run_manifest"]["loss_weights"] = asdict(
        OBJECTIVE_PROFILES["volatility_only_v1"].loss_weights
    )
    with pytest.raises(ProspectiveResumeError, match="loss_weights"):
        _validate(record)
    record = _record()
    record["run_manifest"]["architecture"] = {"encoder_family": "patch_transformer"}
    with pytest.raises(ProspectiveResumeError, match="architecture"):
        _validate(record)
    record = _record()
    record["run_manifest"]["protocol_sha256"] = "9" * 64
    with pytest.raises(ProspectiveResumeError, match="protocol_sha256"):
        _validate(record)


def test_legacy_record_requires_explicit_acceptance() -> None:
    record = _record(with_manifest=False)
    with pytest.raises(ProspectiveResumeError, match="predates embedded run manifests"):
        _validate(record)
    validated = _validate(record, accept_missing_manifest=True)
    assert "run_manifest" not in validated


def test_scan_accepts_partial_grid_and_reports_paths(tmp_path) -> None:
    for profile, seed in (
        ("multitask_v1", 41),
        ("multitask_v1", 42),
        ("multitask_v1", 43),
        ("volatility_only_v1", 41),
    ):
        (tmp_path / record_filename(profile, seed)).write_text("{}", encoding="utf-8")
    found = scan_run_directory(tmp_path, PROFILES, SEEDS)
    assert set(found) == {
        ("multitask_v1", 41),
        ("multitask_v1", 42),
        ("multitask_v1", 43),
        ("volatility_only_v1", 41),
    }
    # The interrupted volatility_only_v1 seeds 42/43 are simply absent, so the
    # caller re-evaluates them rather than trusting anything partial.
    assert ("volatility_only_v1", 42) not in found


def test_scan_rejects_unknown_stray_and_temp_files(tmp_path) -> None:
    (tmp_path / record_filename("multitask_v1", 41)).write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("scratch", encoding="utf-8")
    with pytest.raises(ProspectiveResumeError, match="unknown file"):
        scan_run_directory(tmp_path, PROFILES, SEEDS)
    (tmp_path / "notes.txt").unlink()
    (tmp_path / (record_filename("multitask_v1", 42) + ".tmp")).write_text("{", encoding="utf-8")
    with pytest.raises(ProspectiveResumeError, match="unknown file"):
        scan_run_directory(tmp_path, PROFILES, SEEDS)


def test_scan_rejects_unrequested_profile_or_seed(tmp_path) -> None:
    (tmp_path / record_filename("multitask_v1", 44)).write_text("{}", encoding="utf-8")
    with pytest.raises(ProspectiveResumeError, match="unknown file"):
        scan_run_directory(tmp_path, PROFILES, SEEDS)
    (tmp_path / record_filename("multitask_v1", 44)).unlink()
    (tmp_path / record_filename("news_v9", 41)).write_text("{}", encoding="utf-8")
    with pytest.raises(ProspectiveResumeError, match="unknown file"):
        scan_run_directory(tmp_path, PROFILES, SEEDS)


def test_scan_rejects_completed_report_and_directories(tmp_path) -> None:
    (tmp_path / REPORT_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ProspectiveResumeError, match="final report"):
        scan_run_directory(tmp_path, PROFILES, SEEDS)
    (tmp_path / REPORT_FILENAME).unlink()
    (tmp_path / "subdir").mkdir()
    with pytest.raises(ProspectiveResumeError, match="unexpected entry"):
        scan_run_directory(tmp_path, PROFILES, SEEDS)


def test_corrupt_json_record_is_not_silently_accepted(tmp_path) -> None:
    path = tmp_path / record_filename("multitask_v1", 41)
    path.write_text('{"protocol_version": "global-volat', encoding="utf-8")
    found = scan_run_directory(tmp_path, PROFILES, SEEDS)
    with pytest.raises(json.JSONDecodeError):
        json.loads(found[("multitask_v1", 41)].read_text(encoding="utf-8"))


def _tiny_examples(rows: int = 200) -> VolatilityPanelExamples:
    horizon_count = len(PROTOCOL.horizons)
    dates = np.datetime64("2020-01-01", "D") + np.arange(rows)
    return VolatilityPanelExamples(
        features=np.ones((rows, 4, 3), dtype=np.float32),
        baseline_variance=np.full((rows, horizon_count), 0.5, dtype=np.float32),
        realized_variance=np.full((rows, horizon_count), 0.4, dtype=np.float32),
        cumulative_returns=np.zeros((rows, horizon_count), dtype=np.float32),
        direction_classes=np.ones((rows, horizon_count), dtype=np.int64),
        tickers=np.asarray(["AAA"] * rows, dtype=str),
        origin_dates=dates,
        origin_closes=np.full(rows, 10.0, dtype=np.float64),
        horizons=PROTOCOL.horizons,
        feature_names=("a", "b", "c"),
    )


def _tiny_plan(examples: VolatilityPanelExamples) -> VolatilityFoldPlan:
    fold = VolatilityFold(
        fold=1,
        train_indices=np.arange(150, dtype=np.int64),
        validation_indices=np.arange(170, 190, dtype=np.int64),
        train_end=examples.origin_dates[149],
        validation_start=examples.origin_dates[170],
        validation_end=examples.origin_dates[189],
    )
    return VolatilityFoldPlan(
        folds=(fold,),
        train_tickers=("AAA",),
        asset_holdout_tickers=("ZZZ",),
        temporal_certification_indices=np.empty(0, dtype=np.int64),
        asset_transfer_certification_indices=np.empty(0, dtype=np.int64),
        certification_start=np.datetime64("2026-08-27", "D"),
    )


def test_expected_identity_and_folds_match_evaluation_semantics() -> None:
    examples = _tiny_examples()
    plan = _tiny_plan(examples)
    identity = expected_oof_identity(examples.tickers, examples.origin_dates, plan)
    ordered = np.arange(170, 190, dtype=np.int64)
    assert identity == {
        "rows": 20,
        "first_index": 170,
        "last_index": 189,
        "sha256": hashlib.sha256(ordered.tobytes()).hexdigest(),
    }
    summaries = expected_fold_summaries(examples, plan, PROTOCOL)
    assert summaries == (
        {
            "fold": 1,
            "fit_rows": 57,
            "early_stopping_rows": 63,
            "fit_end": str(examples.origin_dates[56]),
            "early_stopping_start": str(examples.origin_dates[87]),
            "early_stopping_end": str(examples.origin_dates[149]),
            "validation_start": str(examples.origin_dates[170]),
            "validation_end": str(examples.origin_dates[189]),
            "rows": 20,
        },
    )


def test_protocol_fingerprint_is_stable_and_schema_sensitive() -> None:
    assert protocol_fingerprint(PROTOCOL) == protocol_fingerprint(prospective_protocol())
    from volatility_forecasting.contracts import VolatilityForecastProtocol

    altered = VolatilityForecastProtocol(
        protocol_version=PROTOCOL.protocol_version,
        architecture_version=PROTOCOL.architecture_version,
        window_size=59,
    )
    assert protocol_fingerprint(altered) != protocol_fingerprint(PROTOCOL)


def test_atomic_write_json_leaves_no_temp_file(tmp_path) -> None:
    path = tmp_path / "record.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert [entry.name for entry in tmp_path.iterdir()] == ["record.json"]
    atomic_write_json(path, {"a": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 3}
