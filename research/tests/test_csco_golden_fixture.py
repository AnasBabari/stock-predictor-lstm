"""Phase 3 contract tests for the immutable CSCO software golden case."""

from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research.ndx100.csco_fixture import (
    CANONICAL_CSV_NAME,
    CSCO_FIXTURE_CSV,
    CSCO_FIXTURE_MANIFEST,
    DETERMINISTIC_MODELS,
    EXPECTED_ACTUAL_CLOSES,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_EVIDENCE_ROLE,
    EXPECTED_FIRST_SESSION,
    EXPECTED_LAST_SESSION,
    EXPECTED_ROW_COUNT,
    GOLDEN_METRICS,
    GOLDEN_PREDICTIONS,
    IMPLEMENTATION_SENSITIVE_MODELS,
    LAST_TRAIN_CLOSE,
    LOOSE_TOLERANCE,
    METRIC_TOLERANCE_DETERMINISTIC,
    METRIC_TOLERANCE_SENSITIVE,
    REPOSITORY_ROOT,
    TARGET_DAYS,
    TIGHT_TOLERANCE,
    TRAIN_END,
    FixtureIntegrityError,
    compute_csco_feature_frame,
    load_csco_golden_history,
    load_manifest,
    run_csco_benchmark,
    sha256_file,
    verify_canonical_fixture,
)


def _copy_fixture(tmp_path: Path) -> Path:
    fixture_dir = tmp_path / "data" / "fixtures"
    fixture_dir.mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / CSCO_FIXTURE_CSV, fixture_dir / CANONICAL_CSV_NAME)
    shutil.copy2(
        REPOSITORY_ROOT / CSCO_FIXTURE_MANIFEST,
        fixture_dir / CSCO_FIXTURE_MANIFEST.name,
    )
    return tmp_path


@pytest.fixture(scope="module")
def benchmark_result() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    def blocked_network(*_args, **_kwargs):
        raise AssertionError("golden benchmark attempted network access")

    with (
        patch("socket.create_connection", side_effect=blocked_network),
        patch.object(socket.socket, "connect", side_effect=blocked_network),
    ):
        return run_csco_benchmark(device="cpu", seed=41)


def test_manifest_and_canonical_file_identity_are_pinned():
    manifest = load_manifest()
    csv_path = REPOSITORY_ROOT / CSCO_FIXTURE_CSV

    assert manifest["evidence_role"] == EXPECTED_EVIDENCE_ROLE
    assert manifest["certification_eligible"] is False
    assert manifest["canonical_file"] == CANONICAL_CSV_NAME
    assert manifest["canonical_sha256"] == EXPECTED_CANONICAL_SHA256
    assert sha256_file(csv_path) == EXPECTED_CANONICAL_SHA256
    assert verify_canonical_fixture(csv_path) == manifest


def test_csco_fixture_semantics_match_the_golden_contract():
    history = load_csco_golden_history()

    assert len(history) == EXPECTED_ROW_COUNT
    assert history.index.min() == EXPECTED_FIRST_SESSION
    assert history.index.max() == EXPECTED_LAST_SESSION
    assert history.index.is_monotonic_increasing
    assert history.index.is_unique
    assert tuple(history.columns) == ("close", "volume")
    assert not history.isna().any().any()
    assert (history["close"] > 0).all()
    assert (history["volume"] >= 0).all()

    actuals = history.loc[list(TARGET_DAYS), "close"].to_numpy(dtype=float)
    np.testing.assert_allclose(actuals, EXPECTED_ACTUAL_CLOSES, atol=0.01)
    assert np.isclose(history.loc[TRAIN_END, "close"], LAST_TRAIN_CLOSE, atol=0.01)


@pytest.mark.parametrize("missing_name", [CANONICAL_CSV_NAME, CSCO_FIXTURE_MANIFEST.name])
def test_missing_fixture_component_fails_closed(tmp_path: Path, missing_name: str):
    root = _copy_fixture(tmp_path)
    (root / "data" / "fixtures" / missing_name).unlink()

    with pytest.raises(FixtureIntegrityError):
        load_csco_golden_history(root)


def test_corrupt_fixture_fails_hash_verification_before_parse(tmp_path: Path):
    root = _copy_fixture(tmp_path)
    csv_path = root / CSCO_FIXTURE_CSV
    csv_path.write_bytes(csv_path.read_bytes() + b"\ncorrupt")

    with pytest.raises(FixtureIntegrityError, match="hash mismatch"):
        load_csco_golden_history(root)


def test_invalid_or_repointed_manifest_fails_closed(tmp_path: Path):
    root = _copy_fixture(tmp_path)
    manifest_path = root / CSCO_FIXTURE_MANIFEST
    manifest_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(FixtureIntegrityError, match="not valid JSON"):
        load_csco_golden_history(root)

    root = _copy_fixture(tmp_path / "wrong-hash")
    manifest_path = root / CSCO_FIXTURE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FixtureIntegrityError, match="hash does not match"):
        load_csco_golden_history(root)


def test_features_are_invariant_to_future_mutation_and_append():
    history = load_csco_golden_history()
    origin = 750
    closes = history["close"].to_numpy(dtype=float)
    volumes = history["volume"].to_numpy(dtype=float)
    baseline = compute_csco_feature_frame(closes, volumes, index=history.index)

    mutated_closes = closes.copy()
    mutated_volumes = volumes.copy()
    mutated_closes[origin + 1 :] *= np.linspace(1.5, 3.0, len(closes) - origin - 1)
    mutated_volumes[origin + 1 :] += np.arange(1, len(closes) - origin) * 1_000_000
    mutated = compute_csco_feature_frame(mutated_closes, mutated_volumes, index=history.index)
    pd.testing.assert_frame_equal(baseline.iloc[: origin + 1], mutated.iloc[: origin + 1])

    future_index = pd.bdate_range(history.index[-1] + pd.Timedelta(days=1), periods=20, name="date")
    appended = compute_csco_feature_frame(
        np.concatenate([closes, np.full(20, closes[-1] * 10)]),
        np.concatenate([volumes, np.full(20, volumes[-1] * 10)]),
        index=history.index.append(future_index),
    )
    pd.testing.assert_frame_equal(baseline, appended.iloc[: len(baseline)])


def test_golden_predictions_and_metrics_use_reproducibility_class_tolerances(
    benchmark_result: tuple[pd.DataFrame, dict[str, np.ndarray]],
):
    table, predictions = benchmark_result
    assert set(predictions) == set(GOLDEN_PREDICTIONS)

    for model_name in DETERMINISTIC_MODELS:
        np.testing.assert_allclose(
            predictions[model_name],
            GOLDEN_PREDICTIONS[model_name],
            **TIGHT_TOLERANCE,
        )
    for model_name in IMPLEMENTATION_SENSITIVE_MODELS:
        np.testing.assert_allclose(
            predictions[model_name],
            GOLDEN_PREDICTIONS[model_name],
            **LOOSE_TOLERANCE,
        )

    indexed = table.set_index("model")
    for model_name, expected in GOLDEN_METRICS.items():
        tolerance = (
            METRIC_TOLERANCE_SENSITIVE
            if model_name in IMPLEMENTATION_SENSITIVE_MODELS
            else METRIC_TOLERANCE_DETERMINISTIC
        )
        for metric_name in ("mape_pct", "rmse", "mae", "r2"):
            assert float(indexed.loc[model_name, metric_name]) == pytest.approx(
                expected[metric_name], abs=tolerance
            )

    deterministic_ranking = tuple(
        table.loc[table["model"].isin(DETERMINISTIC_MODELS), "model"].tolist()
    )
    assert deterministic_ranking == (
        "drift_random_walk",
        "naive_flat",
        "ridge_lagged_returns",
        "hgb_boosting",
        "seasonal_naive_5d",
    )


def test_synthetic_generator_reproduces_canonical_fixture():
    from research.ndx100.csco_fixture import generate_synthetic_csco_fixture

    generated = generate_synthetic_csco_fixture()
    canonical = load_csco_golden_history()
    pd.testing.assert_frame_equal(generated, canonical)


def test_target_actuals_mutation_only_affects_metrics_not_predictions():
    history = load_csco_golden_history()
    baseline_table, baseline_preds = run_csco_benchmark(history=history, device="cpu", seed=41)

    mutated_history = history.copy()
    for day in TARGET_DAYS:
        mutated_history.loc[day, "close"] = float(mutated_history.loc[day, "close"]) * 1.5

    mutated_table, mutated_preds = run_csco_benchmark(
        history=mutated_history, device="cpu", seed=41
    )

    # Predictions MUST be bit-for-bit identical across all models (no target leakage)
    for model_name in baseline_preds:
        np.testing.assert_allclose(
            mutated_preds[model_name],
            baseline_preds[model_name],
            rtol=1e-12,
            atol=1e-12,
        )

    # Metrics MUST differ because actual targets were altered
    for model_name in baseline_preds:
        base_mape = float(
            baseline_table.loc[baseline_table["model"] == model_name, "mape_pct"].iloc[0]
        )
        mut_mape = float(
            mutated_table.loc[mutated_table["model"] == model_name, "mape_pct"].iloc[0]
        )
        assert abs(base_mape - mut_mape) > 1.0


def test_torch_seed_reproducibility_and_different_seed():
    history = load_csco_golden_history()
    _, preds_seed41_a = run_csco_benchmark(history=history, device="cpu", seed=41)
    _, preds_seed41_b = run_csco_benchmark(history=history, device="cpu", seed=41)
    _, preds_seed99 = run_csco_benchmark(history=history, device="cpu", seed=99)

    # Identical seed produces identical predictions
    np.testing.assert_allclose(preds_seed41_a["lstm_window"], preds_seed41_b["lstm_window"])

    # Different seed produces different neural predictions
    assert not np.allclose(preds_seed41_a["lstm_window"], preds_seed99["lstm_window"])

    # Deterministic models are unaffected by torch seed
    for model_name in DETERMINISTIC_MODELS:
        np.testing.assert_allclose(preds_seed41_a[model_name], preds_seed99[model_name])


def test_cli_requires_explicit_live_selection(monkeypatch):
    import scripts.backtest_csco_2026_07 as cli

    assert cli.build_parser().parse_args([]).live is False
    assert cli.build_parser().parse_args(["--live"]).live is True

    calls: list[str] = []
    history = load_csco_golden_history()
    fake_table = pd.DataFrame([{"model": "naive_flat", "mape_pct": 0.0}])
    fake_predictions = {"naive_flat": np.asarray([1.0] * len(TARGET_DAYS))}

    monkeypatch.setattr(cli, "load_csco_golden_history", lambda: calls.append("offline") or history)
    monkeypatch.setattr(cli, "load_live_history", lambda: calls.append("live") or history)
    monkeypatch.setattr(cli, "run_csco_benchmark", lambda **_kwargs: (fake_table, fake_predictions))
    monkeypatch.setattr(cli, "print_results", lambda *_args, **_kwargs: None)

    assert cli.main([]) == 0
    assert calls == ["offline"]
    calls.clear()
    assert cli.main(["--live"]) == 0
    assert calls == ["live"]


def test_cli_subprocess_runs_from_external_cwd(tmp_path: Path):
    import subprocess
    import sys

    script_path = REPOSITORY_ROOT / "scripts" / "backtest_csco_2026_07.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Script failed with stderr:\n{result.stderr}"
    assert "evidence role: synthetic_software_regression_fixture" in result.stdout
    assert "target week: 2026-07-20..2026-07-24" in result.stdout
    assert "lstm_window" in result.stdout
