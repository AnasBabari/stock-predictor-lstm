"""Focused tests for the additive V11.2 protocol and sealing boundary."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from research.volatility_forecasting.v11_2_evaluation import (
    evaluate_m0_adequacy,
    holm_adjust,
    session_block_bootstrap_ci,
)
from research.volatility_forecasting.v11_2_protocol import V112Protocol, feature_schema_digest
from research.volatility_forecasting.v11_2_sealed_store import (
    V112SealedAccessError,
    load_v112_development,
    seal_v112_dataset,
    unseal_v112_test_once,
)
from research.volatility_forecasting.v11_2_split import (
    create_v112_expanding_folds,
    create_v112_split,
)
from research.volatility_forecasting.v11_2_trainer import (
    make_forecast,
    select_per_horizon_challenger,
    train_epoch_zero_residual_model,
)
from scripts.run_v11_2_numeric_development import _fit_har, _persistence_variance


def _dates(count: int) -> list[str]:
    start = dt.date(2020, 1, 1)
    return [(start + dt.timedelta(days=i)).isoformat() for i in range(count)]


def test_protocol_is_numeric_only_and_horizon_specific() -> None:
    protocol = V112Protocol()
    assert protocol.universe_size == 64
    assert protocol.horizons == (1, 3, 5, 7)
    assert protocol.selection == "per_horizon"
    assert protocol.news_mode == "M2_DISABLED_BY_PROTOCOL"
    assert protocol.feature_schema_version == "deployable_v5"
    assert len(protocol.feature_names) == 26
    assert len(protocol.digest()) == 64


def test_v112_split_groups_rows_by_session_and_hashes_assignments() -> None:
    dates = [date for date in _dates(500) for _ in range(2)]
    split = create_v112_split(dates, [f"SEC-{i % 2}" for i in range(len(dates))])
    assert set(split.train_indices).isdisjoint(split.validation_indices)
    assert set(split.validation_indices).isdisjoint(split.test_indices)
    assert split.test_session_count > 0
    assert split.test_rows == split.test_session_count * 2
    changed = list(dates)
    changed[0] = "2030-01-01"
    changed_split = create_v112_split(changed, [f"SEC-{i % 2}" for i in range(len(changed))])
    assert changed_split.assignment_sha256 != split.assignment_sha256


def test_expanding_folds_stay_inside_development_and_are_ordered() -> None:
    folds = create_v112_expanding_folds(_dates(800), n_folds=5, min_train_sessions=300)
    assert len(folds) == 5
    for left, right in zip(folds[:-1], folds[1:], strict=True):
        assert len(right.train_indices) > len(left.train_indices)
        assert max(left.train_indices) < min(left.validation_indices)


def test_session_block_bootstrap_is_reproducible_and_session_weighted() -> None:
    dates = _dates(100)
    dates = [date for date in dates for _ in range(2)]
    candidate = np.linspace(0.1, 0.2, len(dates))
    comparator = candidate + 0.1
    first = session_block_bootstrap_ci(
        dates,
        candidate,
        comparator,
        block_sessions=20,
        n_replicates=200,
        seed=7,
    )
    second = session_block_bootstrap_ci(
        dates,
        candidate,
        comparator,
        block_sessions=20,
        n_replicates=200,
        seed=7,
    )
    assert first == second
    assert first.mean_delta < 0
    assert first.raw_p_value < 0.05
    assert first.unique_sessions == 100
    assert first.stock_origin_observations == 200


def test_m0_adequacy_holm_corrects_all_eight_comparisons() -> None:
    dates = _dates(100)
    horizons = (1, 3, 5, 7)
    har_losses = {horizon: np.full(100, 0.1) for horizon in horizons}
    constant_losses = {horizon: np.full(100, 0.2) for horizon in horizons}
    persistence_losses = {horizon: np.full(100, 0.3) for horizon in horizons}
    har_crps = {horizon: 0.1 for horizon in horizons}
    constant_crps = {horizon: 0.2 for horizon in horizons}
    persistence_crps = {horizon: 0.3 for horizon in horizons}
    result = evaluate_m0_adequacy(
        dates=dates,
        horizons=horizons,
        har_losses_by_horizon=har_losses,
        constant_losses_by_horizon=constant_losses,
        persistence_losses_by_horizon=persistence_losses,
        har_crps_by_horizon=har_crps,
        constant_crps_by_horizon=constant_crps,
        persistence_crps_by_horizon=persistence_crps,
        block_sessions=20,
        n_replicates=200,
        seed=42,
    )
    assert set(result) == {"har_vs_constant", "har_vs_persistence"}
    assert len(result["har_vs_constant"]) == len(result["har_vs_persistence"]) == 4
    assert all(gate.passed for gates in result.values() for gate in gates)
    assert all(
        gate.holm_p_value == pytest.approx(8 / 201) for gates in result.values() for gate in gates
    )


def test_holm_adjust_preserves_order() -> None:
    assert holm_adjust([0.01, 0.04, 0.2]) == pytest.approx([0.03, 0.08, 0.2])


def test_horizon_gate_requires_qlike_and_calibrated_coverage() -> None:
    dates = _dates(100)
    target = np.full(100, 0.02, dtype=np.float64)
    realized = np.full(100, 0.02, dtype=np.float64)
    har = make_forecast("M0_HAR_BASELINE", 1, np.zeros(100), np.full(100, 0.04), target, realized)
    candidate = make_forecast(
        "RIDGE_LOCATION_HAR_SCALE", 1, np.zeros(100), np.full(100, 0.02), target, realized
    )
    selection = select_per_horizon_challenger(
        horizon=1,
        dates=dates,
        har=har,
        candidates={candidate.family: candidate},
        ranking_scores={candidate.family: 0.0},
        block_sessions=20,
        n_replicates=200,
        seed=42,
    )
    assert not selection.learned_promotion
    assert "coverage" in selection.gates[0].reason


def test_v112_baselines_use_named_v5_columns_not_legacy_positions() -> None:
    features = np.zeros((40, 26), dtype=np.float32)
    features[:, 0] = 0.2  # Return_1D
    features[:, 13] = 0.01  # Vol_C2C_5
    features[:, 15] = 0.02  # Vol_C2C_20
    features[:, 16] = 0.03  # Vol_C2C_60
    features[:, 23:26] = 1000.0  # v5 liquidity columns; must not drive HAR
    persistence = _persistence_variance(features[:1], (1, 3, 5, 7))
    assert persistence[0].tolist() == pytest.approx([0.04, 0.12, 0.20, 0.28])
    train_rv = np.full((30, 4), 0.04, dtype=np.float32)
    train_variance, eval_variance = _fit_har(features[:30], train_rv, features[30:])
    assert np.isfinite(train_variance).all()
    assert np.isfinite(eval_variance).all()


def test_encrypted_holdout_is_not_available_to_development_loader(tmp_path) -> None:
    dates = _dates(500)
    security_ids = [f"SEC-{i % 2}" for i in range(len(dates))]
    split = create_v112_split(dates, security_ids)
    features = np.arange(len(dates) * 3, dtype=np.float32).reshape(len(dates), 3)
    returns = np.zeros((len(dates), 4), dtype=np.float32)
    rv = np.ones((len(dates), 4), dtype=np.float32)
    output_dir = tmp_path / "v112"
    key_path = tmp_path / "private" / "holdout.key"
    metadata = seal_v112_dataset(
        dates=dates,
        features=features,
        returns=returns,
        rv=rv,
        split=split,
        output_dir=output_dir,
        panel_sha256="a" * 64,
        schema_sha256=feature_schema_digest(),
        key_path=key_path,
        repository_root=tmp_path / "repository",
    )
    development = load_v112_development(output_dir)
    assert len(development.train_dates) == split.train_rows
    assert metadata.test_stock_origin_observations == split.test_rows
    assert not (output_dir / "sealed" / "SEALED_TEST_OPENED.json").exists()
    assert not hasattr(development, "test_features")

    payload = unseal_v112_test_once(
        output_dir=output_dir,
        key_path=key_path,
        candidate_digest="c" * 64,
        repository_root=tmp_path / "repository",
    )
    assert len(payload.dates) == split.test_rows
    with pytest.raises(V112SealedAccessError, match="already been opened"):
        unseal_v112_test_once(
            output_dir=output_dir,
            key_path=key_path,
            candidate_digest="d" * 64,
            repository_root=tmp_path / "repository",
        )


def test_epoch_zero_is_evaluated_before_updates_and_can_be_selected() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 3, 5)).astype(np.float32)
    base = np.full(60, 0.04, dtype=np.float32)
    returns = np.zeros(60, dtype=np.float32)
    rv = np.full(60, 0.04, dtype=np.float32)
    result = train_epoch_zero_residual_model(
        x_train=x[:40],
        base_variance_train=base[:40],
        returns_train=returns[:40],
        rv_train=rv[:40],
        x_validation=x[40:],
        base_variance_validation=base[40:],
        returns_validation=returns[40:],
        rv_validation=rv[40:],
        max_epochs=2,
        patience=2,
        seed=42,
        device="cpu",
    )
    assert result.epoch_evidence[0].epoch == 0
    assert result.best_epoch == 0
    assert result.epoch_zero_crps == pytest.approx(result.epoch_evidence[0].validation_crps)
