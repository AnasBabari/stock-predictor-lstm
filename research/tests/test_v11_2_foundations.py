"""Focused tests for the additive V11.2 protocol and sealing boundary."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from research.volatility_forecasting.v11_2_evaluation import (
    holm_adjust,
    session_block_bootstrap_ci,
)
from research.volatility_forecasting.v11_2_protocol import V112Protocol
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
from research.volatility_forecasting.v11_2_trainer import train_epoch_zero_residual_model


def _dates(count: int) -> list[str]:
    start = dt.date(2020, 1, 1)
    return [(start + dt.timedelta(days=i)).isoformat() for i in range(count)]


def test_protocol_is_numeric_only_and_horizon_specific() -> None:
    protocol = V112Protocol()
    assert protocol.universe_size == 64
    assert protocol.horizons == (1, 3, 5, 7)
    assert protocol.selection == "per_horizon"
    assert protocol.news_mode == "M2_DISABLED_BY_PROTOCOL"
    assert len(protocol.digest()) == 64


def test_v112_split_groups_rows_by_session_and_hashes_assignments() -> None:
    dates = [date for date in _dates(500) for _ in range(2)]
    split = create_v112_split(dates, [f"SEC-{i % 2}" for i in range(len(dates))])
    assert set(split.train_indices).isdisjoint(split.validation_indices)
    assert set(split.validation_indices).isdisjoint(split.test_indices)
    assert split.test_session_count > 0
    assert split.test_rows == split.test_session_count * 2
    changed = list(dates)
    changed[0] = "2020-01-02"
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
    assert first.unique_sessions == 100
    assert first.stock_origin_observations == 200


def test_holm_adjust_preserves_order() -> None:
    assert holm_adjust([0.01, 0.04, 0.2]) == pytest.approx([0.03, 0.08, 0.2])


def test_encrypted_holdout_is_not_available_to_development_loader(tmp_path) -> None:
    dates = _dates(500)
    security_ids = [f"SEC-{i % 2}" for i in range(len(dates))]
    split = create_v112_split(dates, security_ids)
    features = np.arange(len(dates) * 3, dtype=np.float32).reshape(len(dates), 3)
    returns = np.zeros((len(dates), 1), dtype=np.float32)
    rv = np.ones((len(dates), 1), dtype=np.float32)
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
        schema_sha256="b" * 64,
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
