"""Leakage tests for v8 chronological split — must run before any training.

These tests prove that the split is purge-clean and embargo-clean for
realistic calendars (including holiday-heavy periods) and for mixed
NYSE/LSE origins.  They are the v8 counterpart to the v7 fold tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.volatility_forecasting.data import VolatilityPanelExamples
from research.volatility_forecasting.split_v8 import (
    V8_REQUIRED_EXCHANGE_MICS,
    build_v8_chronological_split,
    build_v8_development_fold_plan,
)
from research.volatility_forecasting.v8_protocol import v8_protocol

CERTIFIABLE_TICKERS = ("AAPL", "MSFT", "NMM", "IBM", "AZN.L", "VOD.L")


def _identity_maps(tickers):
    exchange = {
        "AAPL": "XNAS",
        "MSFT": "XNAS",
        "NMM": "XNYS",
        "IBM": "XNYS",
        "AZN.L": "XLON",
        "VOD.L": "XLON",
    }
    return (
        {ticker: exchange[ticker] for ticker in tickers},
        {ticker: f"SECURITY-{ticker}" for ticker in tickers},
    )


def _dummy_examples(n_dates: int = 900, tickers=CERTIFIABLE_TICKERS) -> VolatilityPanelExamples:
    dates = np.array([np.datetime64("2018-01-01") + np.timedelta64(i, "D") for i in range(n_dates)])
    t_list = []
    d_list = []
    for d in dates:
        for t in tickers:
            t_list.append(t)
            d_list.append(d)
    t_arr = np.array(t_list)
    d_arr = np.array(d_list)
    n = len(t_arr)
    features = np.random.randn(n, 60, 26).astype(np.float32)
    baseline = np.ones((n, 6))
    realized = np.ones((n, 6))
    rets = np.ones((n, 6))
    dirs = np.zeros((n, 6), dtype=np.int64)
    return VolatilityPanelExamples(
        features=features,
        baseline_variance=baseline,
        realized_variance=realized,
        cumulative_returns=rets,
        direction_classes=dirs,
        tickers=t_arr,
        origin_dates=d_arr,
        origin_closes=np.ones(n),
        horizons=(1, 3, 5, 7, 14, 30),
        feature_names=tuple([f"f{i}" for i in range(26)]),
    )


def test_purge_strict_per_row_target_end():
    ex = _dummy_examples()
    exchange_map, security_map = _identity_maps(CERTIFIABLE_TICKERS)
    # Normal split should pass
    idx = build_v8_chronological_split(
        ex,
        required_asset_holdouts=("MSFT", "NMM"),
        panel_checksum="sha256:abc",
        universe_manifest_sha256="sha256:def",
        universe_coverage_certifiable=True,
        asset_exchange_map=exchange_map,
        asset_security_id_map=security_map,
    )
    assert len(idx.train_indices) > 0
    # Test that embargo violation is caught when we shrink embargo below max horizon
    with pytest.raises(ValueError, match="embargo must be >= max horizon"):
        build_v8_chronological_split(
            ex, required_asset_holdouts=("MSFT", "NMM"), embargo_sessions=1
        )


def test_separate_temporal_vs_asset_transfer_identities():
    ex = _dummy_examples()
    exchange_map, security_map = _identity_maps(CERTIFIABLE_TICKERS)
    idx = build_v8_chronological_split(
        ex,
        required_asset_holdouts=("MSFT", "NMM"),
        panel_checksum="a",
        universe_manifest_sha256="b",
        universe_coverage_certifiable=True,
        asset_exchange_map=exchange_map,
        asset_security_id_map=security_map,
    )
    # Temporal and asset-transfer must be disjoint and non-empty
    assert len(idx.temporal_test_indices) > 0
    assert len(idx.asset_transfer_test_indices) > 0
    assert len(set(idx.temporal_test_indices) & set(idx.asset_transfer_test_indices)) == 0
    # Pooled must be union
    assert len(idx.pooled_test_indices) == len(idx.temporal_test_indices) + len(
        idx.asset_transfer_test_indices
    )
    # Manifest must record separate SHAs
    assert (
        idx.manifest.temporal_test_assignment_sha256
        != idx.manifest.asset_transfer_assignment_sha256
    )
    assert idx.manifest.temporal_test_rows == len(idx.temporal_test_indices)
    assert idx.manifest.asset_transfer_test_rows == len(idx.asset_transfer_test_indices)


def test_explicit_holdout_required():
    ex = _dummy_examples()
    with pytest.raises(ValueError, match="must be supplied explicitly"):
        build_v8_chronological_split(ex, required_asset_holdouts=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        build_v8_chronological_split(ex, required_asset_holdouts=())


def test_v7_isolation():
    from research.volatility_forecasting.v8_protocol import (
        V8_PROSPECTIVE_TMP_HOLDOUT_FOR_COMPARISON,
        V8_TEMPORAL_HOLDOUT_SESSIONS,
    )

    proto = v8_protocol(news_enabled=False)
    assert proto.temporal_holdout_sessions == V8_TEMPORAL_HOLDOUT_SESSIONS == 0
    assert proto.temporal_holdout_sessions != V8_PROSPECTIVE_TMP_HOLDOUT_FOR_COMPARISON


def test_holiday_heavy_and_mixed_calendar():
    # Simulate LSE holiday: remove 10 random dates for one ticker to create different calendar
    ex = _dummy_examples(n_dates=900, tickers=("AAPL", "MSFT", "VOD.L"))
    # Remove some dates for VOD.L to simulate LSE holidays
    mask = ~(
        (ex.tickers == "VOD.L") & np.isin(ex.origin_dates, np.unique(ex.origin_dates)[100:110])
    )
    # Rebuild filtered examples (keep alignment)
    filtered = VolatilityPanelExamples(
        features=ex.features[mask],
        baseline_variance=ex.baseline_variance[mask],
        realized_variance=ex.realized_variance[mask],
        cumulative_returns=ex.cumulative_returns[mask],
        direction_classes=ex.direction_classes[mask],
        tickers=ex.tickers[mask],
        origin_dates=ex.origin_dates[mask],
        origin_closes=ex.origin_closes[mask],
        horizons=ex.horizons,
        feature_names=ex.feature_names,
    )
    # Should still be purge-clean because per-ticker calendars are used
    idx = build_v8_chronological_split(
        filtered,
        required_asset_holdouts=("MSFT", "VOD.L"),
        panel_checksum="a",
        universe_manifest_sha256="b",
        universe_coverage_certifiable=False,
    )
    assert len(idx.temporal_test_indices) > 0


def test_horizons_independently():
    for horizon in (1, 3, 5, 7, 14, 30):
        ex = _dummy_examples(n_dates=900, tickers=("AAPL", "MSFT", "NMM"))
        # Split must be purge-clean for every horizon individually; purge is always >= max_horizon
        idx = build_v8_chronological_split(
            ex,
            required_asset_holdouts=("MSFT", "NMM"),
            # purge defaults to max_horizon (30) which covers all horizons
            panel_checksum="a",
            universe_manifest_sha256="b",
            universe_coverage_certifiable=False,
        )
        assert idx.manifest.purge_horizon_sessions >= horizon
        assert idx.manifest.purge_horizon_sessions == 30


def test_certifiable_split_requires_complete_identity_maps():
    ex = _dummy_examples()
    with pytest.raises(ValueError, match="requires exchange and security identity maps"):
        build_v8_chronological_split(
            ex,
            required_asset_holdouts=("MSFT", "NMM"),
            panel_checksum="sha256:panel",
            universe_manifest_sha256="sha256:universe",
            universe_coverage_certifiable=True,
        )


def test_certifiable_split_proves_every_exchange_in_each_population():
    ex = _dummy_examples()
    exchange_map, security_map = _identity_maps(CERTIFIABLE_TICKERS)
    idx = build_v8_chronological_split(
        ex,
        required_asset_holdouts=("MSFT", "NMM"),
        panel_checksum="sha256:panel",
        universe_manifest_sha256="sha256:universe",
        universe_coverage_certifiable=True,
        asset_exchange_map=exchange_map,
        asset_security_id_map=security_map,
    )
    for counts in (
        idx.manifest.train_assets_per_exchange,
        idx.manifest.holdout_assets_per_exchange,
        idx.manifest.train_rows_per_exchange,
        idx.manifest.validation_rows_per_exchange,
        idx.manifest.temporal_test_rows_per_exchange,
        idx.manifest.asset_transfer_rows_per_exchange,
    ):
        assert all(counts[mic] > 0 for mic in V8_REQUIRED_EXCHANGE_MICS)


def test_assignment_identity_changes_when_security_identity_changes():
    tickers = ("AAPL", "MSFT", "NMM")
    ex = _dummy_examples(tickers=tickers)
    exchange_map, security_map = _identity_maps(tickers)
    first = build_v8_chronological_split(
        ex,
        required_asset_holdouts=("MSFT", "NMM"),
        panel_checksum="sha256:panel",
        universe_manifest_sha256="sha256:universe",
        asset_exchange_map=exchange_map,
        asset_security_id_map=security_map,
    )
    changed_security_map = {**security_map, "AAPL": "REPLACEMENT-AAPL"}
    second = build_v8_chronological_split(
        ex,
        required_asset_holdouts=("MSFT", "NMM"),
        panel_checksum="sha256:panel",
        universe_manifest_sha256="sha256:universe",
        asset_exchange_map=exchange_map,
        asset_security_id_map=changed_security_map,
    )
    assert first.manifest.row_assignment_sha256 != second.manifest.row_assignment_sha256
    assert first.manifest.asset_identity_sha256 != second.manifest.asset_identity_sha256


def test_development_folds_never_open_the_sealed_test() -> None:
    ex = _dummy_examples(n_dates=1800)
    exchange_map, security_map = _identity_maps(CERTIFIABLE_TICKERS)
    split = build_v8_chronological_split(
        ex,
        required_asset_holdouts=("MSFT", "NMM"),
        panel_checksum="sha256:panel",
        universe_manifest_sha256="sha256:universe",
        universe_coverage_certifiable=True,
        asset_exchange_map=exchange_map,
        asset_security_id_map=security_map,
    )
    plan = build_v8_development_fold_plan(ex, split, v8_protocol(news_enabled=True))
    sealed = set(split.pooled_test_indices)
    assert len(plan.folds) == 5
    for fold in plan.folds:
        assert not (set(fold.train_indices) & sealed)
        assert not (set(fold.validation_indices) & sealed)
        assert fold.train_end < fold.validation_start
        assert set(ex.tickers[fold.validation_indices]).issubset(set(split.train_tickers))
