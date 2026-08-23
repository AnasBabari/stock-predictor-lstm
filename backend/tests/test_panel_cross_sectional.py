"""Unit tests and leakage canaries for Protocol V3 cross-sectional features and targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from panel.cross_sectional import (
    compute_cross_sectional_ranks,
    compute_relative_forward_returns,
)
from panel.features import build_features_v5


def make_synthetic_panel(
    n_tickers: int = 40, n_sessions: int = 300, seed: int = 42
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_sessions)
    panels: dict[str, pd.DataFrame] = {}

    for i in range(n_tickers):
        ticker = f"TK_{i:02d}"
        drift = rng.normal(0.0004, 0.015, n_sessions)
        close = 100.0 * np.exp(np.cumsum(drift))
        openp = close * np.exp(rng.normal(0, 0.003, n_sessions))
        high = np.maximum(openp, close) * np.exp(np.abs(rng.normal(0, 0.003, n_sessions)))
        low = np.minimum(openp, close) * np.exp(-np.abs(rng.normal(0, 0.003, n_sessions)))
        volume = rng.integers(50_000, 2_000_000, n_sessions).astype(float)

        raw_df = pd.DataFrame(
            {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        panels[ticker] = build_features_v5(raw_df)

    return panels


def test_cross_sectional_rank_range_and_ties():
    dates = pd.bdate_range("2024-01-01", periods=5)
    tickers = [f"T{i}" for i in range(10)]
    panels = {}
    for t in tickers:
        df = pd.DataFrame(
            {
                "Return_1D": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Return_5D": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Return_10D": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Return_20D": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Overnight_Return": [0.01, 0.02, 0.03, 0.04, 0.05],
                "OpenToClose_Return": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Vol_C2C_20": [0.01, 0.02, 0.03, 0.04, 0.05],
                "EWMA_Var": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Vol_Percentile_252": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Volume_Surprise": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Log_Dollar_Volume": [0.01, 0.02, 0.03, 0.04, 0.05],
                "Amihud_Illiquidity_20": [0.01, 0.02, 0.03, 0.04, 0.05],
            },
            index=dates,
        )
        panels[t] = df

    # All assets identical -> rank should be 0.0
    ranked = compute_cross_sectional_ranks(panels, min_reference_assets=5)
    for t in tickers:
        for c in ["Return_1D_CS_Rank", "Vol_C2C_20_CS_Rank"]:
            assert np.allclose(ranked[t][c], 0.0)


def test_held_out_feature_isolation_canary():
    """Canary: Perturbing held-out (H) feature values must not alter development (D) ranks."""
    panels = make_synthetic_panel(n_tickers=35, n_sessions=100, seed=123)
    dev_tickers = [f"TK_{i:02d}" for i in range(28)]
    transfer_tickers = [f"TK_{i:02d}" for i in range(28, 35)]

    ranked_orig = compute_cross_sectional_ranks(
        panels, dev_tickers=dev_tickers, min_reference_assets=20
    )

    # Corrupt held-out features drastically
    panels_perturbed = {t: df.copy() for t, df in panels.items()}
    for t in transfer_tickers:
        panels_perturbed[t]["Return_20D"] = 99999.0
        panels_perturbed[t]["Vol_C2C_20"] = -99999.0

    ranked_perturbed = compute_cross_sectional_ranks(
        panels_perturbed, dev_tickers=dev_tickers, min_reference_assets=20
    )

    # Assert D asset ranks are bit-for-bit identical
    for t in dev_tickers:
        for col in ["Return_20D_CS_Rank", "Vol_C2C_20_CS_Rank"]:
            np.testing.assert_array_equal(
                ranked_orig[t][col].values,
                ranked_perturbed[t][col].values,
                err_msg=f"Leakage detected: D asset '{t}' feature rank changed when H was perturbed!",
            )


def test_held_out_return_isolation_canary():
    """Canary: Perturbing held-out (H) forward returns must not alter development (D) relative targets."""
    panels = make_synthetic_panel(n_tickers=35, n_sessions=100, seed=456)
    dev_tickers = [f"TK_{i:02d}" for i in range(28)]
    transfer_tickers = [f"TK_{i:02d}" for i in range(28, 35)]

    _, rel_orig = compute_relative_forward_returns(
        panels, horizon=5, dev_tickers=dev_tickers, min_reference_assets=20
    )

    # Corrupt held-out price histories
    panels_perturbed = {t: df.copy() for t, df in panels.items()}
    for t in transfer_tickers:
        panels_perturbed[t]["Close"] = panels_perturbed[t]["Close"] * 1000.0

    _, rel_perturbed = compute_relative_forward_returns(
        panels_perturbed, horizon=5, dev_tickers=dev_tickers, min_reference_assets=20
    )

    # Assert D relative targets are identical
    for t in dev_tickers:
        np.testing.assert_allclose(
            rel_orig[t].dropna().values,
            rel_perturbed[t].dropna().values,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Leakage detected: D asset '{t}' target changed when H future returns were modified!",
        )


def test_target_leave_one_out_benchmark():
    """Verify that development relative target uses LOO mean excluding the asset itself."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    # 3 assets with known forward returns: A=0.10, B=0.20, C=0.30
    panels = {
        "A": pd.DataFrame({"Close": [100.0, 110.0] + [110.0] * 8}, index=dates),
        "B": pd.DataFrame({"Close": [100.0, 120.0] + [120.0] * 8}, index=dates),
        "C": pd.DataFrame({"Close": [100.0, 130.0] + [130.0] * 8}, index=dates),
    }

    raw, rel = compute_relative_forward_returns(
        panels, horizon=1, dev_tickers=["A", "B", "C"], min_reference_assets=3
    )

    r_A = raw["A"].iloc[0]  # log(1.1) ~ 0.09531
    r_B = raw["B"].iloc[0]  # log(1.2) ~ 0.18232
    r_C = raw["C"].iloc[0]  # log(1.3) ~ 0.26236

    # LOO Benchmark for A: (r_B + r_C) / 2
    loo_bm_A = (r_B + r_C) / 2.0
    expected_rel_A = r_A - loo_bm_A

    np.testing.assert_allclose(rel["A"].iloc[0], expected_rel_A, rtol=1e-6)


def test_transfer_d_only_target_benchmark():
    """Verify that held-out transfer (H) relative target subtracts full D mean (no H returns)."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    # Dev assets: A=0.10, B=0.20; Transfer assets: H1=0.50, H2=1.00
    panels = {
        "A": pd.DataFrame({"Close": [100.0, 110.0] + [110.0] * 8}, index=dates),
        "B": pd.DataFrame({"Close": [100.0, 120.0] + [120.0] * 8}, index=dates),
        "H1": pd.DataFrame({"Close": [100.0, 150.0] + [150.0] * 8}, index=dates),
        "H2": pd.DataFrame({"Close": [100.0, 200.0] + [200.0] * 8}, index=dates),
    }

    raw, rel = compute_relative_forward_returns(
        panels, horizon=1, dev_tickers=["A", "B"], min_reference_assets=2
    )

    r_A = raw["A"].iloc[0]
    r_B = raw["B"].iloc[0]
    r_H1 = raw["H1"].iloc[0]

    # Benchmark for H1 must be EXACT mean of D only: (r_A + r_B) / 2
    d_mean = (r_A + r_B) / 2.0
    expected_rel_H1 = r_H1 - d_mean

    np.testing.assert_allclose(rel["H1"].iloc[0], expected_rel_H1, rtol=1e-6)
    # Ensure H2 returns NEVER affected H1 benchmark
    assert not np.isclose(rel["H1"].iloc[0], r_H1 - (r_A + r_B + raw["H2"].iloc[0]) / 3.0)


def test_cross_sectional_rank_missing_and_tied_assets():
    """Verify proper scaling [-0.5, 0.5] with ties and NaN feature missingness."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    # 4 dev assets: T0=0.01, T1=0.02, T2=0.02 (tie), T3=NaN (missing)
    panels = {
        "T0": pd.DataFrame({"Return_20D": [0.01] * 5}, index=dates),
        "T1": pd.DataFrame({"Return_20D": [0.02] * 5}, index=dates),
        "T2": pd.DataFrame({"Return_20D": [0.02] * 5}, index=dates),
        "T3": pd.DataFrame({"Return_20D": [np.nan] * 5}, index=dates),
    }

    # Minimum 3 reference assets: valid assets = T0, T1, T2 (count=3)
    ranked = compute_cross_sectional_ranks(
        panels,
        dev_tickers=["T0", "T1", "T2", "T3"],
        base_columns=("Return_20D",),
        min_reference_assets=3,
    )

    # T0 is rank 1 of 3: (1 - 1)/(3 - 1) - 0.5 = 0/2 - 0.5 = -0.5
    np.testing.assert_allclose(ranked["T0"]["Return_20D_CS_Rank"], -0.5)

    # T1 and T2 tie for ranks 2 and 3 -> average rank = 2.5: (2.5 - 1)/(3 - 1) - 0.5 = 1.5/2 - 0.5 = 0.25
    np.testing.assert_allclose(ranked["T1"]["Return_20D_CS_Rank"], 0.25)
    np.testing.assert_allclose(ranked["T2"]["Return_20D_CS_Rank"], 0.25)

    # T3 is NaN -> rank must be NaN
    assert ranked["T3"]["Return_20D_CS_Rank"].isna().all()
