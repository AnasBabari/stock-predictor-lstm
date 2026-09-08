import numpy as np
import pandas as pd
from services.volatility_snapshot import build_features_v5

from research.volatility_structure.panel import (
    TARGET_HORIZONS,
    apply_forecast_floor,
    arm_a_forecasts,
    arm_b1_forecasts,
    fit_ridge_direct,
    forward_realized_variance,
    partitions,
    qlike,
    range_block_features,
)
from research.volatility_structure.range_estimators import range_variances


def _gbm_frame(n=300, seed=21):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.008,
            "Low": close * 0.992,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )


def test_arm_a_matches_production_rolling_baseline():
    """The frozen champion must reproduce production Vol_C2C_20 * H exactly."""
    frame = _gbm_frame()
    mine = arm_a_forecasts(frame["Close"])
    prod = build_features_v5(frame)["Vol_C2C_20"].to_numpy() ** 2
    for h in TARGET_HORIZONS:
        expected = prod * h
        got = mine[f"f_h{h}"].to_numpy()
        mask = np.isfinite(expected) & np.isfinite(got)
        assert mask.sum() > 200
        np.testing.assert_allclose(got[mask], expected[mask], rtol=1e-12)


def test_causality_mutation_after_origin_leaves_origin_untouched():
    frame = _gbm_frame()
    base_fc = arm_a_forecasts(frame["Close"])
    base_tg = {h: forward_realized_variance(frame["Close"], h) for h in TARGET_HORIZONS}
    t0 = frame.index[200]
    mutated = frame.copy()
    mutated.loc[mutated.index > t0, "Close"] *= 10.0
    new_fc = arm_a_forecasts(mutated["Close"])
    new_tg = {h: forward_realized_variance(mutated["Close"], h) for h in TARGET_HORIZONS}
    before = frame.index <= t0
    for h in TARGET_HORIZONS:
        pd.testing.assert_series_equal(
            base_fc.loc[before, f"f_h{h}"], new_fc.loc[before, f"f_h{h}"]
        )
        # Forward labels ending on or before t0 cannot see the mutation;
        # labels spanning it must change. This is precisely why the panel
        # purges crossing labels rather than origins.
        settled = frame.index[: 200 - h + 1]
        pd.testing.assert_series_equal(base_tg[h].loc[settled], new_tg[h].loc[settled])
        spanning = frame.index[200 - h + 1 : 201]
        assert (base_tg[h].loc[spanning] != new_tg[h].loc[spanning]).all()
    # Sensitivity: the mutation must move at least one later forecast/target.
    assert (base_fc["f_h5"].dropna() != new_fc["f_h5"].dropna()).any()
    assert (base_tg[5].dropna() != new_tg[5].dropna()).any()


def test_partitions_purge_crossing_labels_on_synthetic_calendar():
    # 300 sessions so even h=20 keeps non-empty partitions on both sides.
    dates = pd.bdate_range("2022-01-03", periods=300).to_numpy(dtype="datetime64[ns]")
    sessions = pd.DatetimeIndex(dates)
    for h in TARGET_HORIZONS:
        label_end = pd.DatetimeIndex(
            [sessions[min(i + h, len(sessions) - 1)] for i in range(len(sessions))]
        ).to_numpy(dtype="datetime64[ns]")
        train, validation, val_start, _ = partitions(dates, label_end)
        assert train.any() and validation.any()
        assert (label_end[train] < np.datetime64(val_start)).all()
        val_label_end = label_end[validation]
        assert (val_label_end >= np.datetime64(val_start)).all()


def test_qlike_prefers_accurate_forecast_and_floors_degenerate_inputs():
    assert qlike(0.04, 0.04) == 0.0
    assert qlike(0.04, 0.08) > qlike(0.04, 0.05) > 0.0
    assert qlike(0.04, 0.02) > qlike(0.04, 0.03) > 0.0
    assert qlike(0.0, 0.0) == 0.0
    assert qlike(-1.0, -2.0) == 0.0


def _ohlc_frame(n=300, seed=21):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"o": open_, "h": high, "l": low, "c": close}, index=index)


def test_each_range_estimator_is_causal_under_post_origin_mutation():
    frame = _ohlc_frame()
    t0 = frame.index[200]
    base_range = range_variances(frame["o"], frame["h"], frame["l"], frame["c"])
    base_b1 = arm_b1_forecasts(base_range)
    base_block = range_block_features(base_range)
    mutated = frame.copy()
    for column in ("o", "h", "l", "c"):
        mutated.loc[mutated.index > t0, column] *= 10.0
    new_range = range_variances(mutated["o"], mutated["h"], mutated["l"], mutated["c"])
    new_b1 = arm_b1_forecasts(new_range)
    new_block = range_block_features(new_range)
    before = frame.index <= t0
    for column in ("parkinson_var", "garman_klass_var", "rogers_satchell_var", "yang_zhang_var"):
        pd.testing.assert_series_equal(
            base_range.loc[before, column], new_range.loc[before, column]
        )
    for arm, table in base_b1.items():
        for column in table.columns:
            pd.testing.assert_series_equal(
                table.loc[before, column], new_b1[arm].loc[before, column]
            )
    pd.testing.assert_frame_equal(base_block.loc[before], new_block.loc[before])
    # Sensitivity: later windows must move.
    assert (base_range["parkinson_var"].dropna() != new_range["parkinson_var"].dropna()).any()


def test_floor_applies_only_at_forecast_boundary_and_preserves_raw_features():
    frame = _ohlc_frame()
    raw = range_variances(frame["o"], frame["h"], frame["l"], frame["c"])
    block = range_block_features(raw)
    # Feature construction preserves raw values exactly (negatives included);
    # flooring happens only at the forecast boundary.
    expected_gk = raw["garman_klass_var"].rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(block["range_gk_20"], expected_gk, check_names=False)
    b1 = arm_b1_forecasts(raw)
    floored = apply_forecast_floor(b1["B1-garman_klass"])
    assert (
        floored.filter(like="f_h").to_numpy()[np.isfinite(floored.filter(like="f_h").to_numpy())]
        >= 0
    ).all()
    # Identity on already-valid forecasts (Arm A comparability untouched).
    a_like = pd.DataFrame({"f_h5": [0.0, 0.001, np.nan]})
    pd.testing.assert_frame_equal(apply_forecast_floor(a_like), a_like)


def test_ridge_helper_fits_and_predicts_deterministically():
    rng = np.random.default_rng(4)
    x_train = rng.normal(size=(500, 4))
    y_train = x_train @ np.array([1.0, -0.5, 0.25, 0.0]) + rng.normal(scale=0.1, size=500)
    scaler, model = fit_ridge_direct(x_train, y_train)
    first = model.predict(scaler.transform(x_train[:10]))
    scaler2, model2 = fit_ridge_direct(x_train, y_train)
    np.testing.assert_array_equal(model.predict(scaler.transform(x_train[:10])), first)
    assert model.alpha == 100.0
    assert scaler2.mean_.shape == (4,)


def test_comparison_joins_on_validation_keys_only():
    """Regression: train origins present in both arms must not leak into
    the pairwise comparison (caught live: 5810 joined vs 2586 validation)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.run_vol_structure_ablation import _with_scores, compare_on_common_sample

    dates = pd.bdate_range("2022-01-03", periods=300)
    rows = []
    for arm in ("A", "B1-parkinson"):
        for horizon in (5, 10, 20):
            for i, day in enumerate(dates):
                rows.append(
                    {
                        "ticker": "T",
                        "horizon": horizon,
                        "origin_date": day,
                        "target_end_date": dates[min(i + horizon, 299)],
                        "arm": arm,
                        "forecast": 0.001,
                        "realized": 0.0011,
                    }
                )
    frame = pd.DataFrame(rows)
    scored = _with_scores(frame)
    comparison = compare_on_common_sample(scored[scored.arm == "A"], scored, "B1-parkinson")
    # Validation block is unique[210:255]; an h-session label stays inside
    # only for the first 45-h of those dates.
    for horizon in (5, 10, 20):
        cell = comparison[horizon]
        assert cell["coverage"]["common_origins"] == 45 - horizon
    assert comparison[5]["coverage"]["base_origins"] == 300
