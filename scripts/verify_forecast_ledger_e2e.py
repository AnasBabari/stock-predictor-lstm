"""End-to-end verification of the Phase 5.1 forecast ledger lifecycle."""

import math
import shutil
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
research_dir = root_dir / "research"

for p in (root_dir, backend_dir, research_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from services.forecast_ledger import ForecastLedger  # noqa: E402


def run_simulation() -> None:
    tmp_db_dir = Path("tmp_e2e_ledger")
    tmp_db_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_db_dir / "e2e_ledger.db"
    if db_file.exists():
        db_file.unlink()

    ledger = ForecastLedger(db_file)
    print("=== Step 1: Emit Live Forecast (t=0) ===")
    f_date = "2026-08-03"
    t_date = "2026-08-10"
    p_origin = 220.50
    pred_vol = 0.2450  # 24.5% annualized vol
    recent_rv = 0.2210

    sigma_h = pred_vol * math.sqrt(5 / 252.0)
    p_lower = float(p_origin * math.exp(-1.64485 * sigma_h))
    p_upper = float(p_origin * math.exp(1.64485 * sigma_h))

    rec = ledger.record_forecast(
        forecast_date=f_date,
        ticker="AAPL",
        horizon=5,
        target_date=t_date,
        model_name="rolling_mean",
        predicted_volatility=pred_vol,
        recent_realized_volatility=recent_rv,
        origin_price=p_origin,
        lower_scenario_price=p_lower,
        upper_scenario_price=p_upper,
        record_source="live",
        model_version="deployable_v5",
        feature_set_version="deployable_feature_columns_v5",
        code_commit="bc1cddf",
        data_as_of="2026-08-03",
    )
    print(
        f"  [SUCCESS] Emitted forecast: id={rec.id}, status='{rec.status}', source='{rec.record_source}'"
    )
    assert rec.status == "pending"
    assert rec.predicted_volatility == pred_vol

    print("\n=== Step 2: Test Idempotent Duplicate Request & Provenance Conflicts ===")
    rec_retry = ledger.record_forecast(
        forecast_date=f_date,
        ticker="AAPL",
        horizon=5,
        target_date=t_date,
        model_name="rolling_mean",
        predicted_volatility=pred_vol,
        recent_realized_volatility=recent_rv,
        origin_price=p_origin,
        lower_scenario_price=p_lower,
        upper_scenario_price=p_upper,
        record_source="live",
        model_version="deployable_v5",
        feature_set_version="deployable_feature_columns_v5",
        code_commit="bc1cddf",
        data_as_of="2026-08-03",
    )
    assert rec_retry.id == rec.id
    print(f"  [SUCCESS] Exact provenance duplicate request idempotently returned id={rec_retry.id}")

    # Conflict on code commit
    try:
        ledger.record_forecast(
            forecast_date=f_date,
            ticker="AAPL",
            horizon=5,
            target_date=t_date,
            model_name="rolling_mean",
            predicted_volatility=pred_vol,
            recent_realized_volatility=recent_rv,
            origin_price=p_origin,
            lower_scenario_price=p_lower,
            upper_scenario_price=p_upper,
            record_source="live",
            code_commit="different_commit_sha",
            data_as_of="2026-08-03",
        )
        raise AssertionError("Different code commit must be rejected!")
    except ValueError as e:
        print(f"  [SUCCESS] Code commit conflict rejected: {e}")

    # Conflict on data_as_of
    try:
        ledger.record_forecast(
            forecast_date=f_date,
            ticker="AAPL",
            horizon=5,
            target_date=t_date,
            model_name="rolling_mean",
            predicted_volatility=pred_vol,
            recent_realized_volatility=recent_rv,
            origin_price=p_origin,
            lower_scenario_price=p_lower,
            upper_scenario_price=p_upper,
            record_source="live",
            code_commit="bc1cddf",
            data_as_of="2026-08-04",
        )
        raise AssertionError("Different data_as_of must be rejected!")
    except ValueError as e:
        print(f"  [SUCCESS] Data_as_of conflict rejected: {e}")

    # Conflict on recent RV
    try:
        ledger.record_forecast(
            forecast_date=f_date,
            ticker="AAPL",
            horizon=5,
            target_date=t_date,
            model_name="rolling_mean",
            predicted_volatility=pred_vol,
            recent_realized_volatility=0.2990,
            origin_price=p_origin,
            lower_scenario_price=p_lower,
            upper_scenario_price=p_upper,
            record_source="live",
            code_commit="bc1cddf",
            data_as_of="2026-08-03",
        )
        raise AssertionError("Different recent RV must be rejected!")
    except ValueError as e:
        print(f"  [SUCCESS] Recent RV conflict rejected: {e}")

    print("\n=== Step 3: Test Conflicting Prediction Rejection ===")
    rejected = False
    try:
        ledger.record_forecast(
            forecast_date=f_date,
            ticker="AAPL",
            horizon=5,
            target_date=t_date,
            model_name="rolling_mean",
            predicted_volatility=0.3500,  # Conflicting replacement!
            recent_realized_volatility=recent_rv,
            origin_price=p_origin,
            lower_scenario_price=p_lower,
            upper_scenario_price=p_upper,
            record_source="live",
            code_commit="bc1cddf",
            data_as_of="2026-08-03",
        )
    except ValueError as e:
        rejected = True
        print(f"  [SUCCESS] Conflicting prediction strictly rejected: {e}")
    assert rejected, "Conflicting forecast should have been rejected!"

    print("\n=== Step 4: Add Historical Replay Records & Verify Canonical GARCH MLE ===")
    dates_hist = pd.date_range("2026-01-02", periods=160, freq="B")
    rng = np.random.default_rng(123)
    rets = rng.normal(0.0004, 0.014, size=len(dates_hist))
    sim_prices = 200.0 * np.exp(np.cumsum(rets))
    df_market = pd.DataFrame(
        {
            "Open": sim_prices * 0.998,
            "High": sim_prices * 1.008,
            "Low": sim_prices * 0.992,
            "Close": sim_prices,
            "Volume": np.full(len(dates_hist), 1_000_000.0),
        },
        index=dates_hist,
    )
    replays_added = ledger.generate_historical_replay_ledger(
        "AAPL",
        df_market,
        horizon=5,
        model_name="rolling_mean",
        lookback_sessions=30,
    )
    print(f"  [SUCCESS] Generated {replays_added} historical replay records")

    # Check that live metrics only reflect the 1 live forecast (which is currently pending)
    live_kpi_pending = ledger.get_track_record_metrics(ticker="AAPL", record_source="live")
    replay_kpi = ledger.get_track_record_metrics(ticker="AAPL", record_source="historical_replay")
    assert live_kpi_pending["scored_forecasts"] == 0
    assert replay_kpi["scored_forecasts"] >= 20
    print(
        f"  [SUCCESS] KPI Isolation: Live scored={live_kpi_pending['scored_forecasts']}, Replay scored={replay_kpi['scored_forecasts']}"
    )

    # Verify canonical GARCH MLE equivalence
    from research.volatility_forecasting.simple_pipeline import fit_garch11_mle_from_returns

    garch_replays = ledger.generate_historical_replay_ledger(
        "AAPL",
        df_market,
        horizon=5,
        model_name="garch_11",
        lookback_sessions=10,
    )
    assert garch_replays > 0
    replay_entries = ledger.get_ledger_entries(ticker="AAPL", record_source="historical_replay")
    garch_entries = [e for e in replay_entries if e["model_name"] == "garch_11"]
    last_garch = garch_entries[0]
    garch_idx = dates_hist.strftime("%Y-%m-%d").to_list().index(last_garch["forecast_date"])
    garch_rets = np.log(
        sim_prices[garch_idx - 21 : garch_idx + 1] / sim_prices[garch_idx - 22 : garch_idx]
    )
    expected_mle_vol = fit_garch11_mle_from_returns(garch_rets, horizon=5)
    assert abs(last_garch["predicted_volatility"] - expected_mle_vol) < 1e-6
    print(
        f"  [SUCCESS] Canonical GARCH(1,1) MLE Parity: Replay={last_garch['predicted_volatility']:.5f}, Canonical={expected_mle_vol:.5f}"
    )

    print("\n=== Step 5: Future Sessions Arrive -> Settlement & Scoring ===")
    future_dates = pd.date_range("2026-08-03", periods=15, freq="B")
    live_prices = [p_origin]
    daily_rets = [
        0.005,
        -0.012,
        0.008,
        -0.004,
        0.010,
        0.002,
        -0.003,
        0.005,
        0.001,
        0.002,
        0.003,
        -0.001,
        0.002,
        0.001,
        0.000,
    ]
    for r in daily_rets[1:]:
        live_prices.append(live_prices[-1] * math.exp(r))
    df_live = pd.DataFrame({"Close": live_prices}, index=future_dates)

    scored_live = ledger.score_pending_forecasts("AAPL", df_live)
    assert scored_live >= 1
    print(f"  [SUCCESS] Scored {scored_live} live pending forecast(s)")

    live_kpi_scored = ledger.get_track_record_metrics(ticker="AAPL", record_source="live")
    assert live_kpi_scored["scored_forecasts"] == 1
    assert live_kpi_scored["mean_mae"] is not None
    assert live_kpi_scored["mean_qlike"] is not None
    print(
        f"  [SUCCESS] Live Track Record Scorecard: Mean MAE={live_kpi_scored['mean_mae'] * 100:.2f}%, Mean QLIKE={live_kpi_scored['mean_qlike']:.4f}"
    )

    print("\n=== Step 6: Verify Settled Forecast Cannot Be Overwritten ===")
    mutation_rejected = False
    try:
        ledger.record_forecast(
            forecast_date=f_date,
            ticker="AAPL",
            horizon=5,
            target_date=t_date,
            model_name="rolling_mean",
            predicted_volatility=0.2900,  # Retroactive tamper attempt!
            recent_realized_volatility=recent_rv,
            origin_price=p_origin,
            lower_scenario_price=p_lower,
            upper_scenario_price=p_upper,
            record_source="live",
            code_commit="bc1cddf",
            data_as_of="2026-08-03",
        )
    except ValueError as e:
        mutation_rejected = True
        print(f"  [SUCCESS] Settled forecast mutation rejected: {e}")
    assert mutation_rejected, "Settled forecast mutation should have been rejected!"

    print("\n=== Step 7: Test Legacy SQLite Database Migration & Coexistence ===")
    import sqlite3

    legacy_db_file = tmp_db_dir / "legacy_test.db"
    conn = sqlite3.connect(str(legacy_db_file))
    conn.execute(
        """
        CREATE TABLE forecast_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            target_date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_volatility REAL NOT NULL,
            recent_realized_volatility REAL NOT NULL,
            origin_price REAL NOT NULL,
            lower_scenario_price REAL NOT NULL,
            upper_scenario_price REAL NOT NULL,
            actual_realized_volatility REAL,
            forecast_error REAL,
            abs_error REAL,
            qlike_loss REAL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(forecast_date, ticker, horizon, model_name)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO forecast_ledger (
            forecast_date, ticker, horizon, target_date, model_name,
            predicted_volatility, recent_realized_volatility, origin_price,
            lower_scenario_price, upper_scenario_price, status, created_at
        ) VALUES (
            '2026-08-01', 'MSFT', 5, '2026-08-08', 'rolling_mean',
            0.245, 0.221, 400.0, 380.0, 420.0, 'scored', '2026-08-01T20:00:00Z'
        )
        """
    )
    conn.commit()
    conn.close()

    migrated_ledger = ForecastLedger(legacy_db_file)
    mig_entries = migrated_ledger.get_ledger_entries(ticker="MSFT")
    assert len(mig_entries) == 1
    assert mig_entries[0]["record_source"] == "live"
    assert len(mig_entries[0]["forecast_fingerprint"]) == 64

    # Test that replay record can now coexist with live record under same (date, ticker, horizon, model)
    migrated_ledger.record_forecast(
        forecast_date="2026-08-01",
        ticker="MSFT",
        horizon=5,
        target_date="2026-08-08",
        model_name="rolling_mean",
        predicted_volatility=0.245,
        recent_realized_volatility=0.221,
        origin_price=400.0,
        lower_scenario_price=380.0,
        upper_scenario_price=420.0,
        record_source="historical_replay",
        code_commit="mig_replay_commit",
        data_as_of="2026-08-01",
    )
    all_msft = migrated_ledger.get_ledger_entries(ticker="MSFT")
    assert len(all_msft) == 2
    print(
        f"  [SUCCESS] Legacy DB migrated without loss: {len(all_msft)} records coexisting (live + replay)"
    )

    print("\n=== Step 8: Test Fail-Closed Exchange Calendar ===")
    from research.volatility_forecasting.simple_pipeline import (
        NonTradingSessionError,
        get_session_close_utc,
    )

    holiday_rejected = False
    try:
        get_session_close_utc("2024-12-25")  # Christmas Day
    except NonTradingSessionError:
        holiday_rejected = True
    assert holiday_rejected

    weekend_rejected = False
    try:
        get_session_close_utc("2024-11-30")  # Saturday
    except NonTradingSessionError:
        weekend_rejected = True
    assert weekend_rejected

    early_close = get_session_close_utc("2024-11-29")  # Black Friday
    assert early_close == pd.Timestamp("2024-11-29 18:00:00")
    print(
        f"  [SUCCESS] Exchange calendar fail-closed verified: Holiday/Weekend rejected, Black Friday={early_close}"
    )

    # Clean up temp dir
    shutil.rmtree(tmp_db_dir, ignore_errors=True)
    print("\n>>> ALL PHASE 5.2 INTEGRITY SIMULATION CHECKS PASSED PERFECTLY! <<<")


if __name__ == "__main__":
    run_simulation()
