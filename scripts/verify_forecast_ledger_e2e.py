"""End-to-end verification of the Phase 5.1 forecast ledger lifecycle."""

import math
import shutil
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

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

    print("\n=== Step 2: Test Idempotent Duplicate Request ===")
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
    )
    assert rec_retry.id == rec.id
    print(f"  [SUCCESS] Duplicate request safely returned identical record id={rec_retry.id}")

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
        )
    except ValueError as e:
        rejected = True
        print(f"  [SUCCESS] Conflicting prediction strictly rejected: {e}")
    assert rejected, "Conflicting forecast should have been rejected!"

    print("\n=== Step 4: Add Historical Replay Records & Check KPI Isolation ===")
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

    print("\n=== Step 5: Future Sessions Arrive -> Settlement & Scoring ===")
    # Construct market data for live settlement
    future_dates = pd.date_range("2026-08-03", periods=15, freq="B")
    live_prices = [p_origin]
    # Simulated 5-day realized volatility: ~23.1%
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
        )
    except ValueError as e:
        mutation_rejected = True
        print(f"  [SUCCESS] Settled forecast mutation rejected: {e}")
    assert mutation_rejected, "Settled forecast mutation should have been rejected!"

    # Clean up temp dir
    shutil.rmtree(tmp_db_dir, ignore_errors=True)
    print("\n>>> ALL PHASE 5.1 INTEGRITY SIMULATION CHECKS PASSED PERFECTLY! <<<")


if __name__ == "__main__":
    run_simulation()
