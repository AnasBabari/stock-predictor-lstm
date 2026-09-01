from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.forecast_ledger import ForecastLedger


@pytest.fixture
def temp_ledger(tmp_path: Path) -> ForecastLedger:
    db_file = tmp_path / "test_ledger.db"
    return ForecastLedger(db_file)


def _sample_ohlcv(length: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-02", periods=length, freq="B")
    returns = rng.normal(0.0005, 0.015, size=length)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Open": prices * (1.0 - 0.002),
            "High": prices * (1.0 + 0.01),
            "Low": prices * (1.0 - 0.01),
            "Close": prices,
            "Volume": np.full(length, 1000000),
        },
        index=dates,
    )


def test_record_forecast_and_retrieve(temp_ledger: ForecastLedger) -> None:
    rec = temp_ledger.record_forecast(
        forecast_date="2025-01-10",
        ticker="AAPL",
        horizon=5,
        target_date="2025-01-17",
        model_name="rolling_mean",
        predicted_volatility=0.25,
        recent_realized_volatility=0.22,
        origin_price=150.0,
        lower_scenario_price=140.0,
        upper_scenario_price=160.0,
    )
    assert rec.id is not None
    assert rec.ticker == "AAPL"
    assert rec.horizon == 5
    assert rec.status == "pending"
    assert rec.actual_realized_volatility is None

    entries = temp_ledger.get_ledger_entries(ticker="AAPL")
    assert len(entries) == 1
    assert entries[0]["ticker"] == "AAPL"
    assert entries[0]["status"] == "pending"


def test_record_forecast_upsert_on_conflict(temp_ledger: ForecastLedger) -> None:
    temp_ledger.record_forecast(
        forecast_date="2025-01-10",
        ticker="AAPL",
        horizon=5,
        target_date="2025-01-17",
        model_name="rolling_mean",
        predicted_volatility=0.25,
        recent_realized_volatility=0.22,
        origin_price=150.0,
        lower_scenario_price=140.0,
        upper_scenario_price=160.0,
    )
    temp_ledger.record_forecast(
        forecast_date="2025-01-10",
        ticker="AAPL",
        horizon=5,
        target_date="2025-01-17",
        model_name="rolling_mean",
        predicted_volatility=0.28,
        recent_realized_volatility=0.23,
        origin_price=151.0,
        lower_scenario_price=139.0,
        upper_scenario_price=163.0,
    )
    entries = temp_ledger.get_ledger_entries(ticker="AAPL")
    assert len(entries) == 1
    assert entries[0]["predicted_volatility"] == pytest.approx(0.28)


def test_score_pending_forecasts_and_metrics(temp_ledger: ForecastLedger) -> None:
    df = _sample_ohlcv(50)
    origin_date = df.index[10].strftime("%Y-%m-%d")
    temp_ledger.record_forecast(
        forecast_date=origin_date,
        ticker="AAPL",
        horizon=5,
        target_date="TBD",
        model_name="rolling_mean",
        predicted_volatility=0.24,
        recent_realized_volatility=0.22,
        origin_price=float(df["Close"].iloc[10]),
        lower_scenario_price=90.0,
        upper_scenario_price=110.0,
    )
    scored = temp_ledger.score_pending_forecasts("AAPL", df)
    assert scored == 1
    entries = temp_ledger.get_ledger_entries(ticker="AAPL", status="scored")
    assert len(entries) == 1
    assert entries[0]["status"] == "scored"
    assert entries[0]["actual_realized_volatility"] is not None
    assert entries[0]["actual_realized_volatility"] > 0
    assert entries[0]["forecast_error"] is not None
    assert entries[0]["qlike_loss"] is not None
    assert entries[0]["qlike_loss"] >= 0.0

    metrics = temp_ledger.get_track_record_metrics(ticker="AAPL", horizon=5)
    assert metrics["total_forecasts"] == 1
    assert metrics["mean_mae"] == pytest.approx(entries[0]["abs_error"])
    assert metrics["mean_qlike"] == pytest.approx(entries[0]["qlike_loss"])


def test_seed_historical_test_ledger(temp_ledger: ForecastLedger) -> None:
    df = _sample_ohlcv(140)
    seeded = temp_ledger.seed_historical_test_ledger("MSFT", df, horizon=5, lookback_sessions=20)
    assert seeded > 0
    entries = temp_ledger.get_ledger_entries(ticker="MSFT")
    assert len(entries) == seeded
    assert all(e["status"] == "scored" for e in entries)
    metrics = temp_ledger.get_track_record_metrics(ticker="MSFT", horizon=5)
    assert metrics["scored_forecasts"] == seeded
    assert metrics["mean_mae"] is not None
    assert metrics["mean_qlike"] is not None
