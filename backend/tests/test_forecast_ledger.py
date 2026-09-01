"""Tests for immutable forecast ledger service and provenance tracking."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.forecast_ledger import ForecastLedger


@pytest.fixture
def temp_ledger(tmp_path: Path) -> ForecastLedger:
    db_file = tmp_path / "test_forecast_ledger.db"
    return ForecastLedger(db_file)


def _make_ohlcv(n: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=n, freq="B")
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.015, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "Open": prices * 0.998,
            "High": prices * 1.008,
            "Low": prices * 0.992,
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


def test_record_forecast_and_retrieve_with_provenance(temp_ledger: ForecastLedger) -> None:
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
        record_source="live",
        model_version="deployable_v5",
        feature_set_version="deployable_feature_columns_v5",
        code_commit="abc1234",
        data_as_of="2025-01-10",
    )
    assert rec.ticker == "AAPL"
    assert rec.status == "pending"
    assert rec.record_source == "live"
    assert rec.model_version == "deployable_v5"
    assert rec.code_commit == "abc1234"

    entries = temp_ledger.get_ledger_entries(ticker="AAPL")
    assert len(entries) == 1
    assert entries[0]["ticker"] == "AAPL"
    assert entries[0]["record_source"] == "live"


def test_immutability_identical_retry_is_idempotent(temp_ledger: ForecastLedger) -> None:
    rec1 = temp_ledger.record_forecast(
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
        record_source="live",
    )
    rec2 = temp_ledger.record_forecast(
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
        record_source="live",
    )
    assert rec1.id == rec2.id
    entries = temp_ledger.get_ledger_entries(ticker="AAPL")
    assert len(entries) == 1


def test_immutability_conflicting_replacement_rejected(temp_ledger: ForecastLedger) -> None:
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
        record_source="live",
    )
    with pytest.raises(ValueError, match="Conflicting forecast already exists"):
        temp_ledger.record_forecast(
            forecast_date="2025-01-10",
            ticker="AAPL",
            horizon=5,
            target_date="2025-01-17",
            model_name="rolling_mean",
            predicted_volatility=0.35,  # Conflicting prediction
            recent_realized_volatility=0.22,
            origin_price=150.0,
            lower_scenario_price=140.0,
            upper_scenario_price=160.0,
            record_source="live",
        )


def test_immutability_settled_scored_forecast_cannot_be_mutated(
    temp_ledger: ForecastLedger,
) -> None:
    df = _make_ohlcv(60)
    dates = df.index.strftime("%Y-%m-%d").to_list()
    f_date = dates[10]
    t_date = dates[15]
    p_orig = float(df["Close"].iloc[10])

    temp_ledger.record_forecast(
        forecast_date=f_date,
        ticker="AAPL",
        horizon=5,
        target_date=t_date,
        model_name="rolling_mean",
        predicted_volatility=0.25,
        recent_realized_volatility=0.22,
        origin_price=p_orig,
        lower_scenario_price=p_orig * 0.9,
        upper_scenario_price=p_orig * 1.1,
        record_source="live",
    )
    temp_ledger.score_pending_forecasts("AAPL", df)
    scored = temp_ledger.get_ledger_entries(ticker="AAPL")[0]
    assert scored["status"] == "scored"
    assert scored["actual_realized_volatility"] is not None

    # Conflicting update on scored record must be strictly rejected
    with pytest.raises(ValueError, match="Cannot modify or overwrite a settled/scored forecast"):
        temp_ledger.record_forecast(
            forecast_date=f_date,
            ticker="AAPL",
            horizon=5,
            target_date=t_date,
            model_name="rolling_mean",
            predicted_volatility=0.35,
            recent_realized_volatility=0.22,
            origin_price=p_orig,
            lower_scenario_price=p_orig * 0.9,
            upper_scenario_price=p_orig * 1.1,
            record_source="live",
        )


def test_score_pending_forecasts_and_metrics(temp_ledger: ForecastLedger) -> None:
    df = _make_ohlcv(100)
    dates = df.index.strftime("%Y-%m-%d").to_list()

    temp_ledger.record_forecast(
        forecast_date=dates[20],
        ticker="AAPL",
        horizon=5,
        target_date=dates[25],
        model_name="rolling_mean",
        predicted_volatility=0.24,
        recent_realized_volatility=0.20,
        origin_price=float(df["Close"].iloc[20]),
        lower_scenario_price=float(df["Close"].iloc[20] * 0.95),
        upper_scenario_price=float(df["Close"].iloc[20] * 1.05),
        record_source="live",
    )

    scored = temp_ledger.score_pending_forecasts("AAPL", df)
    assert scored == 1

    entries = temp_ledger.get_ledger_entries(ticker="AAPL")
    assert entries[0]["status"] == "scored"
    assert entries[0]["actual_realized_volatility"] is not None
    assert entries[0]["abs_error"] is not None
    assert entries[0]["qlike_loss"] is not None
    assert entries[0]["qlike_loss"] >= 0.0

    metrics = temp_ledger.get_track_record_metrics(ticker="AAPL", record_source="live")
    assert metrics["total_forecasts"] == 1
    assert metrics["scored_forecasts"] == 1
    assert metrics["mean_mae"] is not None
    assert metrics["mean_qlike"] is not None


def test_generate_historical_replay_ledger_garch11(temp_ledger: ForecastLedger) -> None:
    df = _make_ohlcv(140)
    seeded = temp_ledger.generate_historical_replay_ledger(
        "AAPL",
        df,
        horizon=1,
        model_name="garch_11",
        lookback_sessions=20,
    )
    assert seeded > 0

    entries = temp_ledger.get_ledger_entries(ticker="AAPL", record_source="historical_replay")
    assert len(entries) > 0
    assert all(e["model_name"] == "garch_11" for e in entries)
    assert all(e["record_source"] == "historical_replay" for e in entries)
    assert all(e["predicted_volatility"] > 0.0 for e in entries)


def test_generate_historical_replay_ledger_rejects_unsupported_model(
    temp_ledger: ForecastLedger,
) -> None:
    df = _make_ohlcv(140)
    with pytest.raises(ValueError, match="Unsupported replay model"):
        temp_ledger.generate_historical_replay_ledger(
            "AAPL",
            df,
            horizon=5,
            model_name="random_guessing_model",
        )


def test_live_and_replay_metrics_are_strictly_separated(temp_ledger: ForecastLedger) -> None:
    df = _make_ohlcv(140)
    dates = df.index.strftime("%Y-%m-%d").to_list()

    # Record 1 genuine live forecast
    temp_ledger.record_forecast(
        forecast_date=dates[30],
        ticker="AAPL",
        horizon=5,
        target_date=dates[35],
        model_name="rolling_mean",
        predicted_volatility=0.25,
        recent_realized_volatility=0.22,
        origin_price=float(df["Close"].iloc[30]),
        lower_scenario_price=float(df["Close"].iloc[30] * 0.95),
        upper_scenario_price=float(df["Close"].iloc[30] * 1.05),
        record_source="live",
    )

    # Generate 15 historical replay records
    temp_ledger.generate_historical_replay_ledger(
        "AAPL",
        df,
        horizon=5,
        model_name="rolling_mean",
        lookback_sessions=15,
    )
    temp_ledger.score_pending_forecasts("AAPL", df)

    live_metrics = temp_ledger.get_track_record_metrics(ticker="AAPL", record_source="live")
    replay_metrics = temp_ledger.get_track_record_metrics(
        ticker="AAPL", record_source="historical_replay"
    )

    # Live metrics must contain strictly the 1 live forecast, replay contains the 15 replays
    assert live_metrics["scored_forecasts"] == 1
    assert replay_metrics["scored_forecasts"] >= 10
    assert live_metrics["scored_forecasts"] != replay_metrics["scored_forecasts"]
