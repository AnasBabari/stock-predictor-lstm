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


def test_live_ledger_rejects_unsupported_and_fractional_horizons(
    temp_ledger: ForecastLedger,
) -> None:
    params = {
        "forecast_date": "2025-01-10",
        "ticker": "AAPL",
        "target_date": "2025-01-17",
        "model_name": "rolling_mean",
        "predicted_volatility": 0.25,
        "recent_realized_volatility": 0.22,
        "origin_price": 150.0,
        "lower_scenario_price": 140.0,
        "upper_scenario_price": 160.0,
        "record_source": "live",
    }
    with pytest.raises(ValueError, match="horizon must be one of"):
        temp_ledger.record_forecast(**params, horizon=7)
    with pytest.raises(ValueError, match="horizon must be one of"):
        temp_ledger.record_forecast(**params, horizon=5.5)


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
    with pytest.raises(ValueError, match="Conflicting forecast fingerprint"):
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


def test_schema_migration_from_pre_phase51_sqlite_database(tmp_path: Path) -> None:
    import sqlite3

    old_db_file = tmp_path / "legacy_phase5_ledger.db"

    # 1. Create exact old Phase-5 schema without fingerprint and without record_source in UNIQUE constraint
    conn = sqlite3.connect(str(old_db_file))
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
            '2026-08-01', 'AAPL', 5, '2026-08-08', 'rolling_mean',
            0.245, 0.221, 150.0, 140.0, 160.0, 'scored', '2026-08-01T20:00:00Z'
        )
        """
    )
    conn.commit()
    conn.close()

    # 2. Open with modern ForecastLedger -> automatic migration
    migrated_ledger = ForecastLedger(old_db_file)

    # 3. Check old record survived with valid defaults
    entries = migrated_ledger.get_ledger_entries(ticker="AAPL")
    assert len(entries) == 1
    assert entries[0]["forecast_date"] == "2026-08-01"
    assert entries[0]["record_source"] == "live"
    assert entries[0]["model_version"] == "deployable_v5"
    assert len(entries[0]["forecast_fingerprint"]) == 64

    # 4. Prove live and historical_replay records can now COEXIST under the same (date, ticker, horizon, model)
    migrated_ledger.record_forecast(
        forecast_date="2026-08-01",
        ticker="AAPL",
        horizon=5,
        target_date="2026-08-08",
        model_name="rolling_mean",
        predicted_volatility=0.245,
        recent_realized_volatility=0.221,
        origin_price=150.0,
        lower_scenario_price=140.0,
        upper_scenario_price=160.0,
        record_source="historical_replay",
        model_version="deployable_v5",
        feature_set_version="deployable_feature_columns_v5",
        code_commit="migrated123",
        data_as_of="2026-08-01",
    )

    all_entries = migrated_ledger.get_ledger_entries(ticker="AAPL")
    assert len(all_entries) == 2
    sources = {e["record_source"] for e in all_entries}
    assert sources == {"live", "historical_replay"}


def test_provenance_fingerprint_conflicts_rejected(temp_ledger: ForecastLedger) -> None:
    base_params = {
        "forecast_date": "2026-08-10",
        "ticker": "MSFT",
        "horizon": 5,
        "target_date": "2026-08-17",
        "model_name": "rolling_mean",
        "predicted_volatility": 0.24,
        "recent_realized_volatility": 0.21,
        "origin_price": 400.0,
        "lower_scenario_price": 380.0,
        "upper_scenario_price": 420.0,
        "record_source": "live",
        "model_version": "deployable_v5",
        "feature_set_version": "deployable_feature_columns_v5",
        "code_commit": "commit_1",
        "data_as_of": "2026-08-10",
    }

    # Initial record
    rec1 = temp_ledger.record_forecast(**base_params)
    assert rec1.id is not None

    # Idempotent retry with identical provenance
    rec_retry = temp_ledger.record_forecast(**base_params)
    assert rec_retry.id == rec1.id

    # Conflict on code_commit
    with pytest.raises(ValueError, match="Conflicting forecast fingerprint"):
        conflict_commit = dict(base_params)
        conflict_commit["code_commit"] = "commit_2"
        temp_ledger.record_forecast(**conflict_commit)

    # Conflict on data_as_of
    with pytest.raises(ValueError, match="Conflicting forecast fingerprint"):
        conflict_as_of = dict(base_params)
        conflict_as_of["data_as_of"] = "2026-08-11"
        temp_ledger.record_forecast(**conflict_as_of)

    # Conflict on recent_realized_volatility
    with pytest.raises(ValueError, match="Conflicting forecast fingerprint"):
        conflict_rv = dict(base_params)
        conflict_rv["recent_realized_volatility"] = 0.25
        temp_ledger.record_forecast(**conflict_rv)

    # Conflict on model_version
    with pytest.raises(ValueError, match="Conflicting forecast fingerprint"):
        conflict_mver = dict(base_params)
        conflict_mver["model_version"] = "deployable_v6"
        temp_ledger.record_forecast(**conflict_mver)


def test_record_source_validation(temp_ledger: ForecastLedger) -> None:
    with pytest.raises(ValueError, match="record_source must be one of"):
        temp_ledger.record_forecast(
            forecast_date="2026-08-10",
            ticker="MSFT",
            horizon=5,
            target_date="2026-08-17",
            model_name="rolling_mean",
            predicted_volatility=0.24,
            recent_realized_volatility=0.21,
            origin_price=400.0,
            lower_scenario_price=380.0,
            upper_scenario_price=420.0,
            record_source="invalid_source_type",
        )


def test_canonical_garch_mle_parity(temp_ledger: ForecastLedger) -> None:
    from research.volatility_forecasting.simple_pipeline import fit_garch11_mle_from_returns

    df = _make_ohlcv(140)
    close = df["Close"].to_numpy(dtype=float)
    dates = df.index.strftime("%Y-%m-%d").to_list()
    horizon = 5

    # Target session for comparison
    target_idx = len(df) - horizon - 2
    f_date = dates[target_idx]

    ret_22 = np.log(close[target_idx - 21 : target_idx + 1] / close[target_idx - 22 : target_idx])
    expected_mle_vol = fit_garch11_mle_from_returns(ret_22, horizon=horizon)

    temp_ledger.generate_historical_replay_ledger(
        "AAPL",
        df,
        horizon=horizon,
        model_name="garch_11",
        lookback_sessions=10,
    )

    entries = temp_ledger.get_ledger_entries(ticker="AAPL", record_source="historical_replay")
    matching = [
        e for e in entries if e["forecast_date"] == f_date and e["model_name"] == "garch_11"
    ]
    assert len(matching) == 1
    assert abs(matching[0]["predicted_volatility"] - expected_mle_vol) < 1e-6
