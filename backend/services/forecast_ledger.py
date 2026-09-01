"""Persistent forecast ledger service.

Records volatility forecasts and scores them against subsequently realized
market volatility once future trading sessions resolve.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "forecast_ledger.db"
_EPS = 1e-6


@dataclass(frozen=True)
class ForecastRecord:
    id: int | None
    forecast_date: str
    ticker: str
    horizon: int
    target_date: str
    model_name: str
    predicted_volatility: float
    recent_realized_volatility: float
    origin_price: float
    lower_scenario_price: float
    upper_scenario_price: float
    actual_realized_volatility: float | None
    forecast_error: float | None
    abs_error: float | None
    qlike_loss: float | None
    status: str  # 'pending' | 'scored'
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForecastLedger:
    """SQLite-backed forecast ledger and evaluation ledger."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or _DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_ledger (
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
                "CREATE INDEX IF NOT EXISTS idx_ledger_ticker ON forecast_ledger(ticker, horizon)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_status ON forecast_ledger(status)")
            conn.commit()

    def record_forecast(
        self,
        *,
        forecast_date: str,
        ticker: str,
        horizon: int,
        target_date: str,
        model_name: str,
        predicted_volatility: float,
        recent_realized_volatility: float,
        origin_price: float,
        lower_scenario_price: float,
        upper_scenario_price: float,
    ) -> ForecastRecord:
        """Insert or update a forecast entry in the ledger."""
        now_iso = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO forecast_ledger (
                    forecast_date, ticker, horizon, target_date, model_name,
                    predicted_volatility, recent_realized_volatility, origin_price,
                    lower_scenario_price, upper_scenario_price, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(forecast_date, ticker, horizon, model_name) DO UPDATE SET
                    predicted_volatility=excluded.predicted_volatility,
                    recent_realized_volatility=excluded.recent_realized_volatility,
                    origin_price=excluded.origin_price,
                    lower_scenario_price=excluded.lower_scenario_price,
                    upper_scenario_price=excluded.upper_scenario_price,
                    created_at=excluded.created_at
                """,
                (
                    str(forecast_date),
                    str(ticker).upper(),
                    int(horizon),
                    str(target_date),
                    str(model_name),
                    float(predicted_volatility),
                    float(recent_realized_volatility),
                    float(origin_price),
                    float(lower_scenario_price),
                    float(upper_scenario_price),
                    now_iso,
                ),
            )
            conn.commit()

            cursor = conn.execute(
                """
                SELECT * FROM forecast_ledger
                WHERE forecast_date=? AND ticker=? AND horizon=? AND model_name=?
                """,
                (str(forecast_date), str(ticker).upper(), int(horizon), str(model_name)),
            )
            row = cursor.fetchone()
            return self._row_to_record(row)

    def score_pending_forecasts(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
        *,
        annualization: float = 252.0,
    ) -> int:
        """Score pending forecasts where future realized market returns are now available."""
        if ohlcv_df.empty or "Close" not in ohlcv_df.columns:
            return 0

        df = ohlcv_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df.index = pd.to_datetime(df["Date"])
            else:
                return 0
        df = df.sort_index()
        close_series = df["Close"].astype(float)
        date_str_list = df.index.strftime("%Y-%m-%d").to_list()

        scored_count = 0
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM forecast_ledger
                WHERE ticker=? AND status='pending'
                """,
                (str(ticker).upper(),),
            )
            rows = cursor.fetchall()

            for row in rows:
                f_date = row["forecast_date"]
                horizon = row["horizon"]
                pred_vol = row["predicted_volatility"]

                if f_date not in date_str_list:
                    continue
                origin_idx = date_str_list.index(f_date)

                if origin_idx + horizon >= len(close_series):
                    continue

                price_slice = close_series.iloc[origin_idx : origin_idx + horizon + 1].to_numpy()
                log_returns = np.log(price_slice[1:] / price_slice[:-1])
                squared_sum = float(np.sum(log_returns**2))
                actual_rv = math.sqrt(max(squared_sum * annualization / horizon, _EPS))

                error = float(pred_vol - actual_rv)
                abs_error = abs(error)

                pred_var = max(pred_vol**2, _EPS)
                actual_var = max(actual_rv**2, _EPS)
                ratio = actual_var / pred_var
                qlike = float(max(ratio - math.log(max(ratio, _EPS)) - 1.0, 0.0))

                actual_target_date = date_str_list[origin_idx + horizon]

                conn.execute(
                    """
                    UPDATE forecast_ledger
                    SET target_date=?,
                        actual_realized_volatility=?,
                        forecast_error=?,
                        abs_error=?,
                        qlike_loss=?,
                        status='scored'
                    WHERE id=?
                    """,
                    (actual_target_date, actual_rv, error, abs_error, qlike, row["id"]),
                )
                scored_count += 1

            conn.commit()

        return scored_count

    def get_ledger_entries(
        self,
        *,
        ticker: str | None = None,
        horizon: int | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent forecast ledger entries."""
        query = "SELECT * FROM forecast_ledger WHERE 1=1"
        params: list[Any] = []

        if ticker:
            query += " AND ticker=?"
            params.append(str(ticker).upper())
        if horizon:
            query += " AND horizon=?"
            params.append(int(horizon))
        if status:
            query += " AND status=?"
            params.append(str(status))

        query += " ORDER BY forecast_date DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_record(row).to_dict() for row in cursor.fetchall()]

    def get_track_record_metrics(
        self,
        *,
        ticker: str | None = None,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate empirical track record accuracy metrics from scored forecasts."""
        query = "SELECT * FROM forecast_ledger WHERE status='scored'"
        params: list[Any] = []

        if ticker:
            query += " AND ticker=?"
            params.append(str(ticker).upper())
        if horizon:
            query += " AND horizon=?"
            params.append(int(horizon))

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return {
                "total_forecasts": 0,
                "scored_forecasts": 0,
                "mean_mae": None,
                "mean_rmse": None,
                "mean_qlike": None,
                "median_qlike": None,
                "direction_accuracy_pct": None,
            }

        abs_errors = [row["abs_error"] for row in rows if row["abs_error"] is not None]
        sq_errors = [
            row["forecast_error"] ** 2 for row in rows if row["forecast_error"] is not None
        ]
        qlikes = [row["qlike_loss"] for row in rows if row["qlike_loss"] is not None]

        direction_hits = 0
        total_direction = 0
        for r in rows:
            if r["actual_realized_volatility"] is not None:
                pred_delta = r["predicted_volatility"] - r["recent_realized_volatility"]
                actual_delta = r["actual_realized_volatility"] - r["recent_realized_volatility"]
                if (pred_delta >= 0 and actual_delta >= 0) or (pred_delta < 0 and actual_delta < 0):
                    direction_hits += 1
                total_direction += 1

        return {
            "total_forecasts": len(rows),
            "scored_forecasts": len(rows),
            "mean_mae": float(np.mean(abs_errors)) if abs_errors else None,
            "mean_rmse": float(np.sqrt(np.mean(sq_errors))) if sq_errors else None,
            "mean_qlike": float(np.mean(qlikes)) if qlikes else None,
            "median_qlike": float(np.median(qlikes)) if qlikes else None,
            "direction_accuracy_pct": (
                float(direction_hits / total_direction * 100.0) if total_direction > 0 else None
            ),
        }

    def seed_historical_test_ledger(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
        *,
        horizon: int = 5,
        model_name: str = "rolling_mean",
        lookback_sessions: int = 40,
    ) -> int:
        """Seed ledger with recent chronological out-of-sample sessions for live track record display."""
        if len(ohlcv_df) < 120:
            return 0

        df = ohlcv_df.sort_index()
        close = df["Close"].to_numpy(dtype=float)
        dates = df.index.strftime("%Y-%m-%d").to_list()
        n = len(df)

        seeded = 0
        start_idx = max(65, n - lookback_sessions - horizon)
        for i in range(start_idx, n - horizon):
            f_date = dates[i]
            t_date = dates[i + horizon]
            p_origin = float(close[i])

            ret_22 = np.log(close[i - 21 : i + 1] / close[i - 22 : i])
            recent_rv = float(np.std(ret_22, ddof=1) * math.sqrt(252))

            if model_name == "rolling_mean":
                ret_60 = np.log(close[i - 59 : i + 1] / close[i - 60 : i])
                pred_vol = float(np.std(ret_60, ddof=1) * math.sqrt(252))
            else:
                pred_vol = recent_rv

            sigma_h = pred_vol * math.sqrt(horizon / 252.0)
            p_lower = float(p_origin * math.exp(-1.64485 * sigma_h))
            p_upper = float(p_origin * math.exp(1.64485 * sigma_h))

            self.record_forecast(
                forecast_date=f_date,
                ticker=ticker,
                horizon=horizon,
                target_date=t_date,
                model_name=model_name,
                predicted_volatility=pred_vol,
                recent_realized_volatility=recent_rv,
                origin_price=p_origin,
                lower_scenario_price=p_lower,
                upper_scenario_price=p_upper,
            )
            seeded += 1

        self.score_pending_forecasts(ticker, ohlcv_df)
        return seeded

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ForecastRecord:
        return ForecastRecord(
            id=row["id"],
            forecast_date=row["forecast_date"],
            ticker=row["ticker"],
            horizon=row["horizon"],
            target_date=row["target_date"],
            model_name=row["model_name"],
            predicted_volatility=float(row["predicted_volatility"]),
            recent_realized_volatility=float(row["recent_realized_volatility"]),
            origin_price=float(row["origin_price"]),
            lower_scenario_price=float(row["lower_scenario_price"]),
            upper_scenario_price=float(row["upper_scenario_price"]),
            actual_realized_volatility=(
                float(row["actual_realized_volatility"])
                if row["actual_realized_volatility"] is not None
                else None
            ),
            forecast_error=(
                float(row["forecast_error"]) if row["forecast_error"] is not None else None
            ),
            abs_error=float(row["abs_error"]) if row["abs_error"] is not None else None,
            qlike_loss=float(row["qlike_loss"]) if row["qlike_loss"] is not None else None,
            status=row["status"],
            created_at=row["created_at"],
        )


_DEFAULT_LEDGER: ForecastLedger | None = None


def get_forecast_ledger(db_path: Path | str | None = None) -> ForecastLedger:
    global _DEFAULT_LEDGER
    if db_path is not None:
        return ForecastLedger(db_path)
    if _DEFAULT_LEDGER is None:
        _DEFAULT_LEDGER = ForecastLedger()
    return _DEFAULT_LEDGER
