"""Persistent forecast ledger service with strict immutability and provenance tracking.

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

SUPPORTED_REPLAY_MODELS = ("rolling_mean", "har_rv", "ewma", "persistence", "garch_11")


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
    record_source: str  # 'live' | 'historical_replay'
    model_version: str
    feature_set_version: str
    code_commit: str
    data_as_of: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForecastLedger:
    """SQLite-backed immutable forecast ledger with strict provenance tracking."""

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
                    record_source TEXT NOT NULL DEFAULT 'live',
                    model_version TEXT NOT NULL DEFAULT 'deployable_v5',
                    feature_set_version TEXT NOT NULL DEFAULT 'deployable_feature_columns_v5',
                    code_commit TEXT NOT NULL DEFAULT 'head',
                    data_as_of TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(forecast_date, ticker, horizon, model_name, record_source)
                )
                """
            )
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(forecast_ledger)").fetchall()
            }
            if "record_source" not in existing_cols:
                conn.execute(
                    "ALTER TABLE forecast_ledger ADD COLUMN record_source TEXT NOT NULL DEFAULT 'live'"
                )
            if "model_version" not in existing_cols:
                conn.execute(
                    "ALTER TABLE forecast_ledger ADD COLUMN model_version TEXT NOT NULL DEFAULT 'deployable_v5'"
                )
            if "feature_set_version" not in existing_cols:
                conn.execute(
                    "ALTER TABLE forecast_ledger ADD COLUMN feature_set_version TEXT NOT NULL DEFAULT 'deployable_feature_columns_v5'"
                )
            if "code_commit" not in existing_cols:
                conn.execute(
                    "ALTER TABLE forecast_ledger ADD COLUMN code_commit TEXT NOT NULL DEFAULT 'head'"
                )
            if "data_as_of" not in existing_cols:
                conn.execute(
                    "ALTER TABLE forecast_ledger ADD COLUMN data_as_of TEXT NOT NULL DEFAULT ''"
                )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_ticker_source ON forecast_ledger(ticker, horizon, record_source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_status_source ON forecast_ledger(status, record_source)"
            )
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
        record_source: str = "live",
        model_version: str = "deployable_v5",
        feature_set_version: str = "deployable_feature_columns_v5",
        code_commit: str = "head",
        data_as_of: str = "",
    ) -> ForecastRecord:
        """Insert a forecast entry into the ledger with strict immutability semantics.

        - If no record exists: inserts a new pending forecast record.
        - If an identical forecast record already exists: returns the existing record idempotently.
        - If a conflicting forecast record exists (different prediction or parameters): raises ValueError.
        - If an existing record has already settled (status='scored'): strictly forbids mutation.
        """
        now_iso = datetime.now(UTC).isoformat()
        f_date_str = str(forecast_date)
        ticker_str = str(ticker).upper()
        h_int = int(horizon)
        m_name_str = str(model_name)
        source_str = str(record_source)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM forecast_ledger
                WHERE forecast_date=? AND ticker=? AND horizon=? AND model_name=? AND record_source=?
                """,
                (f_date_str, ticker_str, h_int, m_name_str, source_str),
            )
            existing = cursor.fetchone()

            if existing is not None:
                existing_pred = float(existing["predicted_volatility"])
                existing_p_origin = float(existing["origin_price"])
                existing_p_lower = float(existing["lower_scenario_price"])
                existing_p_upper = float(existing["upper_scenario_price"])

                is_identical = (
                    abs(existing_pred - float(predicted_volatility)) < 1e-5
                    and abs(existing_p_origin - float(origin_price)) < 1e-4
                    and abs(existing_p_lower - float(lower_scenario_price)) < 1e-4
                    and abs(existing_p_upper - float(upper_scenario_price)) < 1e-4
                )

                if existing["status"] == "scored":
                    if not is_identical:
                        raise ValueError(
                            f"Cannot modify or overwrite a settled/scored forecast for {ticker_str} "
                            f"on {f_date_str} (horizon={h_int}d, model={m_name_str}, source={source_str}). "
                            f"Existing pred={existing_pred:.4f}, incoming pred={float(predicted_volatility):.4f}."
                        )
                    return self._row_to_record(existing)

                if not is_identical:
                    raise ValueError(
                        f"Conflicting forecast already exists for {ticker_str} on {f_date_str} "
                        f"(horizon={h_int}d, model={m_name_str}, source={source_str}). "
                        f"Existing pred={existing_pred:.4f}, incoming pred={float(predicted_volatility):.4f}. "
                        f"Forecast records are immutable."
                    )
                return self._row_to_record(existing)

            conn.execute(
                """
                INSERT INTO forecast_ledger (
                    forecast_date, ticker, horizon, target_date, model_name,
                    predicted_volatility, recent_realized_volatility, origin_price,
                    lower_scenario_price, upper_scenario_price, status,
                    record_source, model_version, feature_set_version,
                    code_commit, data_as_of, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    f_date_str,
                    ticker_str,
                    h_int,
                    str(target_date),
                    m_name_str,
                    float(predicted_volatility),
                    float(recent_realized_volatility),
                    float(origin_price),
                    float(lower_scenario_price),
                    float(upper_scenario_price),
                    source_str,
                    str(model_version),
                    str(feature_set_version),
                    str(code_commit),
                    str(data_as_of) or f_date_str,
                    now_iso,
                ),
            )
            conn.commit()

            cursor = conn.execute(
                """
                SELECT * FROM forecast_ledger
                WHERE forecast_date=? AND ticker=? AND horizon=? AND model_name=? AND record_source=?
                """,
                (f_date_str, ticker_str, h_int, m_name_str, source_str),
            )
            row = cursor.fetchone()
            return self._row_to_record(row)

    def score_pending_forecasts(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
        *,
        record_source: str | None = None,
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
            query = "SELECT * FROM forecast_ledger WHERE ticker=? AND status='pending'"
            params: list[Any] = [str(ticker).upper()]
            if record_source is not None:
                query += " AND record_source=?"
                params.append(str(record_source))

            cursor = conn.execute(query, params)
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
        record_source: str | None = None,
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
        if record_source:
            query += " AND record_source=?"
            params.append(str(record_source))

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
        record_source: str = "live",
    ) -> dict[str, Any]:
        """Compute aggregate empirical track record accuracy metrics for a specific record source."""
        query = "SELECT * FROM forecast_ledger WHERE status='scored' AND record_source=?"
        params: list[Any] = [str(record_source)]

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
                "record_source": record_source,
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
            "record_source": record_source,
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

    def generate_historical_replay_ledger(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
        *,
        horizon: int = 5,
        model_name: str = "rolling_mean",
        lookback_sessions: int = 40,
    ) -> int:
        """Generate historical replay forecast entries from past data using the named model.

        Tagged explicitly with record_source='historical_replay' to ensure historical
        replays are never mixed with genuine live-emitted forecasts.
        Raises ValueError if an unsupported model_name is requested.
        """
        if model_name not in SUPPORTED_REPLAY_MODELS:
            raise ValueError(
                f"Unsupported replay model '{model_name}'. Must be one of {SUPPORTED_REPLAY_MODELS}"
            )

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
            elif model_name == "har_rv":
                ret_1 = abs(float(np.log(close[i] / close[i - 1]))) * math.sqrt(252)
                ret_5 = float(
                    np.std(np.log(close[i - 4 : i + 1] / close[i - 5 : i]), ddof=1) * math.sqrt(252)
                )
                pred_vol = float(max(0.38 * ret_1 + 0.35 * ret_5 + 0.27 * recent_rv, _EPS))
            elif model_name == "ewma":
                lam = 0.94
                weights = (1 - lam) * (lam ** np.arange(len(ret_22) - 1, -1, -1))
                weights /= np.sum(weights)
                var_ewma = float(np.sum(weights * (ret_22**2)))
                pred_vol = float(math.sqrt(max(var_ewma * 252.0, _EPS)))
            elif model_name == "persistence":
                pred_vol = recent_rv
            elif model_name == "garch_11":
                pred_vol = self._fit_garch11_replay(ret_22, horizon=horizon)
            else:
                raise ValueError(f"Unhandled replay model: {model_name}")

            sigma_h = pred_vol * math.sqrt(horizon / 252.0)
            p_lower = float(p_origin * math.exp(-1.64485 * sigma_h))
            p_upper = float(p_origin * math.exp(1.64485 * sigma_h))

            try:
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
                    record_source="historical_replay",
                )
                seeded += 1
            except ValueError:
                pass

        self.score_pending_forecasts(ticker, ohlcv_df, record_source="historical_replay")
        return seeded

    @staticmethod
    def _fit_garch11_replay(returns: np.ndarray, horizon: int) -> float:
        """Causal GARCH(1,1) multi-step annualized volatility estimator."""
        ret = np.asarray(returns, dtype=float).reshape(-1)
        ret = ret - np.mean(ret)
        var = float(np.var(ret, ddof=1)) if len(ret) > 1 else 0.04**2 / 252.0
        if len(ret) < 20 or var <= 1e-8:
            return float(np.std(ret, ddof=1) * math.sqrt(252)) if len(ret) > 1 else 0.20

        omega = 0.05 * var
        alpha = 0.08
        beta = 0.87

        sigma2 = np.zeros_like(ret)
        sigma2[0] = var
        for t in range(1, len(ret)):
            sigma2[t] = omega + alpha * (ret[t - 1] ** 2) + beta * sigma2[t - 1]

        last_sigma2 = float(sigma2[-1])
        last_ret2 = float(ret[-1] ** 2)
        h_1 = omega + alpha * last_ret2 + beta * last_sigma2
        persist = alpha + beta
        uncond = omega / max(1.0 - persist, 1e-4)

        if horizon == 1:
            cum_var = h_1
        else:
            steps = [uncond + (persist ** (k - 1)) * (h_1 - uncond) for k in range(1, horizon + 1)]
            cum_var = sum(steps)

        ann_vol = math.sqrt(max(cum_var * 252.0 / horizon, _EPS))
        return float(ann_vol)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ForecastRecord:
        keys = row.keys()
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
            record_source=row["record_source"] if "record_source" in keys else "live",
            model_version=row["model_version"] if "model_version" in keys else "deployable_v5",
            feature_set_version=(
                row["feature_set_version"]
                if "feature_set_version" in keys
                else "deployable_feature_columns_v5"
            ),
            code_commit=row["code_commit"] if "code_commit" in keys else "head",
            data_as_of=row["data_as_of"] if "data_as_of" in keys else "",
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
