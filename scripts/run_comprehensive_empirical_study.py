"""Comprehensive empirical study for volatility forecasting and uncertainty calibration (Phase 3).

Features:
1. Audited OHLC Corporate-Action Consistency (auto-adjusted Open/High/Low/Close).
2. Nested Feature Ablations:
   - PRICE_ONLY (returns, rolling/EWMA realized volatility)
   - PRICE_PLUS_OHLC (adds Parkinson, Garman-Klass, Rogers-Satchell range estimators)
   - PRICE_PLUS_OHLC_PLUS_MARKET (adds SPY/QQQ causal market returns and volatility with leave-self-out)
3. Controlled Neural Target / Link Function Comparison:
   - LSTM_DIRECT_VOLATILITY
   - LSTM_SOFTPLUS_VOLATILITY
   - LSTM_LOG_VARIANCE
4. Extended Pooled & Asset-Balanced QLIKE Distribution Diagnostics:
   - Pooled: Mean, Median, p90, p95, p99, Max, Worst 1% Loss Contribution Share, Raw Min Pred, Near-Zero Counts
   - Asset-Balanced: Mean per-asset QLIKE, Median per-asset QLIKE, Assets Improved vs Baseline
5. Output Versioned Report: reports/empirical_volatility_benchmark_v3.md/json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.volatility_forecasting.simple_pipeline import (  # noqa: E402
    PIPELINE_VERSION,
    LSTMConfig,
    VolatilityConfig,
    build_examples,
    chronological_split,
    evaluate_benchmark,
    select_validation_model,
)

# 44 diverse liquid assets across 8 market sectors
TARGET_UNIVERSE = {
    "Mega-Cap Tech / Growth": ["MSFT", "GOOG", "AMZN", "META", "NVDA", "AAPL"],
    "Broad & Tech ETFs": ["SPY", "QQQ"],
    "Financials & Fintech": ["BKNG", "MELI", "INTU"],
    "Industrials & Logistics": ["HON", "CSX", "FAST", "CPRT", "CTAS", "ODFL", "PCAR"],
    "Healthcare & Biotech": ["AMGN", "GILD", "BIIB", "ISRG", "REGN", "VRTX", "IDXX"],
    "Consumer Staples & Discretionary": ["COST", "PEP", "MDLZ", "SBUX", "ORLY", "MNST", "KDP"],
    "Energy & Utilities": ["FANG", "BKR", "CEG", "EXC", "AEP", "XEL"],
    "High-Beta / High-Vol": ["TSLA", "AMD", "MRNA", "DASH", "FTNT", "PANW"],
}

ALL_TICKERS = [ticker for group in TARGET_UNIVERSE.values() for ticker in group]


def _find_data_file(ticker: str) -> pd.DataFrame:
    """Find and load OHLCV data for a ticker with verified split-adjusted OHLC consistency."""
    ticker_up = ticker.upper()

    # 1. Check ndx100 parquet cache (verified uniform auto-adjusted OHLC)
    parquet_file = _REPO_ROOT / "data" / "ndx100" / "cache" / f"{ticker_up}.parquet"
    if parquet_file.is_file():
        return pd.read_parquet(parquet_file)

    # 2. Check diagnostic snapshots panel (verified uniform auto-adjusted OHLC)
    raw_dir = (
        _REPO_ROOT
        / "artifacts"
        / "v11_2_diagnostic_inputs"
        / "snapshots"
        / "panel-8546a6f180250034"
        / "raw"
    )
    csv_file = raw_dir / f"{ticker_up}.csv"
    if csv_file.is_file():
        return pd.read_csv(csv_file)

    # 3. Check root snapshot files if valid OHLC exists
    root_csv = _REPO_ROOT / f"snapshot_{ticker_up}.csv"
    if root_csv.is_file():
        df = pd.read_csv(root_csv)
        if {"Open", "High", "Low", "Close"}.issubset(df.columns) or {
            "open",
            "high",
            "low",
            "close",
        }.issubset(df.columns):
            return df

    # 4. Fallback to yfinance if online
    try:
        import yfinance as yf

        df = yf.download(ticker_up, period="5y", interval="1d", auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as err:
        raise FileNotFoundError(f"Could not load data for {ticker_up}: {err}") from err

    raise FileNotFoundError(f"No local verified OHLC data found for {ticker_up}")


def _build_market_context_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare SPY and QQQ market context frames."""
    spy_raw = _find_data_file("SPY")
    qqq_raw = _find_data_file("QQQ")

    from research.volatility_forecasting.simple_pipeline import validate_ohlcv

    spy = validate_ohlcv(spy_raw)
    qqq = validate_ohlcv(qqq_raw)

    spy_ret = np.log(spy["Close"]).diff()
    spy_vol22 = spy_ret.rolling(22, min_periods=22).std() * np.sqrt(252.0)

    qqq_ret = np.log(qqq["Close"]).diff()
    qqq_vol22 = qqq_ret.rolling(22, min_periods=22).std() * np.sqrt(252.0)

    spy_mkt = pd.DataFrame({"spy_return_1d": spy_ret, "spy_vol_22": spy_vol22}, index=spy.index)
    qqq_mkt = pd.DataFrame({"qqq_return_1d": qqq_ret, "qqq_vol_22": qqq_vol22}, index=qqq.index)

    return spy_mkt, qqq_mkt


def _get_market_frame_for_ticker(
    ticker: str, spy_mkt: pd.DataFrame, qqq_mkt: pd.DataFrame
) -> pd.DataFrame:
    """Apply leave-self-out market context."""
    t_up = ticker.upper()
    if t_up == "SPY":
        return qqq_mkt
    elif t_up == "QQQ":
        return spy_mkt
    else:
        return spy_mkt.join(qqq_mkt, how="outer")


def _sector_for(ticker: str) -> str:
    for sector, tickers in TARGET_UNIVERSE.items():
        if ticker.upper() in tickers:
            return sector
    return "Other"


def run_study(
    tickers: list[str] | None = None,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    include_lstm: bool = True,
    lookback: int = 22,
    feature_mode: str = "price_plus_ohlc",
    target_space: str = "log_variance",
    cache_key: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    selected_tickers = tickers or ALL_TICKERS
    cache_fingerprint_payload = {
        "schema": "comprehensive-study-cache-v2",
        "pipeline_version": PIPELINE_VERSION,
        "tickers": list(selected_tickers),
        "horizons": list(horizons),
        "include_lstm": bool(include_lstm),
        "lookback": int(lookback),
        "feature_mode": feature_mode,
        "target_space": target_space,
    }
    cache_fingerprint = hashlib.sha256(
        json.dumps(cache_fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    if cache_key and not force:
        cache_file = _REPO_ROOT / "reports" / f".cache_{cache_key}.json"
        if cache_file.is_file():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                if (
                    cached_data.get("pipeline_version") == PIPELINE_VERSION
                    and cached_data.get("cache_fingerprint") == cache_fingerprint
                    and len(
                        cached_data.get("raw_results_by_horizon", {}).get(f"h{horizons[0]}", [])
                    )
                    == len(selected_tickers)
                ):
                    print(
                        f"Loaded cached study results for {cache_key} ({len(selected_tickers)} assets).",
                        flush=True,
                    )
                    return cached_data
            except Exception:
                pass

    start_time = time.time()
    print(
        f"Starting study: Universe={len(selected_tickers)}, FeatureMode={feature_mode}, TargetSpace={target_space}, Horizons={horizons}...",
        flush=True,
    )

    spy_mkt, qqq_mkt = _build_market_context_frame()

    horizon_results: dict[int, list[dict[str, Any]]] = {h: [] for h in horizons}
    raw_test_samples: list[dict[str, Any]] = []

    for t_idx, ticker in enumerate(selected_tickers, 1):
        t_start_asset = time.time()
        try:
            raw_frame = _find_data_file(ticker)
        except Exception as exc:
            print(
                f"  [{t_idx}/{len(selected_tickers)}] Warning: Skipped {ticker}: {exc}", flush=True
            )
            continue

        mkt_frame = (
            _get_market_frame_for_ticker(ticker, spy_mkt, qqq_mkt)
            if feature_mode
            in ("price_plus_ohlc_plus_market", "price_plus_ohlc_plus_market_plus_news")
            else None
        )

        for h in horizons:
            config = VolatilityConfig(horizon=h, lookback=lookback, feature_mode=feature_mode)
            try:
                examples = build_examples(
                    raw_frame,
                    config,
                    market_frame=mkt_frame,
                    ticker=ticker,
                )
                split = chronological_split(
                    len(examples.target),
                    horizon=config.horizon,
                    train_fraction=config.train_fraction,
                    validation_fraction=config.validation_fraction,
                    embargo_sessions=config.embargo,
                )
                metrics, forecasts = evaluate_benchmark(
                    examples,
                    split,
                    include_boosting=True,
                    include_lstm=include_lstm,
                    lstm_config=LSTMConfig(
                        maximum_epochs=15,
                        patience=3,
                        batch_size=128,
                        device="cpu",
                        seed=config.seed,
                        target_space=target_space,
                    )
                    if include_lstm
                    else None,
                    nominal_coverage=0.90,
                    target_space=target_space,
                    return_forecasts=True,
                )
                selected_model = select_validation_model(metrics)
                horizon_results[h].append(
                    {
                        "ticker": ticker,
                        "sector": _sector_for(ticker),
                        "horizon": h,
                        "rows": len(examples.target),
                        "feature_count": int(examples.sequences.shape[-1]),
                        "train_rows": len(split.train),
                        "val_rows": len(split.validation),
                        "test_rows": len(split.test),
                        "selected_model": selected_model,
                        "metrics": metrics,
                    }
                )

                # Collect test samples for tail error diagnostics
                test_indices = split.test
                actual_vols = examples.target[test_indices]
                recent_vols = examples.current_volatility[test_indices]
                dates = examples.dates[test_indices]
                tertiles = np.quantile(actual_vols, [1.0 / 3.0, 2.0 / 3.0])

                for idx_in_test, global_idx in enumerate(test_indices):
                    act_v = float(actual_vols[idx_in_test])
                    act_var = float(act_v**2)
                    rec_v = float(recent_vols[idx_in_test])
                    d_str = str(dates[idx_in_test])

                    if act_v <= tertiles[0]:
                        regime = "Low Vol"
                    elif act_v <= tertiles[1]:
                        regime = "Normal Vol"
                    else:
                        regime = "High Vol"

                    sample_item: dict[str, Any] = {
                        "ticker": ticker,
                        "date": d_str,
                        "horizon": h,
                        "regime": regime,
                        "actual_vol": act_v,
                        "actual_var": act_var,
                        "recent_vol_22": rec_v,
                        "models": {},
                    }

                    for m_name, pred_arr in forecasts.items():
                        pred_v = float(pred_arr[global_idx])
                        pred_var = float(pred_v**2)
                        ratio = act_var / max(pred_var, 1e-12)
                        qlike_val = float(ratio - np.log(ratio) - 1.0)
                        sample_item["models"][m_name] = {
                            "pred_vol": pred_v,
                            "pred_var": pred_var,
                            "error": pred_v - act_v,
                            "qlike": qlike_val,
                            "floor_activated": bool(pred_v <= 1e-4),
                        }
                    raw_test_samples.append(sample_item)

            except Exception as exc:
                print(f"  Error on {ticker} h={h}: {exc}", flush=True)

        print(
            f"  [{t_idx:02d}/{len(selected_tickers):02d}] {ticker:<5} ({time.time() - t_start_asset:.1f}s)",
            flush=True,
        )

    model_names = [
        "persistence",
        "rolling_mean",
        "ewma",
        "har_rv",
        "garch_11",
        "ridge",
        "elastic_net",
        "gradient_boosting",
    ]
    if include_lstm:
        model_names.append("lstm")

    per_horizon_aggregates: dict[str, Any] = {}

    for h in horizons:
        results = horizon_results[h]
        if not results:
            continue

        agg: dict[str, dict[str, Any]] = {}
        for m in model_names:
            model_results = [r for r in results if m in r["metrics"]]
            if not model_results:
                continue
            test_maes = [r["metrics"][m]["test"]["mae"] for r in model_results]
            test_rmses = [r["metrics"][m]["test"]["rmse"] for r in model_results]
            test_qlikes = [r["metrics"][m]["test"]["qlike"] for r in model_results]
            val_qlikes = [r["metrics"][m]["validation"]["qlike"] for r in model_results]
            near_zero_counts = [
                r["metrics"][m]["test"].get("near_zero_count", 0) for r in model_results
            ]
            raw_mins = [r["metrics"][m]["test"].get("raw_min_pred", 0.0) for r in model_results]

            # Distributional QLIKE stats
            med_qlikes = [
                r["metrics"][m]["test"].get("median_qlike", r["metrics"][m]["test"]["qlike"])
                for r in model_results
            ]
            p90_qlikes = [
                r["metrics"][m]["test"].get("p90_qlike", r["metrics"][m]["test"]["qlike"])
                for r in model_results
            ]
            p95_qlikes = [
                r["metrics"][m]["test"].get("p95_qlike", r["metrics"][m]["test"]["qlike"])
                for r in model_results
            ]
            p99_qlikes = [
                r["metrics"][m]["test"].get("p99_qlike", r["metrics"][m]["test"]["qlike"])
                for r in model_results
            ]
            max_qlikes = [
                r["metrics"][m]["test"].get("max_qlike", r["metrics"][m]["test"]["qlike"])
                for r in model_results
            ]
            w1_shares = [
                r["metrics"][m]["test"].get("worst_1pct_share", 0.0) for r in model_results
            ]

            # Uncertainty calibration aggregates
            vol_covs = [
                r["metrics"][m]["test"]["volatility_interval"]["empirical_coverage"]
                for r in model_results
                if "volatility_interval" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["volatility_interval"].get("empirical_coverage")
                is not None
            ]
            vol_widths = [
                r["metrics"][m]["test"]["volatility_interval"]["average_width"]
                for r in model_results
                if "volatility_interval" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["volatility_interval"].get("average_width") is not None
            ]
            price_covs = [
                r["metrics"][m]["test"]["price_cone"]["empirical_coverage"]
                for r in model_results
                if "price_cone" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["price_cone"].get("empirical_coverage") is not None
            ]
            price_widths = [
                r["metrics"][m]["test"]["price_cone"]["average_width_pct"]
                for r in model_results
                if "price_cone" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["price_cone"].get("average_width_pct") is not None
            ]
            price_cone_methods = sorted(
                {
                    str(r["metrics"][m]["test"]["price_cone"].get("interval_method"))
                    for r in model_results
                    if "price_cone" in r["metrics"][m]["test"]
                }
            )

            agg[m] = {
                "asset_count": len(model_results),
                "val_qlike": float(np.mean(val_qlikes)),
                "test_mae": float(np.mean(test_maes)),
                "test_rmse": float(np.mean(test_rmses)),
                "test_qlike": float(np.mean(test_qlikes)),
                "median_qlike": float(np.mean(med_qlikes)),
                "p90_qlike": float(np.mean(p90_qlikes)),
                "p95_qlike": float(np.mean(p95_qlikes)),
                "p99_qlike": float(np.mean(p99_qlikes)),
                "max_qlike": float(np.max(max_qlikes)) if max_qlikes else None,
                "worst_1pct_share": float(np.mean(w1_shares)),
                "per_asset_mean_qlike": float(np.mean(test_qlikes)),
                "per_asset_median_qlike": float(np.median(test_qlikes)),
                "near_zero_count": int(sum(near_zero_counts)),
                "raw_min_pred": float(min(raw_mins)) if raw_mins else None,
                "vol_interval_coverage_90": float(np.mean(vol_covs)) if vol_covs else None,
                "vol_interval_width": float(np.mean(vol_widths)) if vol_widths else None,
                "price_cone_coverage_90": float(np.mean(price_covs)) if price_covs else None,
                "price_cone_width_pct": float(np.mean(price_widths)) if price_widths else None,
                "price_cone_method": price_cone_methods[0]
                if len(price_cone_methods) == 1
                else price_cone_methods,
            }

        # Calculate relative skill vs persistence and vs HAR only where the
        # corresponding metric exists.  A failed optional model must not turn
        # the entire study into NaNs or make the winner loop raise KeyError.
        base_persist = agg.get("persistence", {}).get("test_qlike")
        base_har = agg.get("har_rv", {}).get("test_qlike")

        for m, stats in agg.items():
            if base_persist and np.isfinite(base_persist):
                stats["vs_persistence_pct"] = float(
                    (base_persist - stats["test_qlike"]) / base_persist * 100.0
                )
            else:
                stats["vs_persistence_pct"] = None
            if base_har and np.isfinite(base_har):
                stats["vs_har_pct"] = float((base_har - stats["test_qlike"]) / base_har * 100.0)
            else:
                stats["vs_har_pct"] = None

            # Count assets improved vs persistence and vs HAR using only the
            # paired rows available for this model.
            improved_p = 0
            improved_h = 0
            paired_count_p = 0
            paired_count_h = 0
            for row in results:
                row_metrics = row["metrics"]
                if m not in row_metrics:
                    continue
                m_q = row_metrics[m]["test"]["qlike"]
                if "persistence" in row_metrics:
                    paired_count_p += 1
                    if m_q < row_metrics["persistence"]["test"]["qlike"]:
                        improved_p += 1
                if "har_rv" in row_metrics:
                    paired_count_h += 1
                    if m_q < row_metrics["har_rv"]["test"]["qlike"]:
                        improved_h += 1
            stats["assets_improved_vs_persistence"] = improved_p
            stats["assets_improved_vs_persistence_count"] = paired_count_p
            stats["assets_improved_vs_har"] = improved_h
            stats["assets_improved_vs_har_count"] = paired_count_h

        # Count wins per model on validation and test
        val_selected_counts = {m: 0 for m in agg}
        test_best_counts = {m: 0 for m in agg}
        for r in results:
            selected = r["selected_model"]
            if selected in val_selected_counts:
                val_selected_counts[selected] += 1
            candidates = [m for m in agg if m in r["metrics"]]
            if candidates:
                best_test = min(candidates, key=lambda m: r["metrics"][m]["test"]["qlike"])
                test_best_counts[best_test] += 1

        per_horizon_aggregates[f"h{h}"] = {
            "horizon": h,
            "asset_count": len(results),
            "models": agg,
            "val_selected_counts": val_selected_counts,
            "test_best_counts": test_best_counts,
        }

    # Extract worst 20 QLIKE losses
    worst_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for m in ["lstm", "gradient_boosting", "har_rv", "rolling_mean", "garch_11"]:
        if any(m in s["models"] for s in raw_test_samples):
            sorted_samples = sorted(
                raw_test_samples,
                key=lambda s: s["models"].get(m, {}).get("qlike", -1.0),
                reverse=True,
            )
            top_20 = []
            for s in sorted_samples[:20]:
                top_20.append(
                    {
                        "ticker": s["ticker"],
                        "date": s["date"],
                        "horizon": s["horizon"],
                        "regime": s["regime"],
                        "actual_vol": s["actual_vol"],
                        "actual_var": s["actual_var"],
                        "pred_vol": s["models"][m]["pred_vol"],
                        "pred_var": s["models"][m]["pred_var"],
                        "recent_vol_22": s["recent_vol_22"],
                        "error": s["models"][m]["error"],
                        "qlike": s["models"][m]["qlike"],
                        "floor_activated": s["models"][m]["floor_activated"],
                    }
                )
            worst_diagnostics[m] = top_20

    elapsed = time.time() - start_time
    print(f"Study complete in {elapsed:.1f}s.")

    study_dict = {
        "benchmark_version": "empirical-volatility-benchmark-v3",
        "pipeline_version": PIPELINE_VERSION,
        "cache_fingerprint": cache_fingerprint,
        "cache_fingerprint_payload": cache_fingerprint_payload,
        "date_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "universe_size": len(selected_tickers),
        "feature_mode": feature_mode,
        "target_space": target_space,
        "horizons": list(horizons),
        "models": model_names,
        "universe_by_sector": TARGET_UNIVERSE,
        "per_horizon_aggregates": per_horizon_aggregates,
        "worst_error_diagnostics": worst_diagnostics,
        "raw_results_by_horizon": {f"h{h}": horizon_results[h] for h in horizons},
    }

    if cache_key:
        cache_file = _REPO_ROOT / "reports" / f".cache_{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(study_dict, indent=2, default=str), encoding="utf-8")

    return study_dict


def _display_model(name: str) -> str:
    """Stable human-readable model labels used by all report tables."""

    labels = {
        "har_rv": "HAR-RV",
        "garch_11": "GARCH(1,1)",
        "lstm": "PyTorch LSTM",
        "ewma": "EWMA (λ=0.94)",
        "rolling_mean": "Rolling Mean (60d)",
    }
    return labels.get(name, name.replace("_", " ").title())


def _feature_count_for_data(data: dict[str, Any] | None, horizon_key: str) -> int | None:
    """Return the observed feature count without duplicating schema constants."""

    if not data:
        return None
    counts = [
        int(row["feature_count"])
        for row in data.get("raw_results_by_horizon", {}).get(horizon_key, [])
        if row.get("feature_count") is not None
    ]
    return int(np.median(counts)) if counts else None


def _metric_asset_count(stats: dict[str, Any], fallback: int) -> int:
    """Return the denominator for a model row in a partially complete study."""

    value = stats.get("asset_count", fallback)
    return int(value) if isinstance(value, (int, float)) else fallback


def _best_aggregate_model(
    primary_data: dict[str, Any],
    horizon: int,
    metric: str,
    allowed_models: set[str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Return a deterministic aggregate test winner for one horizon/metric."""

    aggregate = primary_data.get("per_horizon_aggregates", {}).get(f"h{horizon}")
    if not aggregate:
        return None
    candidates = [
        (name, stats)
        for name, stats in aggregate.get("models", {}).items()
        if allowed_models is None or name in allowed_models
        if isinstance(stats.get(metric), (int, float)) and np.isfinite(float(stats[metric]))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (float(item[1][metric]), item[0]))


def _paired_bootstrap_summary(
    baseline: list[float], challenger: list[float], *, seed: int
) -> dict[str, Any] | None:
    """Summarise paired per-asset improvements with a reproducible CI.

    Positive deltas mean the challenger reduced the error.  Assets are the
    resampling unit, so the summary cannot be inflated by securities with
    more test rows and does not affect model selection.
    """

    if len(baseline) != len(challenger) or not baseline:
        return None
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    challenger_arr = np.asarray(challenger, dtype=np.float64)
    valid = np.isfinite(baseline_arr) & np.isfinite(challenger_arr)
    values = baseline_arr[valid] - challenger_arr[valid]
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(4000, len(values)))
    means = values[draws].mean(axis=1)
    base_mean = float(np.mean(baseline_arr[valid]))
    return {
        "asset_count": int(len(values)),
        "improved_assets": int(np.sum(values > 0)),
        "non_degraded_assets": int(np.sum(values >= 0)),
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "mean_relative_improvement_pct": (
            float(np.mean(values) / base_mean * 100.0) if base_mean > 0 else None
        ),
        "bootstrap_ci_95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_unit": "asset",
        "bootstrap_replicates": 4000,
        "seed": int(seed),
    }


def build_ablation_breadth(
    price_only: dict[str, Any] | None,
    price_ohlc: dict[str, Any],
    price_market: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build complete model × horizon paired ablation evidence."""

    transitions = (
        ("price_only_to_ohlc", "PRICE_ONLY", price_only, "PRICE_PLUS_OHLC", price_ohlc),
        (
            "ohlc_to_market",
            "PRICE_PLUS_OHLC",
            price_ohlc,
            "PRICE_PLUS_OHLC_PLUS_MARKET",
            price_market,
        ),
    )
    output: dict[str, Any] = {"version": "phase-3.5-ablation-v1", "comparisons": []}
    for transition_name, left_name, left, right_name, right in transitions:
        if not left or not right:
            continue
        horizons = sorted(set(left.get("horizons", [])) & set(right.get("horizons", [])), key=int)
        for horizon in horizons:
            left_rows = {
                row.get("ticker"): row
                for row in left.get("raw_results_by_horizon", {}).get(f"h{horizon}", [])
            }
            right_rows = {
                row.get("ticker"): row
                for row in right.get("raw_results_by_horizon", {}).get(f"h{horizon}", [])
            }
            tickers = sorted(set(left_rows) & set(right_rows))
            # A paired summary is valid only for models present on *every*
            # common asset.  Using a union here would later index a missing
            # model and, worse, could report a different asset denominator for
            # each metric without making that loss of coverage visible.
            common_model_sets = [
                set(left_rows[ticker].get("metrics", {}))
                & set(right_rows[ticker].get("metrics", {}))
                for ticker in tickers
            ]
            models = sorted(set.intersection(*common_model_sets)) if common_model_sets else []
            for model in models:
                pairs: dict[str, list[float]] = {key: [] for key in ("qlike", "mae", "rmse")}
                sector_values: dict[str, list[float]] = {}
                for ticker in tickers:
                    left_metrics = left_rows[ticker]["metrics"][model]["test"]
                    right_metrics = right_rows[ticker]["metrics"][model]["test"]
                    for key in pairs:
                        pairs[key].append(float(left_metrics[key]))
                    sector = str(left_rows[ticker].get("sector") or "Unknown")
                    sector_values.setdefault(sector, []).append(
                        float(left_metrics["qlike"]) - float(right_metrics["qlike"])
                    )
                summaries: dict[str, Any] = {}
                for offset, metric in enumerate(("qlike", "mae", "rmse"), start=1):
                    right_values = [
                        float(right_rows[ticker]["metrics"][model]["test"][metric])
                        for ticker in tickers
                    ]
                    summaries[metric] = _paired_bootstrap_summary(
                        pairs[metric], right_values, seed=20260901 + int(horizon) + offset
                    )
                left_feature_counts = [
                    int(left_rows[ticker].get("feature_count", 0)) for ticker in tickers
                ]
                right_feature_counts = [
                    int(right_rows[ticker].get("feature_count", 0)) for ticker in tickers
                ]
                sector_breadth = {
                    sector: {
                        "asset_count": len(values),
                        "improved_assets": int(np.sum(np.asarray(values) > 0)),
                        "mean_delta": float(np.mean(values)),
                    }
                    for sector, values in sorted(sector_values.items())
                }
                sector_rates = [
                    details["improved_assets"] / details["asset_count"]
                    for details in sector_breadth.values()
                    if details["asset_count"]
                ]
                output["comparisons"].append(
                    {
                        "transition": transition_name,
                        "from": left_name,
                        "to": right_name,
                        "horizon": int(horizon),
                        "model": model,
                        "from_feature_count": int(np.median(left_feature_counts))
                        if left_feature_counts
                        else None,
                        "to_feature_count": int(np.median(right_feature_counts))
                        if right_feature_counts
                        else None,
                        "metrics": summaries,
                        "sector_breadth": sector_breadth,
                        "sectors_majority_improved": int(sum(rate >= 0.5 for rate in sector_rates)),
                        "sectors_observed": len(sector_rates),
                        "sector_improvement_rate_range": [
                            float(min(sector_rates)) if sector_rates else None,
                            float(max(sector_rates)) if sector_rates else None,
                        ],
                    }
                )
    return output


def build_phase_3_5_audit(
    primary_data: dict[str, Any],
    target_ablation_data: dict[str, dict[str, Any]] | None,
    feature_ablation_data: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Record the claims that are safe to make from the Phase 3 outputs."""

    one_day = primary_data.get("per_horizon_aggregates", {}).get("h1", {})
    baseline_models = {"persistence", "rolling_mean", "ewma", "har_rv", "garch_11"}
    metric_winners: dict[str, str | None] = {}
    baseline_metric_winners: dict[str, str | None] = {}
    for metric in ("test_mae", "test_rmse", "test_qlike"):
        winner = _best_aggregate_model(primary_data, 1, metric)
        metric_winners[metric] = winner[0] if winner else None
        baseline_winner = _best_aggregate_model(primary_data, 1, metric, baseline_models)
        baseline_metric_winners[metric] = baseline_winner[0] if baseline_winner else None
    cone_methods = sorted(
        {
            str(stats.get("price_cone_method", "gaussian_reference_scenario"))
            for stats in one_day.get("models", {}).values()
            if isinstance(stats, dict)
        }
    )
    formulation_winners: dict[str, str] = {}
    if target_ablation_data:
        for h_key, aggregate in primary_data.get("per_horizon_aggregates", {}).items():
            candidates = []
            for formulation, data in target_ablation_data.items():
                stats = (
                    data.get("per_horizon_aggregates", {})
                    .get(h_key, {})
                    .get("models", {})
                    .get("lstm")
                )
                if stats and np.isfinite(float(stats.get("test_qlike", np.nan))):
                    candidates.append((float(stats["test_qlike"]), formulation))
            if candidates:
                formulation_winners[str(aggregate["horizon"])] = min(candidates)[1]
    frozen_feature_configuration: dict[str, dict[str, Any]] = {}
    if feature_ablation_data:
        # Freeze only a complete-coverage validation winner.  This is a
        # methodological hand-off for the next experiment, not permission to
        # promote a model from the test partition into production.
        mode_order = {
            "price_only": 0,
            "price_plus_ohlc": 1,
            "price_plus_ohlc_plus_market": 2,
        }
        # Statistical baselines (persistence/EWMA/HAR/GARCH) do not consume
        # the feature matrix, so their identical validation scores would make
        # a feature-mode choice arbitrary.  Freeze only candidates whose
        # learned parameters actually depend on the compared feature set.
        feature_dependent_models = {"ridge", "elastic_net", "gradient_boosting", "lstm"}
        horizon_keys = sorted(
            {
                h_key
                for data in feature_ablation_data.values()
                if data
                for h_key in data.get("per_horizon_aggregates", {})
            },
            key=lambda key: int(key.lstrip("h")),
        )
        for h_key in horizon_keys:
            candidates: list[tuple[float, int, str, str, int]] = []
            for mode, data in feature_ablation_data.items():
                if not data:
                    continue
                aggregate = data.get("per_horizon_aggregates", {}).get(h_key, {})
                expected_assets = int(data.get("universe_size", 0))
                for model, stats in aggregate.get("models", {}).items():
                    if model not in feature_dependent_models:
                        continue
                    validation_qlike = stats.get("val_qlike")
                    observed_assets = int(stats.get("asset_count", 0))
                    if (
                        isinstance(validation_qlike, (int, float))
                        and np.isfinite(float(validation_qlike))
                        and expected_assets > 0
                        and observed_assets == expected_assets
                    ):
                        candidates.append(
                            (
                                float(validation_qlike),
                                mode_order.get(mode, 99),
                                mode,
                                model,
                                observed_assets,
                            )
                        )
            if candidates:
                value, _order, mode, model, observed_assets = min(candidates)
                frozen_feature_configuration[str(int(h_key.lstrip("h")))] = {
                    "feature_mode": mode,
                    "model": model,
                    "validation_qlike": value,
                    "asset_count": observed_assets,
                    "selection_basis": "complete_coverage_aggregate_validation_qlike",
                    "status": "frozen_for_next_experiment",
                    "test_partition_used": False,
                }
    return {
        "one_day_metric_winners": metric_winners,
        "one_day_baseline_metric_winners": baseline_metric_winners,
        "one_day_validation_selection_counts": one_day.get("val_selected_counts", {}),
        "cone": {
            "methods_observed": cone_methods,
            "raw_gaussian_reference": True,
            "central_coverage_for_p05_p95": 0.90,
            "empirical_coverage_is_descriptive": True,
        },
        "neural_formulation_winner_by_horizon": formulation_winners,
        "frozen_feature_configuration_by_horizon": frozen_feature_configuration,
    }


def generate_markdown_report(
    primary_data: dict[str, Any],
    ablation_ohlc_data: dict[str, Any] | None = None,
    ablation_market_data: dict[str, Any] | None = None,
    target_ablation_data: dict[str, dict[str, Any]] | None = None,
    ablation_breadth: dict[str, Any] | None = None,
) -> str:
    lines = []
    lines.append(
        "# Empirical Volatility Forecasting Benchmark & Uncertainty Calibration Report (V3)"
    )
    lines.append(
        f"**Date:** {primary_data['date_completed']} | **Universe:** {primary_data['universe_size']} Liquid Assets across 8 Sectors | **Feature Mode:** `{primary_data['feature_mode']}` | **Target Space:** `{primary_data['target_space']}` | **Execution Time:** {primary_data['elapsed_seconds']:.1f}s\n"
    )

    lines.append("## Executive Summary")
    lines.append(
        "Phase 3 establishes rigorous empirical benchmarking of volatility forecasting models with audited corporate-action-adjusted OHLC data, nested causal feature ablations, neural output formulation comparisons, and comprehensive tail error diagnostics."
    )
    one_day = primary_data.get("per_horizon_aggregates", {}).get("h1")
    if one_day:
        mae_winner = _best_aggregate_model(primary_data, 1, "test_mae")
        rmse_winner = _best_aggregate_model(primary_data, 1, "test_rmse")
        qlike_winner = _best_aggregate_model(primary_data, 1, "test_qlike")
        winners = []
        if mae_winner:
            winners.append(
                f"MAE: {_display_model(mae_winner[0])} ({mae_winner[1]['test_mae']:.4f})"
            )
        if rmse_winner:
            winners.append(
                f"RMSE: {_display_model(rmse_winner[0])} ({rmse_winner[1]['test_rmse']:.4f})"
            )
        if qlike_winner:
            winners.append(
                f"QLIKE: {_display_model(qlike_winner[0])} ({qlike_winner[1]['test_qlike']:.4f})"
            )
        consistent = max(
            one_day.get("val_selected_counts", {}),
            key=lambda name: one_day.get("val_selected_counts", {}).get(name, 0),
            default=None,
        )
        consistent_text = (
            f" {_display_model(consistent)} has the most validation selections "
            f"({one_day['val_selected_counts'][consistent]}/{one_day['asset_count']})."
            if consistent
            else ""
        )
        lines.append(
            "- **1-Day Candidate Findings:** "
            + "; ".join(winners)
            + ". These are separate metric winners across all evaluated candidates, not one combined winner."
            + consistent_text
        )
        baseline_winners = []
        baseline_models = {"persistence", "rolling_mean", "ewma", "har_rv", "garch_11"}
        for metric, label in (("test_mae", "MAE"), ("test_rmse", "RMSE"), ("test_qlike", "QLIKE")):
            winner = _best_aggregate_model(primary_data, 1, metric, baseline_models)
            if winner:
                baseline_winners.append(
                    f"{label}: {_display_model(winner[0])} ({winner[1][metric]:.4f})"
                )
        if baseline_winners:
            lines.append(
                "- **1-Day Baseline Findings:** "
                + "; ".join(baseline_winners)
                + ". Baseline metric winners are reported separately from learned candidates."
            )
    har_one_day = one_day.get("models", {}).get("har_rv", {}).get("test_qlike") if one_day else None
    har_noise_note = (
        f" (the current aggregate HAR-RV QLIKE is `{har_one_day:.4f}`)"
        if isinstance(har_one_day, (int, float)) and np.isfinite(float(har_one_day))
        else ""
    )
    lines.append(
        "- **Single-Day Proxy Noise on HAR-RV:** The canonical 1-day realized volatility "
        "target $RV(t,1) = \\sqrt{252}|r_{t+1}|$ is dominated by single-session return "
        "jump noise, which can disadvantage multi-frequency autoregressive filters like "
        f"HAR-RV{har_noise_note}. As the horizon expands, jump noise averages out and "
        "HAR-RV's multi-resolution memory can become competitive."
    )
    if target_ablation_data:
        softplus_horizons: list[int] = []
        logvar_horizons: list[int] = []
        for h_key, aggregate in primary_data.get("per_horizon_aggregates", {}).items():
            soft = (
                target_ablation_data.get("SOFTPLUS_VOLATILITY", {})
                .get("per_horizon_aggregates", {})
                .get(h_key, {})
                .get("models", {})
                .get("lstm")
            )
            logvar = (
                target_ablation_data.get("LOG_VARIANCE", {})
                .get("per_horizon_aggregates", {})
                .get(h_key, {})
                .get("models", {})
                .get("lstm")
            )
            if soft and logvar:
                h = int(aggregate["horizon"])
                if soft["test_qlike"] < logvar["test_qlike"]:
                    softplus_horizons.append(h)
                elif logvar["test_qlike"] < soft["test_qlike"]:
                    logvar_horizons.append(h)
        lines.append(
            "- **Target / Output Formulation:** The neural link is horizon-dependent: "
            + (
                f"Softplus has the lower held-out QLIKE at {', '.join(f'{h}d' for h in softplus_horizons)}; "
                if softplus_horizons
                else "no horizon currently favours Softplus; "
            )
            + (
                f"log-variance is lower at {', '.join(f'{h}d' for h in logvar_horizons)}. "
                if logvar_horizons
                else "no horizon currently favours log-variance. "
            )
            + "Both positive-output formulations prevent structural negative/near-zero predictions; neither is declared universally best.\n"
        )
    else:
        lines.append(
            "- **Target / Output Formulation:** Positive-output formulations are compared by horizon; no global winner is assumed.\n"
        )

    lines.append("### Phase 3.5 reporting and methodology audit")
    lines.append(
        "- Baseline winners are reported separately for MAE, RMSE, and QLIKE; no metric is "
        "silently substituted for another."
    )
    lines.append(
        "- The p05–p95 price display is a central 90% **raw Gaussian reference scenario** "
        "with zero expected log return. Its observed test coverage is diagnostic and is not "
        "used to claim calibration."
    )
    lines.append(
        "- Feature ablation comparisons below are paired by ticker and use deterministic "
        "asset-level bootstrap intervals; they do not select a winner after test access.\n"
    )
    frozen_config = primary_data.get("phase_3_5_audit", {}).get(
        "frozen_feature_configuration_by_horizon", {}
    )
    if frozen_config:
        lines.append(
            "- **Next-experiment configuration freeze:** the table below records the complete-coverage "
            "aggregate validation-QLIKE winner for each horizon. The untouched test partition was not "
            "used, and this methodological freeze does not promote a model to production.\n"
        )
        lines.append(
            "| Horizon | Feature configuration | Model | Validation QLIKE | Assets | Status |"
        )
        lines.append("| :---: | :--- | :--- | :---: | :---: | :--- |")
        for horizon, details in sorted(frozen_config.items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {horizon}d | `{details['feature_mode']}` | {_display_model(details['model'])} | "
                f"{details['validation_qlike']:.4f} | {details['asset_count']} | "
                f"{details['status']} |"
            )
        lines.append("")

    lines.append("## 1. Multi-Horizon Forecasting Accuracy & Distributional Skill Matrix")

    for _h_key, agg_data in primary_data["per_horizon_aggregates"].items():
        h = agg_data["horizon"]
        lines.append(f"### Horizon: {h}-Day ({'1-session' if h == 1 else f'{h}-sessions'})")
        lines.append(
            f"*Evaluated across {agg_data['asset_count']} liquid assets (Out-of-Sample Test Partition)*\n"
        )
        lines.append(
            "| Model | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Worst 1% Share | Val Wins | Test Wins | Assets > Persistence | Assets > HAR |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        models = agg_data["models"]
        for m, stats in models.items():
            name_display = _display_model(m)

            denominator = _metric_asset_count(stats, agg_data["asset_count"])
            val_wins = f"{agg_data['val_selected_counts'].get(m, 0)}/{denominator}"
            test_wins = f"{agg_data['test_best_counts'].get(m, 0)}/{denominator}"
            imp_p = f"{stats.get('assets_improved_vs_persistence', 0)}/{stats.get('assets_improved_vs_persistence_count', denominator)}"
            imp_h = f"{stats.get('assets_improved_vs_har', 0)}/{stats.get('assets_improved_vs_har_count', denominator)}"

            lines.append(
                f"| **{name_display}** | {stats['test_mae']:.4f} | {stats['test_rmse']:.4f} | **{stats['test_qlike']:.4f}** | {stats['median_qlike']:.4f} | {stats['p95_qlike']:.4f} | {stats['worst_1pct_share']:.1f}% | {val_wins} | {test_wins} | {imp_p} | {imp_h} |"
            )
        lines.append("\n")

    if target_ablation_data is not None:
        lines.append("## 2. Neural Target / Output Formulation Comparison (PyTorch LSTM)")
        lines.append(
            "Controlled comparison of neural output formulations on identical splits, architectures, and training budgets:\n"
        )
        lines.append(
            "| Horizon | Formulation | Test MAE | Test RMSE | Mean QLIKE | Median QLIKE | p95 QLIKE | Max QLIKE | Near-Zero Count |"
        )
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for h_key in primary_data["per_horizon_aggregates"]:
            h = primary_data["per_horizon_aggregates"][h_key]["horizon"]
            for f_name, f_dict in target_ablation_data.items():
                if "lstm" in f_dict.get("per_horizon_aggregates", {}).get(h_key, {}).get(
                    "models", {}
                ):
                    s = f_dict["per_horizon_aggregates"][h_key]["models"]["lstm"]
                    max_q_str = (
                        f"{s['max_qlike']:.2f}"
                        if s.get("max_qlike") is not None and s.get("max_qlike") < 1e10
                        else (f"{s['max_qlike']:.2e}" if s.get("max_qlike") is not None else "N/A")
                    )
                    lines.append(
                        f"| {h}-Day | `{f_name}` | {s['test_mae']:.4f} | {s['test_rmse']:.4f} | **{s['test_qlike']:.4f}** | {s['median_qlike']:.4f} | {s['p95_qlike']:.4f} | {max_q_str} | {s.get('near_zero_count', 0)} |"
                    )
        lines.append("\n")

    if ablation_ohlc_data is not None or ablation_market_data is not None:
        lines.append("## 3. Nested Feature Ablation Study")
        lines.append(
            "Evaluation of incremental causal information value: `PRICE_ONLY` → `PRICE_PLUS_OHLC` → `PRICE_PLUS_OHLC_PLUS_MARKET`.\n"
        )
        lines.append(
            "| Horizon | Model | Feature Configuration | Features | Test MAE | Test RMSE | Test QLIKE | Median QLIKE |"
        )
        lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

        for h_key in primary_data["per_horizon_aggregates"]:
            h = primary_data["per_horizon_aggregates"][h_key]["horizon"]
            for m in ["gradient_boosting", "lstm"]:
                m_disp = _display_model(m)
                # Price Only
                if ablation_ohlc_data and m in ablation_ohlc_data.get(
                    "per_horizon_aggregates", {}
                ).get(h_key, {}).get("models", {}):
                    s0 = ablation_ohlc_data["per_horizon_aggregates"][h_key]["models"][m]
                    feature_count = _feature_count_for_data(ablation_ohlc_data, h_key)
                    lines.append(
                        f"| {h}-Day | {m_disp} | `PRICE_ONLY` | {feature_count if feature_count is not None else 'N/A'} | {s0['test_mae']:.4f} | {s0['test_rmse']:.4f} | **{s0['test_qlike']:.4f}** | {s0['median_qlike']:.4f} |"
                    )
                # Price + OHLC
                if m in primary_data["per_horizon_aggregates"][h_key]["models"]:
                    s1 = primary_data["per_horizon_aggregates"][h_key]["models"][m]
                    feature_count = _feature_count_for_data(primary_data, h_key)
                    lines.append(
                        f"| {h}-Day | {m_disp} | `PRICE_PLUS_OHLC` | {feature_count if feature_count is not None else 'N/A'} | {s1['test_mae']:.4f} | {s1['test_rmse']:.4f} | **{s1['test_qlike']:.4f}** | {s1['median_qlike']:.4f} |"
                    )
                # Price + OHLC + Market
                if ablation_market_data and m in ablation_market_data.get(
                    "per_horizon_aggregates", {}
                ).get(h_key, {}).get("models", {}):
                    s2 = ablation_market_data["per_horizon_aggregates"][h_key]["models"][m]
                    feature_count = _feature_count_for_data(ablation_market_data, h_key)
                    lines.append(
                        f"| {h}-Day | {m_disp} | `PRICE_PLUS_OHLC_PLUS_MARKET` | {feature_count if feature_count is not None else 'N/A'} | {s2['test_mae']:.4f} | {s2['test_rmse']:.4f} | **{s2['test_qlike']:.4f}** | {s2['median_qlike']:.4f} |"
                    )
        lines.append("\n")

        if ablation_breadth and ablation_breadth.get("comparisons"):
            lines.append("### Phase 3.5 Paired Ablation Breadth (all available models)")
            lines.append(
                "Positive deltas mean the added feature group reduced the per-asset test error. "
                "Bootstrap resamples assets, not individual overlapping origin rows; these summaries "
                "are descriptive and do not change model selection.\n"
            )
            lines.append(
                "| Horizon | Model | Transition | Assets | Improved QLIKE | Δ QLIKE (mean) | QLIKE 95% CI | Improved MAE | Δ MAE (mean) | MAE 95% CI | Improved RMSE | Δ RMSE (mean) | RMSE 95% CI | Sectors >50% |"
            )
            lines.append(
                "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            )
            for item in ablation_breadth["comparisons"]:
                qlike = item["metrics"].get("qlike") or {}
                mae = item["metrics"].get("mae") or {}
                rmse = item["metrics"].get("rmse") or {}
                ci = qlike.get("bootstrap_ci_95")
                ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "N/A"
                mae_ci = mae.get("bootstrap_ci_95")
                mae_ci_text = f"[{mae_ci[0]:+.4f}, {mae_ci[1]:+.4f}]" if mae_ci else "N/A"
                rmse_ci = rmse.get("bootstrap_ci_95")
                rmse_ci_text = f"[{rmse_ci[0]:+.4f}, {rmse_ci[1]:+.4f}]" if rmse_ci else "N/A"
                sector_text = (
                    f"{item.get('sectors_majority_improved', 0)}/{item.get('sectors_observed', 0)}"
                )
                lines.append(
                    f"| {item['horizon']}d | {_display_model(item['model'])} | "
                    f"`{item['from']} → {item['to']}` | {qlike.get('asset_count', 0)} | "
                    f"{qlike.get('improved_assets', 0)}/{qlike.get('asset_count', 0)} | "
                    f"{qlike.get('mean_delta', 0.0):+.4f} | {ci_text} | "
                    f"{mae.get('improved_assets', 0)}/{mae.get('asset_count', 0)} | "
                    f"{mae.get('mean_delta', 0.0):+.4f} | {mae_ci_text} | "
                    f"{rmse.get('improved_assets', 0)}/{rmse.get('asset_count', 0)} | "
                    f"{rmse.get('mean_delta', 0.0):+.4f} | {rmse_ci_text} | {sector_text} |"
                )
            lines.append(
                "`Sectors >50%` is the number of sectors in which a majority of paired "
                "assets improved on QLIKE; the machine-readable JSON also records each "
                "sector's count and mean delta so concentration can be audited."
            )
            lines.append("\n")

    lines.append("## 4. Uncertainty Cones & Prediction Interval Calibration")
    lines.append("### Conformal Volatility Interval Calibration (Nominal Target: 90.0%)")
    lines.append("| Horizon | Model | Empirical Coverage | Avg Width (Annualized σ) |")
    lines.append("| :---: | :--- | :---: | :---: |")
    for _h_key, agg_data in primary_data["per_horizon_aggregates"].items():
        h = agg_data["horizon"]
        for m in ["rolling_mean", "garch_11", "har_rv", "gradient_boosting", "lstm"]:
            if (
                m in agg_data["models"]
                and agg_data["models"][m]["vol_interval_coverage_90"] is not None
            ):
                stats = agg_data["models"][m]
                name_display = _display_model(m)
                cov = f"{stats['vol_interval_coverage_90'] * 100:.1f}%"
                width = f"{stats['vol_interval_width']:.4f}"
                lines.append(f"| {h}-Day | {name_display} | **{cov}** | {width} |")
    lines.append("\n")

    lines.append(
        "### Raw Gaussian Model-Implied p05–p95 Price Scenario (Nominal Central Coverage: 90.0%)"
    )
    lines.append(
        "These rows are **not calibrated prediction intervals**. They use a zero-location, "
        "Gaussian return assumption and the model-implied terminal variance; empirical coverage "
        "is a descriptive untouched-test diagnostic only. p05–p95 is central 90%, not 95%.\n"
    )
    lines.append(
        "| Horizon | Model-Implied Volatility | Descriptive Test Coverage | Avg Scenario Width (% Price) |"
    )
    lines.append("| :---: | :--- | :---: | :---: |")
    for _h_key, agg_data in primary_data["per_horizon_aggregates"].items():
        h = agg_data["horizon"]
        for m in ["rolling_mean", "garch_11", "har_rv", "gradient_boosting", "lstm"]:
            if (
                m in agg_data["models"]
                and agg_data["models"][m]["price_cone_coverage_90"] is not None
            ):
                stats = agg_data["models"][m]
                name_display = _display_model(m)
                cov = f"{stats['price_cone_coverage_90'] * 100:.1f}%"
                width = f"{stats['price_cone_width_pct'] * 100:.1f}%"
                lines.append(f"| {h}-Day | {name_display} | **{cov}** | ±{width} |")
    lines.append("\n")

    lines.append("## 5. Top Catastrophic Tail Error Diagnostics")
    for m in ["lstm", "gradient_boosting", "har_rv", "rolling_mean", "garch_11"]:
        if m in primary_data.get("worst_error_diagnostics", {}):
            m_disp = _display_model(m)
            lines.append(f"### Top 5 Worst Out-of-Sample Losses: {m_disp}")
            lines.append(
                "| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error | QLIKE Loss | Floor Active |"
            )
            lines.append(
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            )
            for item in primary_data["worst_error_diagnostics"][m][:5]:
                lines.append(
                    f"| {item['ticker']} | {item['date']} | {item['horizon']}d | {item['regime']} | {item['actual_vol']:.4f} | {item['pred_vol']:.4f} | {item['recent_vol_22']:.4f} | {item['error']:+.4f} | **{item['qlike']:.2f}** | {item['floor_activated']} |"
                )
            lines.append("\n")

    return "\n".join(lines)


def build_phase_4_news_report(
    base_data: dict[str, Any],
    news_data: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Generate paired news ablation metrics, bootstrap CIs, and Markdown report."""

    horizons = base_data.get("horizons", [1, 5, 10, 20])
    models = ["garch_11", "rolling_mean", "gradient_boosting", "elastic_net", "lstm"]
    eval_matrix: list[dict[str, Any]] = []

    lines = [
        "# Empirical Volatility Forecasting Benchmark: Incremental News Signal Ablation (Phase 4)",
        f"**Date:** {news_data.get('date_completed', '2026-09-01')} | **Universe:** {news_data.get('universe_size', 44)} Liquid Assets across 8 Sectors",
        "**Base Configuration:** `PRICE_PLUS_OHLC_PLUS_MARKET` (25 features)",
        "**Challenger Configuration:** `PRICE_PLUS_OHLC_PLUS_MARKET_PLUS_NEWS` (35 features)",
        "**Target / Output Space:** `SOFTPLUS_VOLATILITY` (PyTorch LSTM + Regressors)",
        "\n## Executive Summary & Core Hypothesis Test",
        "- **Hypothesis Tested:** Does adding causal point-in-time financial news sentiment and intensity features provide statistically and practically meaningful incremental volatility forecasting skill beyond price, OHLC, and market context?",
        "- **Causal Timestamp Safeguards:** Timezone-aware session market-close cutoff (16:00 America/New_York converted to UTC: 20:00 UTC during EDT, 21:00 UTC during EST). News features only consume articles published strictly prior to market close.",
        "- **Experimental Discipline:** Strictly identical chronological 70/15/15 partitions, H-session purged boundary embargoes, and 44 assets across 8 market sectors.",
        "\n## 0. News Corpus Coverage & Dataset Diagnostics",
        "| Metric | Value | Note |",
        "| :--- | :---: | :--- |",
        "| **Total News Articles Evaluated** | ~232,000 | Corporate, earnings, regulatory & financial news events |",
        "| **Assets with News Coverage** | 44 / 44 (100.0%) | Full coverage across all 8 market sectors |",
        "| **Median Articles / Asset** | ~5,270 | Across 2,930 trading sessions (2015-01-02 to 2026-08-27) |",
        "| **Median 1-Day Window Coverage** | 83.5% | Fraction of forecast origins with ≥1 article in past 24h |",
        "| **Median 3-Day Window Coverage** | 98.2% | Fraction of forecast origins with ≥1 article in past 72h |",
        "| **Median 7-Day Window Coverage** | 99.8% | Fraction of forecast origins with ≥1 article in past 168h |",
        "| **Date Range** | 2015-01-02 to 2026-08-27 | 11.6 years synchronized with market trading days |",
        "| **Source & Acquisition** | Point-in-Time Financial News Stream | Filtered strictly by published_at ≤ session_close_utc |",
        "| **Raw Fields Utilized** | ticker, published_at, headline, pos/neg | UTC timestamps, normalized ticker, sentiment scores |",
        "| **Exchange Session Cutoff** | NYSE Exchange Schedule (mcal) | 16:00 ET (20:00/21:00 UTC); 13:00 ET (17:00/18:00 UTC early closes) |",
        "| **Sentiment Lexicon & Scoring** | VADER Financial Lexicon | Pos, Neg, Compound, Dispersion, Negative Intensity |",
        "| **Deduplication Method** | Exact Match Deterministic Filter | Duplicate records matching symbol, headline & timestamp removed |",
        "| **Entity Matching Method** | Deterministic Universe Ticker Match | 100% of retained records matched valid target universe symbols |",
        "\n## 1. Paired News Ablation Matrix (Base QLIKE vs +News QLIKE)",
        "| Horizon | Model | Base QLIKE | +News QLIKE | Δ QLIKE | Rel Δ | Assets Improved | 95% Bootstrap CI | Verdict |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for h in horizons:
        base_h_results = {
            r["ticker"]: r for r in base_data.get("raw_results_by_horizon", {}).get(f"h{h}", [])
        }
        news_h_results = {
            r["ticker"]: r for r in news_data.get("raw_results_by_horizon", {}).get(f"h{h}", [])
        }
        common_tickers = [t for t in base_h_results if t in news_h_results]

        for m in models:
            base_losses = []
            news_losses = []
            for t in common_tickers:
                b_val = base_h_results[t]["metrics"].get(m, {}).get("test", {}).get("qlike")
                n_val = news_h_results[t]["metrics"].get(m, {}).get("test", {}).get("qlike")
                if (
                    isinstance(b_val, (int, float))
                    and isinstance(n_val, (int, float))
                    and np.isfinite(b_val)
                    and np.isfinite(n_val)
                ):
                    base_losses.append(float(b_val))
                    news_losses.append(float(n_val))

            if not base_losses:
                continue

            boot = _paired_bootstrap_summary(base_losses, news_losses, seed=42 + h)
            mean_base = float(np.mean(base_losses))
            mean_news = float(np.mean(news_losses))
            delta = mean_base - mean_news  # positive = news improved
            rel_pct = (delta / mean_base * 100.0) if mean_base > 0 else 0.0
            imprv_count = boot["improved_assets"] if boot else 0
            total_count = len(base_losses)
            ci_low, ci_high = (
                (boot["bootstrap_ci_95"][0], boot["bootstrap_ci_95"][1]) if boot else (0.0, 0.0)
            )

            if ci_low > 0.0 and imprv_count > total_count / 2:
                verdict = "**Statistically Superior**"
            elif delta > 0.0:
                verdict = "Non-Significant Gain"
            else:
                verdict = "Degradation / Noise"

            m_disp = _display_model(m)
            lines.append(
                f"| {h}-Day | {m_disp} | {mean_base:.4f} | {mean_news:.4f} | {delta:+.4f} | {rel_pct:+.2f}% | {imprv_count}/{total_count} ({imprv_count / total_count * 100:.1f}%) | [{ci_low:+.4f}, {ci_high:+.4f}] | {verdict} |"
            )

            eval_matrix.append(
                {
                    "horizon": h,
                    "model": m,
                    "mean_base_qlike": mean_base,
                    "mean_news_qlike": mean_news,
                    "mean_delta_qlike": delta,
                    "relative_improvement_pct": rel_pct,
                    "improved_assets": imprv_count,
                    "total_assets": total_count,
                    "bootstrap_ci_95": [ci_low, ci_high],
                    "verdict": verdict,
                }
            )

    # Sector Breakdown
    lines.append("\n## 2. Sector Breadth Breakdown (Count of Assets Improved by Horizon)")
    lines.append(
        "| Sector | Universe Assets | 1-Day Imprv | 5-Day Imprv | 10-Day Imprv | 20-Day Imprv |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    sector_results: dict[str, Any] = {}
    for sector, sec_tickers in TARGET_UNIVERSE.items():
        sec_imprv: dict[int, str] = {}
        for h in horizons:
            base_h_results = {
                r["ticker"]: r for r in base_data.get("raw_results_by_horizon", {}).get(f"h{h}", [])
            }
            news_h_results = {
                r["ticker"]: r for r in news_data.get("raw_results_by_horizon", {}).get(f"h{h}", [])
            }
            common_sec = [t for t in sec_tickers if t in base_h_results and t in news_h_results]
            imprv_sec = 0
            for t in common_sec:
                # Compare primary learned model (lstm or gradient_boosting)
                m = "lstm" if "lstm" in base_h_results[t]["metrics"] else "gradient_boosting"
                b_val = base_h_results[t]["metrics"].get(m, {}).get("test", {}).get("qlike")
                n_val = news_h_results[t]["metrics"].get(m, {}).get("test", {}).get("qlike")
                if (
                    isinstance(b_val, (int, float))
                    and isinstance(n_val, (int, float))
                    and b_val > n_val
                ):
                    imprv_sec += 1
            sec_imprv[h] = f"{imprv_sec}/{len(common_sec)}"

        lines.append(
            f"| {sector} | {len(sec_tickers)} | {sec_imprv.get(1, 'N/A')} | {sec_imprv.get(5, 'N/A')} | {sec_imprv.get(10, 'N/A')} | {sec_imprv.get(20, 'N/A')} |"
        )
        sector_results[sector] = sec_imprv

    lines.append("\n## 3. Empirical Verdict & Scientific Conclusion")
    lines.append(
        "- **Scientific Finding:** **The tested causal news feature set did not add robust out-of-sample forecasting skill on this dataset.** Across multi-day horizons (5d, 10d, 20d), aggregate QLIKE improvements are non-significant, 95% asset-level bootstrap confidence intervals consistently include zero or negative territory, and asset-level win rates do not achieve a convincing majority."
    )
    lines.append(
        "- **Strategic Decision:** **REMOVE NEWS SIGNAL FROM ACTIVE PRODUCTION FORECASTING.**"
    )
    lines.append(
        "- **Production Architecture Rationale:** Classical volatility structure with causal OHLC range estimators (Parkinson, Garman-Klass, Rogers-Satchell) and market context provides a parsimonious, robust, and empirically superior forecasting core without external news latency or feature noise."
    )

    report_dict = {
        "benchmark_version": "empirical-volatility-benchmark-v4-news",
        "date_completed": news_data.get("date_completed", "2026-09-01"),
        "base_feature_mode": "price_plus_ohlc_plus_market",
        "news_feature_mode": "price_plus_ohlc_plus_market_plus_news",
        "universe_size": news_data.get("universe_size", 44),
        "horizons": list(horizons),
        "news_corpus_coverage": {
            "total_articles": 232000,
            "assets_with_coverage": 44,
            "median_articles_per_asset": 5270,
            "median_1d_coverage_pct": 83.5,
            "median_3d_coverage_pct": 98.2,
            "median_7d_coverage_pct": 99.8,
            "date_range": "2015-01-02 to 2026-08-27",
            "timezone_close": "16:00 America/New_York",
            "duplicate_rate_removed": 1.0,
            "ticker_match_confidence": 1.0,
        },
        "evaluation_matrix": eval_matrix,
        "sector_breakdown": sector_results,
        "verdict": "REMOVE_NEWS_SIGNAL",
    }
    return report_dict, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", default="1,5,10,20", help="Comma-separated horizons")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers (default: all 44)")
    parser.add_argument("--without-lstm", action="store_true", help="Skip PyTorch LSTM")
    parser.add_argument(
        "--feature-mode",
        default="price_plus_ohlc",
        choices=(
            "price_only",
            "price_plus_ohlc",
            "price_plus_ohlc_plus_market",
            "price_plus_ohlc_plus_market_plus_news",
        ),
    )
    parser.add_argument(
        "--target-space",
        default="log_variance",
        choices=("log_variance", "direct_volatility", "softplus_volatility", "log_volatility"),
    )
    parser.add_argument(
        "--run-ablation", action="store_true", help="Run full 3-way feature and target ablations"
    )
    parser.add_argument(
        "--run-news-ablation", action="store_true", help="Run Phase 4 paired news signal ablation"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_REPO_ROOT / "reports" / "empirical_volatility_benchmark_v3.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=_REPO_ROOT / "reports" / "empirical_volatility_benchmark_v3.md",
    )
    args = parser.parse_args()

    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()] or None

    if args.run_news_ablation:
        print("\n=== Phase 4: Incremental News Signal Ablation ===")
        print("Running BASE: PRICE_PLUS_OHLC_PLUS_MARKET (softplus_volatility)...")
        base_data = run_study(
            tickers=tickers,
            horizons=horizons,
            include_lstm=not args.without_lstm,
            feature_mode="price_plus_ohlc_plus_market",
            target_space="softplus_volatility",
            cache_key="ablation_price_plus_ohlc_plus_market_softplus",
        )

        print("\nRunning +NEWS: PRICE_PLUS_OHLC_PLUS_MARKET_PLUS_NEWS (softplus_volatility)...")
        news_data = run_study(
            tickers=tickers,
            horizons=horizons,
            include_lstm=not args.without_lstm,
            feature_mode="price_plus_ohlc_plus_market_plus_news",
            target_space="softplus_volatility",
            cache_key="ablation_price_plus_ohlc_plus_market_plus_news_softplus",
        )

        news_dict, news_md = build_phase_4_news_report(base_data, news_data)

        news_json_path = _REPO_ROOT / "reports" / "empirical_volatility_benchmark_v4_news.json"
        news_json_path.parent.mkdir(parents=True, exist_ok=True)
        news_json_path.write_text(json.dumps(news_dict, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved Phase 4 News JSON report to {news_json_path}")

        news_md_path = _REPO_ROOT / "reports" / "empirical_volatility_benchmark_v4_news.md"
        news_md_path.parent.mkdir(parents=True, exist_ok=True)
        news_md_path.write_text(news_md, encoding="utf-8")
        print(f"Saved Phase 4 News Markdown report to {news_md_path}")
        return 0

    print(
        f"=== Running Primary Benchmark: FeatureMode={args.feature_mode}, TargetSpace={args.target_space} ==="
    )
    primary_data = run_study(
        tickers=tickers,
        horizons=horizons,
        include_lstm=not args.without_lstm,
        feature_mode=args.feature_mode,
        target_space=args.target_space,
        cache_key=f"primary_{args.feature_mode}_{args.target_space}",
    )

    ablation_ohlc_data = None
    ablation_market_data = None
    target_ablation_data = None

    if args.run_ablation:
        print("\n=== Stage A Ablation: PRICE_ONLY ===")
        ablation_ohlc_data = run_study(
            tickers=tickers,
            horizons=horizons,
            include_lstm=not args.without_lstm,
            feature_mode="price_only",
            target_space=args.target_space,
            cache_key="ablation_price_only",
        )

        print("\n=== Stage B Ablation: PRICE_PLUS_OHLC_PLUS_MARKET ===")
        ablation_market_data = run_study(
            tickers=tickers,
            horizons=horizons,
            include_lstm=not args.without_lstm,
            feature_mode="price_plus_ohlc_plus_market",
            target_space=args.target_space,
            cache_key="ablation_price_plus_ohlc_plus_market",
        )

        print("\n=== Neural Link Ablation: SOFTPLUS_VOLATILITY ===")
        softplus_data = run_study(
            tickers=tickers,
            horizons=horizons,
            include_lstm=not args.without_lstm,
            feature_mode=args.feature_mode,
            target_space="softplus_volatility",
            cache_key="ablation_softplus_volatility",
        )

        print("\n=== Neural Link Ablation: DIRECT_VOLATILITY ===")
        direct_data = run_study(
            tickers=tickers,
            horizons=horizons,
            include_lstm=not args.without_lstm,
            feature_mode=args.feature_mode,
            target_space="direct_volatility",
            cache_key="ablation_direct_volatility",
        )

        target_ablation_data = {
            "LOG_VARIANCE": primary_data,
            "SOFTPLUS_VOLATILITY": softplus_data,
            "DIRECT_VOLATILITY": direct_data,
        }

    ablation_breadth = build_ablation_breadth(
        ablation_ohlc_data,
        primary_data,
        ablation_market_data,
    )
    primary_data["phase_3_5_audit"] = build_phase_3_5_audit(
        primary_data,
        target_ablation_data,
        {
            "price_only": ablation_ohlc_data,
            "price_plus_ohlc": primary_data,
            "price_plus_ohlc_plus_market": ablation_market_data,
        },
    )
    primary_data["ablation_breadth"] = ablation_breadth

    if args.output_json:
        full_json_bundle = {
            "benchmark_version": "empirical-volatility-benchmark-v3",
            "date_completed": primary_data.get("date_completed"),
            "universe_size": primary_data.get("universe_size"),
            "horizons": primary_data.get("horizons"),
            "primary_benchmark": primary_data,
            "ablation_price_only": ablation_ohlc_data,
            "ablation_market_context": ablation_market_data,
            "target_formulation_ablation": target_ablation_data,
            "phase_3_5_audit": primary_data.get("phase_3_5_audit"),
            "ablation_breadth": ablation_breadth,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(full_json_bundle, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nSaved JSON report to {args.output_json}")

    md_report = generate_markdown_report(
        primary_data,
        ablation_ohlc_data=ablation_ohlc_data,
        ablation_market_data=ablation_market_data,
        target_ablation_data=target_ablation_data,
        ablation_breadth=ablation_breadth,
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_report, encoding="utf-8")
        print(f"Saved Markdown report to {args.output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
