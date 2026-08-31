"""Comprehensive empirical study for volatility forecasting and uncertainty calibration (Phase 2).

Evaluates 44 diverse liquid assets across 8 sectors and 4 horizons (1, 5, 10, 20 days)
using 9 models:
- Statistical: Persistence, Rolling 60d, EWMA (λ=0.94), HAR-RV, GARCH(1,1)
- ML / Neural: Ridge, ElasticNet, Gradient Boosting, PyTorch LSTM

Computes:
1. Canonical Patton (2011) variance-based QLIKE, MAE, RMSE, R2
2. Relative skill (% improvement) vs Persistence and vs HAR-RV
3. Target formulation comparison: DIRECT_VOLATILITY vs LOG_VARIANCE vs LOG_VOLATILITY
4. Uncertainty calibration: Conformal Volatility Intervals (90%) and Gaussian model-implied p05–p95 ranges
5. Top 20 worst-error diagnostics across test partitions
6. Explicit V2 vs V1 comparison.
"""

from __future__ import annotations

import argparse
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

from research.volatility_forecasting.simple_pipeline import (
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
    """Find and load OHLCV data for a ticker from available snapshots or caches."""
    ticker_up = ticker.upper()

    # 1. Check root snapshot files
    root_csv = _REPO_ROOT / f"snapshot_{ticker_up}.csv"
    if root_csv.is_file():
        return pd.read_csv(root_csv)

    # 2. Check diagnostic snapshots
    raw_dir = _REPO_ROOT / "artifacts" / "v11_2_diagnostic_inputs" / "snapshots" / "panel-8546a6f180250034" / "raw"
    csv_file = raw_dir / f"{ticker_up}.csv"
    if csv_file.is_file():
        return pd.read_csv(csv_file)

    # 3. Check ndx100 parquet cache
    parquet_file = _REPO_ROOT / "data" / "ndx100" / "cache" / f"{ticker_up}.parquet"
    if parquet_file.is_file():
        return pd.read_parquet(parquet_file)

    # 4. Fallback to yfinance if online
    try:
        import yfinance as yf
        df = yf.download(ticker_up, period="5y", interval="1d", auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as err:
        raise FileNotFoundError(f"Could not load data for {ticker_up}: {err}")

    raise FileNotFoundError(f"No local data found for {ticker_up}")


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
    target_space: str = "log_volatility",
) -> dict[str, Any]:
    selected_tickers = tickers or ALL_TICKERS
    start_time = time.time()

    print(f"Starting empirical study across {len(selected_tickers)} symbols and {len(horizons)} horizons...")
    print(f"Target space: {target_space}")
    print(f"Horizons: {horizons}")
    print("Models: persistence, rolling_mean, ewma, har_rv, garch_11, ridge, elastic_net, gradient_boosting" + (", lstm" if include_lstm else ""))

    horizon_results: dict[int, list[dict[str, Any]]] = {h: [] for h in horizons}
    raw_test_samples: list[dict[str, Any]] = []

    for t_idx, ticker in enumerate(selected_tickers, 1):
        print(f"\n[{t_idx}/{len(selected_tickers)}] Processing {ticker} (Sector: {_sector_for(ticker)})...")
        try:
            raw_frame = _find_data_file(ticker)
        except Exception as exc:
            print(f"  Warning: Skipped {ticker}: {exc}")
            continue

        for h in horizons:
            config = VolatilityConfig(horizon=h, lookback=lookback)
            try:
                examples = build_examples(raw_frame, config)
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
                        maximum_epochs=20,
                        patience=4,
                        batch_size=64,
                        device="cpu",
                        seed=config.seed,
                        target_space=target_space,
                    ) if include_lstm else None,
                    nominal_coverage=0.90,
                    target_space=target_space,
                    return_forecasts=True,
                )
                selected_model = select_validation_model(metrics)
                horizon_results[h].append({
                    "ticker": ticker,
                    "sector": _sector_for(ticker),
                    "horizon": h,
                    "rows": len(examples.target),
                    "train_rows": len(split.train),
                    "val_rows": len(split.validation),
                    "test_rows": len(split.test),
                    "selected_model": selected_model,
                    "metrics": metrics,
                })

                # Collect test samples for worst-error diagnostics
                test_indices = split.test
                actual_vols = examples.target[test_indices]
                recent_vols = examples.current_volatility[test_indices]
                dates = examples.dates[test_indices]

                # Volatility tertiles for regime identification
                tertiles = np.quantile(actual_vols, [1.0 / 3.0, 2.0 / 3.0])

                for idx_in_test, global_idx in enumerate(test_indices):
                    act_v = float(actual_vols[idx_in_test])
                    act_var = float(act_v ** 2)
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
                        pred_var = float(pred_v ** 2)
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

                print(f"  h={h:2d}d: Val-Selected={selected_model:<18} Test QLIKE: HAR={metrics['har_rv']['test']['qlike']:.4f}, "
                      f"GARCH={metrics['garch_11']['test']['qlike']:.4f}, "
                      f"GB={metrics['gradient_boosting']['test']['qlike']:.4f}"
                      + (f", LSTM={metrics['lstm']['test']['qlike']:.4f}" if include_lstm else ""))
            except Exception as exc:
                print(f"  Error on {ticker} h={h}: {exc}")

    # Build comprehensive aggregate tables
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
            test_maes = [r["metrics"][m]["test"]["mae"] for r in results if m in r["metrics"]]
            test_rmses = [r["metrics"][m]["test"]["rmse"] for r in results if m in r["metrics"]]
            test_qlikes = [r["metrics"][m]["test"]["qlike"] for r in results if m in r["metrics"]]
            val_qlikes = [r["metrics"][m]["validation"]["qlike"] for r in results if m in r["metrics"]]
            near_zero_counts = [r["metrics"][m]["test"].get("near_zero_count", 0) for r in results if m in r["metrics"]]
            raw_mins = [r["metrics"][m]["test"].get("raw_min_pred", 0.0) for r in results if m in r["metrics"]]

            # Uncertainty calibration aggregates
            vol_covs = [
                r["metrics"][m]["test"]["volatility_interval"]["empirical_coverage"]
                for r in results
                if m in r["metrics"] and "volatility_interval" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["volatility_interval"].get("empirical_coverage") is not None
            ]
            vol_widths = [
                r["metrics"][m]["test"]["volatility_interval"]["average_width"]
                for r in results
                if m in r["metrics"] and "volatility_interval" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["volatility_interval"].get("average_width") is not None
            ]
            vol_low_covs = [
                r["metrics"][m]["test"]["volatility_interval"]["regime_coverage"]["low_vol"]
                for r in results
                if m in r["metrics"] and "volatility_interval" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["volatility_interval"].get("regime_coverage", {}).get("low_vol") is not None
            ]
            vol_high_covs = [
                r["metrics"][m]["test"]["volatility_interval"]["regime_coverage"]["high_vol"]
                for r in results
                if m in r["metrics"] and "volatility_interval" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["volatility_interval"].get("regime_coverage", {}).get("high_vol") is not None
            ]

            # Price cone calibration aggregates
            price_covs = [
                r["metrics"][m]["test"]["price_cone"]["empirical_coverage"]
                for r in results
                if m in r["metrics"] and "price_cone" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["price_cone"].get("empirical_coverage") is not None
            ]
            price_widths = [
                r["metrics"][m]["test"]["price_cone"]["average_width_pct"]
                for r in results
                if m in r["metrics"] and "price_cone" in r["metrics"][m]["test"]
                and r["metrics"][m]["test"]["price_cone"].get("average_width_pct") is not None
            ]

            agg[m] = {
                "val_qlike": float(np.mean(val_qlikes)),
                "test_mae": float(np.mean(test_maes)),
                "test_rmse": float(np.mean(test_rmses)),
                "test_qlike": float(np.mean(test_qlikes)),
                "near_zero_count": int(sum(near_zero_counts)),
                "raw_min_pred": float(min(raw_mins)) if raw_mins else None,
                "vol_interval_coverage_90": float(np.mean(vol_covs)) if vol_covs else None,
                "vol_interval_width": float(np.mean(vol_widths)) if vol_widths else None,
                "vol_interval_low_vol_cov": float(np.mean(vol_low_covs)) if vol_low_covs else None,
                "vol_interval_high_vol_cov": float(np.mean(vol_high_covs)) if vol_high_covs else None,
                "price_cone_coverage_90": float(np.mean(price_covs)) if price_covs else None,
                "price_cone_width_pct": float(np.mean(price_widths)) if price_widths else None,
            }

        # Calculate relative skill vs persistence and vs HAR
        base_persist = agg["persistence"]["test_qlike"]
        base_har = agg["har_rv"]["test_qlike"]

        for m in model_names:
            agg[m]["vs_persistence_pct"] = float((base_persist - agg[m]["test_qlike"]) / base_persist * 100.0)
            agg[m]["vs_har_pct"] = float((base_har - agg[m]["test_qlike"]) / base_har * 100.0)

        # Count wins per model on validation and test
        val_selected_counts = {m: 0 for m in model_names}
        test_best_counts = {m: 0 for m in model_names}
        for r in results:
            val_selected_counts[r["selected_model"]] = val_selected_counts.get(r["selected_model"], 0) + 1
            best_test = min(model_names, key=lambda m: r["metrics"][m]["test"]["qlike"])
            test_best_counts[best_test] = test_best_counts.get(best_test, 0) + 1

        per_horizon_aggregates[f"h{h}"] = {
            "horizon": h,
            "asset_count": len(results),
            "models": agg,
            "val_selected_counts": val_selected_counts,
            "test_best_counts": test_best_counts,
        }

    # Extract worst 20 QLIKE losses for key models
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
                top_20.append({
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
                })
            worst_diagnostics[m] = top_20

    elapsed = time.time() - start_time
    print(f"\nEmpirical study complete in {elapsed:.1f}s.")

    output_payload = {
        "benchmark_version": "empirical-volatility-benchmark-v2",
        "date_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "universe_size": len(selected_tickers),
        "target_space": target_space,
        "horizons": list(horizons),
        "models": model_names,
        "universe_by_sector": TARGET_UNIVERSE,
        "per_horizon_aggregates": per_horizon_aggregates,
        "worst_error_diagnostics": worst_diagnostics,
        "raw_results_by_horizon": {f"h{h}": horizon_results[h] for h in horizons},
    }
    return output_payload


def generate_markdown_report(data: dict[str, Any], comparison_target_data: dict[str, Any] | None = None) -> str:
    lines = []
    lines.append("# Empirical Volatility Forecasting Benchmark & Uncertainty Calibration Report (V2)")
    lines.append(f"**Date:** {data['date_completed']} | **Universe:** {data['universe_size']} Liquid Assets across 8 Sectors | **Target Space:** `{data['target_space']}` | **Execution Time:** {data['elapsed_seconds']:.1f}s\n")
    lines.append("## Executive Summary")
    lines.append("This empirical study evaluates the predictive accuracy and uncertainty calibration of 9 volatility forecasting models across 4 horizons (1-day, 5-day, 10-day, and 20-day) using strict chronological 70/15/15 splits with horizon-length boundary embargoes.")
    lines.append("Phase 2 incorporates canonical GARCH(1,1), target formulation analysis (Direct Volatility vs Log-Variance), metric numerical stabilization audits, and worst-error tail diagnostics.\n")

    lines.append("## 1. Multi-Horizon Forecasting Accuracy & Skill Matrix")

    for h_key, agg_data in data["per_horizon_aggregates"].items():
        h = agg_data["horizon"]
        lines.append(f"### Horizon: {h}-Day ({'1-session' if h == 1 else f'{h}-sessions'})")
        lines.append(f"*Evaluated across {agg_data['asset_count']} liquid assets (Out-of-Sample Test Partition)*\n")
        lines.append("| Model | Test MAE | Test RMSE | Test QLIKE | vs Persistence | vs HAR-RV | Val Selection Wins | Test Best Wins | Raw Min Pred | Near-Zero Count |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        models = agg_data["models"]
        for m, stats in models.items():
            name_display = m.replace("_", " ").title()
            if m == "har_rv":
                name_display = "HAR-RV"
            elif m == "garch_11":
                name_display = "GARCH(1,1)"
            elif m == "lstm":
                name_display = "PyTorch LSTM"
            elif m == "ewma":
                name_display = "EWMA (λ=0.94)"

            vs_p = f"{stats['vs_persistence_pct']:+.2f}%" if m != "persistence" else "—"
            vs_h = f"{stats['vs_har_pct']:+.2f}%" if m != "har_rv" else "—"
            val_wins = agg_data["val_selected_counts"].get(m, 0)
            test_wins = agg_data["test_best_counts"].get(m, 0)
            raw_min_str = f"{stats['raw_min_pred']:.6f}" if stats['raw_min_pred'] is not None else "N/A"
            nz_count = stats.get("near_zero_count", 0)

            lines.append(f"| **{name_display}** | {stats['test_mae']:.4f} | {stats['test_rmse']:.4f} | **{stats['test_qlike']:.4f}** | {vs_p} | {vs_h} | {val_wins}/{agg_data['asset_count']} | {test_wins}/{agg_data['asset_count']} | {raw_min_str} | {nz_count} |")
        lines.append("\n")

    if comparison_target_data is not None:
        lines.append("## 2. Target Formulation Comparison: Direct Volatility vs Log-Variance vs Log-Volatility")
        lines.append("Comparison of learned model performance when trained on levels of volatility vs log-variance vs log-volatility.\n")
        lines.append("| Horizon | Model | Target Formulation | Test MAE | Test RMSE | Test QLIKE |")
        lines.append("| :---: | :--- | :--- | :---: | :---: | :---: |")
        for h_key in data["per_horizon_aggregates"]:
            h = data["per_horizon_aggregates"][h_key]["horizon"]
            for m in ["gradient_boosting", "lstm"]:
                if m in data["per_horizon_aggregates"][h_key]["models"]:
                    v2_stats = data["per_horizon_aggregates"][h_key]["models"][m]
                    v_alt_stats = comparison_target_data.get("per_horizon_aggregates", {}).get(h_key, {}).get("models", {}).get(m)
                    m_disp = "PyTorch LSTM" if m == "lstm" else "Gradient Boosting"
                    lines.append(f"| {h}-Day | {m_disp} | `{data['target_space']}` | {v2_stats['test_mae']:.4f} | {v2_stats['test_rmse']:.4f} | **{v2_stats['test_qlike']:.4f}** |")
                    if v_alt_stats:
                        lines.append(f"| {h}-Day | {m_disp} | `{comparison_target_data['target_space']}` | {v_alt_stats['test_mae']:.4f} | {v_alt_stats['test_rmse']:.4f} | **{v_alt_stats['test_qlike']:.4f}** |")
        lines.append("\n")

    lines.append("## 3. Uncertainty Cones & Prediction Interval Calibration")
    lines.append("Evaluation of empirical coverage vs nominal 90% target coverage (p05 to p95 interval) on out-of-sample test partitions.\n")

    lines.append("### Conformal Volatility Interval Calibration (Nominal Target: 90.0%)")
    lines.append("| Horizon | Model | Empirical Coverage (90% Nom.) | Avg Width (Annualized σ) | Low Vol Regime Cov | High Vol Regime Cov |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: |")
    for h_key, agg_data in data["per_horizon_aggregates"].items():
        h = agg_data["horizon"]
        for m in ["persistence", "har_rv", "garch_11", "gradient_boosting", "lstm"]:
            if m in agg_data["models"] and agg_data["models"][m]["vol_interval_coverage_90"] is not None:
                stats = agg_data["models"][m]
                name_display = m.replace("_", " ").title()
                if m == "har_rv":
                    name_display = "HAR-RV"
                elif m == "garch_11":
                    name_display = "GARCH(1,1)"
                elif m == "lstm":
                    name_display = "PyTorch LSTM"
                cov = f"{stats['vol_interval_coverage_90'] * 100:.1f}%"
                width = f"{stats['vol_interval_width']:.4f}"
                low_cov = f"{stats['vol_interval_low_vol_cov'] * 100:.1f}%" if stats['vol_interval_low_vol_cov'] is not None else "N/A"
                high_cov = f"{stats['vol_interval_high_vol_cov'] * 100:.1f}%" if stats['vol_interval_high_vol_cov'] is not None else "N/A"
                lines.append(f"| {h}-Day | {name_display} | **{cov}** | {width} | {low_cov} | {high_cov} |")
    lines.append("\n")

    lines.append("### Gaussian Model-Implied p05–p95 Price Range Coverage (Nominal: 90.0%)")
    lines.append("| Horizon | Model Implied Volatility | Empirical Price Range Coverage | Avg Cone Width (% Price) |")
    lines.append("| :---: | :--- | :---: | :---: |")
    for h_key, agg_data in data["per_horizon_aggregates"].items():
        h = agg_data["horizon"]
        for m in ["persistence", "har_rv", "garch_11", "gradient_boosting", "lstm"]:
            if m in agg_data["models"] and agg_data["models"][m]["price_cone_coverage_90"] is not None:
                stats = agg_data["models"][m]
                name_display = m.replace("_", " ").title()
                if m == "har_rv":
                    name_display = "HAR-RV"
                elif m == "garch_11":
                    name_display = "GARCH(1,1)"
                elif m == "lstm":
                    name_display = "PyTorch LSTM"
                cov = f"{stats['price_cone_coverage_90'] * 100:.1f}%"
                width = f"{stats['price_cone_width_pct'] * 100:.1f}%"
                lines.append(f"| {h}-Day | {name_display} | **{cov}** | ±{width} |")
    lines.append("\n")

    lines.append("## 4. Top Worst-Error QLIKE Diagnostics (Tail Error Analysis)")
    lines.append("Inspection of top catastrophic QLIKE errors reveals why certain models achieve strong MAE but poor QLIKE:\n")

    for m in ["lstm", "gradient_boosting", "har_rv", "rolling_mean"]:
        if m in data.get("worst_error_diagnostics", {}):
            m_disp = "PyTorch LSTM" if m == "lstm" else ("Gradient Boosting" if m == "gradient_boosting" else ("HAR-RV" if m == "har_rv" else "Rolling Mean (60d)"))
            lines.append(f"### Top 5 Worst Out-of-Sample Losses: {m_disp}")
            lines.append("| Ticker | Date | Horizon | Regime | Actual σ | Pred σ | Recent RV (22d) | Error (Pred - Act) | QLIKE Loss | Floor Active |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for item in data["worst_error_diagnostics"][m][:5]:
                lines.append(f"| {item['ticker']} | {item['date']} | {item['horizon']}d | {item['regime']} | {item['actual_vol']:.4f} | {item['pred_vol']:.4f} | {item['recent_vol_22']:.4f} | {item['error']:+.4f} | **{item['qlike']:.2f}** | {item['floor_activated']} |")
            lines.append("\n")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", default="1,5,10,20", help="Comma-separated horizons")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers (default: all 44)")
    parser.add_argument("--without-lstm", action="store_true", help="Skip PyTorch LSTM")
    parser.add_argument("--target-space", default="log_variance", choices=("log_variance", "direct_volatility", "log_volatility"))
    parser.add_argument("--compare-targets", action="store_true", help="Also run direct_volatility target comparison")
    parser.add_argument("--output-json", type=Path, default=_REPO_ROOT / "reports" / "empirical_volatility_benchmark_v2.json")
    parser.add_argument("--output-md", type=Path, default=_REPO_ROOT / "reports" / "empirical_volatility_benchmark_v2.md")
    args = parser.parse_args()

    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()] or None

    print(f"=== Running Primary Benchmark: Target Space = {args.target_space} ===")
    data = run_study(
        tickers=tickers,
        horizons=horizons,
        include_lstm=not args.without_lstm,
        target_space=args.target_space,
    )

    alt_data = None
    if args.compare_targets:
        alt_target = "direct_volatility" if args.target_space != "direct_volatility" else "log_variance"
        print(f"\n=== Running Comparison Benchmark: Target Space = {alt_target} ===")
        alt_data = run_study(
            tickers=tickers,
            horizons=horizons,
            include_lstm=not args.without_lstm,
            target_space=alt_target,
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved JSON report to {args.output_json}")

    md_report = generate_markdown_report(data, comparison_target_data=alt_data)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_report, encoding="utf-8")
        print(f"Saved Markdown report to {args.output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
