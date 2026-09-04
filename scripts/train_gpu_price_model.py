"""Train the five-ticker seven-session price LSTM on the local NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.price_forecasting import (  # noqa: E402
    DEFAULT_TICKERS,
    PriceTrainingConfig,
    build_global_price_dataset,
    train_cuda_price_model,
)
from research.price_forecasting.gpu_pipeline import TRI_EXCHANGE_TICKERS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        choices=["tri_exchange", "five_ticker", "custom"],
        default="tri_exchange",
        help="Target universe: tri_exchange (30 LSE/NASDAQ/NYSE), five_ticker (US), or custom",
    )
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker list override")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory containing per-ticker .parquet cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tri_exchange_gpu_v1",
        help="Target directory to save model checkpoint and report",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--feature-mode",
        choices=["price_only", "price_plus_news", "compare_ablation"],
        default="price_only",
        help="Feature mode: price_only, price_plus_news, or compare_ablation",
    )
    parser.add_argument(
        "--news-dir",
        type=Path,
        default=REPO_ROOT / "data" / "news" / "alpaca",
        help="Directory containing per-ticker .jsonl news archives",
    )
    return parser.parse_args()


def _format_ablation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GPU Price Forecasting: News Signal Ablation Report",
        "",
        f"- **Model Architecture:** `{summary['model_version']}`",
        f"- **Device:** {summary['device']}",
        f"- **Tickers:** {', '.join(summary['tickers'])}",
        f"- **Test Rows (Untouched 15%):** {summary['rows']['test']}",
        "",
        "## 1. Pooled Test Performance Comparison",
        "",
        "| Metric | Baseline (Price-Only) | Challenger (+News) | Absolute Δ | Relative Δ |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    base_pool = summary["baseline"]["untouched_test"]["pooled"]
    chal_pool = summary["challenger"]["untouched_test"]["pooled"]
    delta_pool = summary["delta"]["pooled"]

    mae_rel = delta_pool["mae_percent_delta_relative"] * 100.0
    lines.append(
        f"| MAE (%) | {base_pool['mae_percent']:.4f}% | {chal_pool['mae_percent']:.4f}% | {delta_pool['mae_percent_delta']:+.4f}% | {mae_rel:+.2f}% |"
    )
    rmse_rel = delta_pool["rmse_percent_delta_relative"] * 100.0
    lines.append(
        f"| RMSE (%) | {base_pool['rmse_percent']:.4f}% | {chal_pool['rmse_percent']:.4f}% | {delta_pool['rmse_percent_delta']:+.4f}% | {rmse_rel:+.2f}% |"
    )
    dir_acc_delta = delta_pool["direction_accuracy_delta"] * 100.0
    lines.append(
        f"| Direction Accuracy | {base_pool['direction_accuracy'] * 100:.2f}% | {chal_pool['direction_accuracy'] * 100:.2f}% | {dir_acc_delta:+.2f}% | — |"
    )
    lines.append(
        f"| Rel MAE vs Persistence | {base_pool['relative_mae_vs_persistence']:.4f}× | {chal_pool['relative_mae_vs_persistence']:.4f}× | {chal_pool['relative_mae_vs_persistence'] - base_pool['relative_mae_vs_persistence']:+.4f}× | — |"
    )

    lines.extend(
        [
            "",
            "## 2. Per-Horizon Test Breakdown (Day 1 to 7)",
            "",
            "| Horizon | Base MAE | +News MAE | Δ MAE | Base DirAcc | +News DirAcc | Δ DirAcc |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for h in range(7):
        b_h = base_pool["per_horizon"][h]
        c_h = chal_pool["per_horizon"][h]
        d_mae = c_h["mae_percent"] - b_h["mae_percent"]
        d_dir = (c_h["direction_accuracy"] - b_h["direction_accuracy"]) * 100.0
        lines.append(
            f"| Day {h + 1} | {b_h['mae_percent']:.4f}% | {c_h['mae_percent']:.4f}% | {d_mae:+.4f}% | {b_h['direction_accuracy'] * 100:.1f}% | {c_h['direction_accuracy'] * 100:.1f}% | {d_dir:+.1f}% |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Ticker Test Summary",
            "",
            "| Ticker | Base MAE | +News MAE | Base DirAcc | +News DirAcc |",
            "| :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for ticker in summary["tickers"]:
        b_t = summary["baseline"]["untouched_test"]["per_ticker"][ticker]
        c_t = summary["challenger"]["untouched_test"]["per_ticker"][ticker]
        lines.append(
            f"| {ticker} | {b_t['mae_percent']:.4f}% | {c_t['mae_percent']:.4f}% | {b_t['direction_accuracy'] * 100:.1f}% | {c_t['direction_accuracy'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## 4. Empirical Verdict",
            "",
            f"- **Verdict:** {summary['verdict']}",
            f"- **Rationale:** {summary['rationale']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_ablation(
    frames: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("=== RUNNING PRICE-ONLY BASELINE ===", flush=True)
    base_config = PriceTrainingConfig(
        maximum_epochs=args.epochs,
        patience=args.patience,
        feature_mode="price_only",
    )
    base_dataset = build_global_price_dataset(frames, base_config, feature_mode="price_only")
    base_dir = args.output_dir / "baseline_price_only"
    base_report = train_cuda_price_model(base_dataset, base_dir, base_config)

    print("\n=== RUNNING PRICE + NEWS CHALLENGER ===", flush=True)
    chal_config = PriceTrainingConfig(
        maximum_epochs=args.epochs,
        patience=args.patience,
        feature_mode="price_plus_news",
    )
    chal_dataset = build_global_price_dataset(
        frames,
        chal_config,
        feature_mode="price_plus_news",
        news_archives=args.news_dir,
    )
    chal_dir = args.output_dir / "challenger_price_plus_news"
    chal_report = train_cuda_price_model(chal_dataset, chal_dir, chal_config)

    base_pooled = base_report["untouched_test"]["pooled"]
    chal_pooled = chal_report["untouched_test"]["pooled"]

    mae_delta = chal_pooled["mae_percent"] - base_pooled["mae_percent"]
    mae_rel_delta = (
        mae_delta / base_pooled["mae_percent"] if base_pooled["mae_percent"] > 0 else 0.0
    )
    rmse_delta = chal_pooled["rmse_percent"] - base_pooled["rmse_percent"]
    rmse_rel_delta = (
        rmse_delta / base_pooled["rmse_percent"] if base_pooled["rmse_percent"] > 0 else 0.0
    )
    dir_acc_delta = chal_pooled["direction_accuracy"] - base_pooled["direction_accuracy"]

    if mae_delta < -0.05 and dir_acc_delta > 0.01:
        verdict = "NEWS_IMPROVES_ACCURACY"
        rationale = "News feature integration demonstrated clear error reduction and improved directional accuracy on untouched test data."
    elif abs(mae_delta) <= 0.05 and abs(dir_acc_delta) <= 0.01:
        verdict = "NEUTRAL_SIGNAL"
        rationale = "News feature integration yielded comparable performance to price-only features without statistically significant divergence."
    elif mae_delta > 0:
        verdict = "PRICE_ONLY_SUPERIOR"
        rationale = "Adding news features introduced variance or slight degradation over the price-only model on the untouched test partition."
    else:
        verdict = "MARGINAL_DIFFERENCE"
        rationale = f"News feature delta: MAE delta {mae_delta:+.4f}%, Direction Accuracy delta {dir_acc_delta * 100:+.2f}%."

    ablation_summary = {
        "model_version": chal_report["model_version"],
        "device": chal_report["device"],
        "tickers": list(frames.keys()),
        "rows": base_report["rows"],
        "baseline": base_report,
        "challenger": chal_report,
        "delta": {
            "pooled": {
                "mae_percent_delta": mae_delta,
                "mae_percent_delta_relative": mae_rel_delta,
                "rmse_percent_delta": rmse_delta,
                "rmse_percent_delta_relative": rmse_rel_delta,
                "direction_accuracy_delta": dir_acc_delta,
            }
        },
        "verdict": verdict,
        "rationale": rationale,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_summary.json").write_text(
        json.dumps(ablation_summary, indent=2), encoding="utf-8"
    )
    md_content = _format_ablation_markdown(ablation_summary)
    (args.output_dir / "ablation_summary.md").write_text(md_content, encoding="utf-8")
    print(f"\nAblation reports written to {args.output_dir}")
    print(md_content)
    return ablation_summary


def main() -> int:
    args = parse_args()
    if args.tickers:
        tickers = tuple(value.strip().upper() for value in args.tickers.split(",") if value.strip())
        cache_dir = args.cache_dir or (REPO_ROOT / "data" / "tri_exchange" / "cache")
    elif args.universe == "tri_exchange":
        tickers = TRI_EXCHANGE_TICKERS
        cache_dir = args.cache_dir or (REPO_ROOT / "data" / "tri_exchange" / "cache")
    elif args.universe == "five_ticker":
        tickers = DEFAULT_TICKERS
        cache_dir = args.cache_dir or (REPO_ROOT / "data" / "ndx100" / "cache")
    else:
        tickers = TRI_EXCHANGE_TICKERS
        cache_dir = args.cache_dir or (REPO_ROOT / "data" / "tri_exchange" / "cache")

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker}.parquet"
        if path.is_file():
            frames[ticker] = pd.read_parquet(path)
        else:
            # Fallback to downloading and caching
            from backend.data_pipeline import _download_ohlcv
            from research.price_forecasting.gpu_pipeline import _normalise_ohlcv

            df = _download_ohlcv(ticker)
            norm = _normalise_ohlcv(df)
            cache_dir.mkdir(parents=True, exist_ok=True)
            norm.to_parquet(path)
            frames[ticker] = norm

    if args.feature_mode == "compare_ablation":
        run_ablation(frames, args)
        return 0

    config = PriceTrainingConfig(
        maximum_epochs=args.epochs,
        patience=args.patience,
        feature_mode=args.feature_mode,
    )
    dataset = build_global_price_dataset(
        frames,
        config,
        feature_mode=args.feature_mode,
        news_archives=args.news_dir if args.feature_mode == "price_plus_news" else None,
    )
    print(
        f"dataset={len(dataset.sequences)} train={len(dataset.split_train)} "
        f"validation={len(dataset.split_validation)} test={len(dataset.split_test)} "
        f"features={len(dataset.feature_names)} mode={dataset.feature_mode}",
        flush=True,
    )
    report = train_cuda_price_model(dataset, args.output_dir, config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": report["device"],
                "epochs": report["selection"]["epochs"],
                "feature_mode": report["feature_mode"],
                "feature_count": report["feature_count"],
                "test": report["untouched_test"]["pooled"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
