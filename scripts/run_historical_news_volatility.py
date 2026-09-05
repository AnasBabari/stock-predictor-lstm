"""Resumable five-stock news acquisition followed by matched validation-only volatility fits."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.price_forecasting.gpu_pipeline import (  # noqa: E402
    FEATURE_NAMES,
    _normalise_ohlcv,
    build_price_features,
)  # noqa: E402
from research.price_forecasting.news_archive import (  # noqa: E402
    NEWS_FEATURE_NAMES,
    apply_revision_policy,
    build_causal_news_features,
    collect_alpaca_news,
    merge_news_archive,
)
from research.price_forecasting.paired_validation import hac_mean  # noqa: E402

TICKERS = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN")
EVENTS = {
    "earnings": r"earnings|quarterly results",
    "guidance": r"guidance|outlook",
    "analyst_rating": r"price target|upgrade|downgrade",
    "merger_acquisition": r"merger|acquisition|acquire",
    "regulatory": r"regulator|fda|\bsec\b",
    "litigation": r"lawsuit|litigation|sues",
    "management": r"\bceo\b|resigns",
    "product": r"launch|product",
    "capital_raise": r"offering|capital raise",
    "buyback": r"buyback|repurchase",
    "dividend": r"dividend",
    "macro": r"inflation|federal reserve",
    "geopolitical": r"war|sanction",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def canonicalize(records):
    eligible = []
    missing_updates = 0
    for item in records:
        # Do not treat missing/broken provider updates as proof of historical availability.
        updated = pd.to_datetime(item.get("provider_updated_at"), utc=True, errors="coerce")
        if pd.isna(updated):
            missing_updates += 1
            continue
        eligible.append(item)
    gated, diagnostics = apply_revision_policy(eligible)
    seen, output = {}, []
    for item in sorted(gated, key=lambda r: r["t_available"]):
        title = " ".join(re.sub(r"[^\w\s]", " ", item["headline"].lower()).split())
        bucket = int(pd.Timestamp(item["published_at"]).timestamp() // 1800)
        key = (item["ticker"], bucket, title)
        if key in seen:
            prior = output[seen[key]]
            prior["duplicate_providers"] = sorted(
                set(prior["duplicate_providers"]) | {item["provider"]}
            )
            continue
        seen[key] = len(output)
        row = dict(
            item,
            schema_version="news-event-v2",
            event_type=next((k for k, v in EVENTS.items() if re.search(v, title)), "other"),
            source_class="provider_news",
            relevance_score=None,
            duplicate_providers=[item["provider"]],
        )
        output.append(row)
    return output, {
        **diagnostics,
        "missing_updates": missing_updates,
        "duplicate_count": len(gated) - len(output),
        "unique_kept": len(output),
    }


def fit_and_score(frames, output, estimator="ridge", split=None):
    reports, predictions = [], []
    for horizon in (5, 10, 20):
        pieces = []
        for ticker, (prices, features) in frames.items():
            returns = np.log(prices.Close / prices.Close.shift(1))
            # Mean future daily squared log returns: variance target, no demean.
            target = pd.concat(
                [returns.shift(-k).pow(2) for k in range(1, horizon + 1)], axis=1
            ).mean(axis=1, skipna=False)
            part = features.copy()
            part["y"] = target
            part["label_end"] = pd.Series(prices.index, index=prices.index).shift(-horizon)
            part["date"], part["ticker"] = part.index, ticker
            part["rolling"] = returns.pow(2).rolling(60).mean()
            part["frozen_rolling"] = returns.rolling(60).var(ddof=1)
            pieces.append(part.dropna())
        table = pd.concat(pieces, ignore_index=True)
        train = (
            (table.date >= "2019-01-01")
            & (table.date < "2023-01-01")
            & (table.label_end < "2023-01-01")
        )
        val = (
            (table.date >= "2023-01-01")
            & (table.date < "2025-01-01")
            & (table.label_end < "2025-01-01")
        )
        if split is not None:
            train = (
                (table.date >= split["start"])
                & (table.date < split["train_end"])
                & (table.label_end < split["train_end"])
            )
            val = (
                (table.date >= split["score_start"])
                & (table.date < split["score_end"])
                & (table.label_end < split["score_end"])
            )
        if not train.any() or not val.any():
            raise ValueError("Empty partition")
        actual = table.loc[val, "y"].to_numpy()
        selected = table.loc[val, ["date", "ticker", "y"]].copy()
        selected["horizon"] = horizon
        for arm, columns in {
            "rolling": [],
            "frozen_rolling": [],
            "A_ohlcv": list(FEATURE_NAMES),
            "B_activity": list(FEATURE_NAMES)
            + list(NEWS_FEATURE_NAMES[:3])
            + ["news_hours_since_latest_article"],
            "C_sentiment": list(FEATURE_NAMES) + list(NEWS_FEATURE_NAMES),
        }.items():
            if columns:
                scaler = StandardScaler().fit(table.loc[train, columns])
                if estimator == "xgboost_cuda":
                    from xgboost import XGBRegressor

                    model = XGBRegressor(
                        device="cuda",
                        tree_method="hist",
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.03,
                        reg_lambda=10,
                        min_child_weight=20,
                        subsample=1,
                        colsample_bytree=1,
                        random_state=42,
                        n_jobs=4,
                    )
                else:
                    model = Ridge(alpha=100)
                print(f"training estimator={estimator} horizon={horizon} arm={arm}", flush=True)
                model.fit(
                    scaler.transform(table.loc[train, columns]),
                    np.log(np.maximum(table.loc[train, "y"], 1e-12)),
                )
                forecast = np.exp(
                    np.clip(model.predict(scaler.transform(table.loc[val, columns])), -27, 5)
                )
                if estimator == "xgboost_cuda":
                    runtime = json.loads(model.get_booster().save_config())
                    if not runtime["learner"]["generic_param"]["device"].startswith("cuda"):
                        raise RuntimeError("Requested GPU training fell back to CPU")
                    write_json(output / f"runtime_h{horizon}_{arm}.json", runtime)
                    model.save_model(output / f"weights_h{horizon}_{arm}.ubj")
                write_json(
                    output / f"model_h{horizon}_{arm}.json",
                    {
                        "features": columns,
                        "estimator": estimator,
                        "coef": model.coef_.tolist() if estimator == "ridge" else None,
                        "intercept": float(model.intercept_) if estimator == "ridge" else None,
                        "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(),
                    },
                )
            else:
                forecast = table.loc[val, arm].to_numpy()
            forecast = np.maximum(forecast, 1e-12)
            ratio = np.maximum(actual, 1e-12) / forecast
            loss = ratio - np.log(ratio) - 1
            selected[arm + "_variance"] = forecast
            selected[arm + "_qlike"] = loss
            vol_error = np.sqrt(252 * forecast) - np.sqrt(252 * actual)
            reports.append(
                {
                    "horizon": horizon,
                    "arm": arm,
                    "train_rows": int(train.sum()),
                    (
                        "test_rows" if split and split["partition"] == "test" else "validation_rows"
                    ): int(val.sum()),
                    "qlike": float(loss.mean()),
                    "annualized_vol_mae": float(abs(vol_error).mean()),
                    "annualized_vol_rmse": float(np.sqrt(np.mean(vol_error**2))),
                }
            )
        predictions.append(selected)
    rows = pd.concat(predictions, ignore_index=True)
    partition = split["partition"] if split else "validation"
    rows.to_parquet(output / f"{partition}_predictions.parquet", index=False)
    comparisons = []
    for horizon in (5, 10, 20):
        sub = rows[rows.horizon == horizon]
        for arm in ("B_activity", "C_sentiment"):
            daily = (sub.A_ohlcv_qlike - sub[arm + "_qlike"]).groupby(sub.date).mean()
            for lag in (horizon - 1, 2 * (horizon - 1), 3 * (horizon - 1)):
                comparisons.append(
                    {"horizon": horizon, "arm": arm, "lag": lag, **hac_mean(daily.to_numpy(), lag)}
                )
    write_json(
        output / "results.json",
        {
            "metrics": reports,
            "estimator": estimator,
            "paired_hac": comparisons,
            "test_scored": partition == "test",
            "evaluation_partition": partition,
            "split": split,
            "deployment": False,
            "status": "exploratory_split_complete" if split else "first_matched_baseline_complete",
            "limitations": [
                "Selected surviving stocks only; not a point-in-time universe",
                "Retrieved version availability relies on provider update semantics",
                "Exploratory unadjusted HAC",
                "SEC, GARCH, event, macro, FinBERT arms not included",
            ],
        },
    )
    print(json.dumps(reports), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--estimator", choices=("ridge", "xgboost_cuda"), default="ridge")
    p.add_argument(
        "--reuse-only",
        action="store_true",
        help="Fail on any missing chunk; never request credentials or network",
    )
    p.add_argument(
        "--all-us", action="store_true", help="Use the frozen 195-US-stock cache universe"
    )
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect chunks without building features or fitting",
    )
    args = p.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tickers = TICKERS
    if args.all_us:
        universe = json.loads(
            (
                ROOT / "artifacts/price_validation_comparison_20260905_003731/protocol.json"
            ).read_text()
        )
        tickers = tuple(sorted(t for t in universe["file_hashes"] if not t.endswith(".L")))
    write_json(
        args.output_dir / "protocol.json",
        {
            "tickers": tickers,
            "train": "2019-2022",
            "validation": "2023-2024",
            "test": "2025 onward NOT SCORED",
            "horizons": [5, 10, 20],
            "alpha": 100,
            "estimator": args.estimator,
            "gpu_configuration": {
                "trees": 200,
                "depth": 3,
                "learning_rate": 0.03,
                "reg_lambda": 10,
                "min_child_weight": 20,
                "seed": 42,
                "selection": "fixed before this run; no test access or hyperparameter search",
            }
            if args.estimator == "xgboost_cuda"
            else None,
            "coverage_rule": "Each ticker train and validation: >=100 retained articles and >=20% sessions with news in trailing 7 calendar days",
            "target": "mean future squared daily log returns; MAE/RMSE on sqrt(252*variance)",
            "collection_end_exclusive": "2026-09-05",
            "stage": "first five-stock matched linear experiment; later arms separate",
        },
    )
    key = (
        "" if args.reuse_only else (os.getenv("ALPACA_API_KEY_ID") or getpass.getpass("API key: "))
    )
    secret = (
        ""
        if args.reuse_only
        else (os.getenv("ALPACA_API_SECRET_KEY") or getpass.getpass("Secret: "))
    )
    boundaries = list(pd.date_range("2019-01-01", "2026-09-01", freq="QS")) + [
        pd.Timestamp("2026-09-05")
    ]
    frames, coverage = {}, []
    for ticker in tickers:
        combined = []
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            chunk = args.corpus_dir / f"{ticker}_{start.date()}.jsonl"
            if chunk.exists():
                manifest = json.loads(chunk.with_suffix(".manifest.json").read_text())
                if (
                    manifest.get("pagination_complete") is not True
                    or manifest.get("requested_start") != str(start)
                    or manifest.get("requested_end_exclusive") != str(end)
                ):
                    raise ValueError(f"Incomplete or mismatched chunk request: {chunk.name}")
                if hashlib.sha256(chunk.read_bytes()).hexdigest() != manifest["sha256"]:
                    raise ValueError("Chunk hash mismatch")
                records = [json.loads(line) for line in chunk.read_text().splitlines() if line]
            else:
                if args.reuse_only:
                    raise ValueError(f"Missing chunk in offline mode: {chunk.name}")
                records = collect_alpaca_news(
                    ticker,
                    key_id=key,
                    secret_key=secret,
                    start=start.isoformat() + "Z",
                    end=end.isoformat() + "Z",
                )
                manifest = merge_news_archive(chunk, records)
                manifest.update(
                    {
                        "requested_start": str(start),
                        "requested_end_exclusive": str(end),
                        "pagination_complete": True,
                    }
                )
                write_json(chunk.with_suffix(".manifest.json"), manifest)
                time.sleep(0.25)
            combined.extend(records)
            print(f"collected {ticker} {start.date()} records={len(records)}", flush=True)
        if args.collect_only:
            write_json(
                args.output_dir / f"{ticker}_collection.json",
                {
                    "ticker": ticker,
                    "records": len(combined),
                    "quarters_complete": len(boundaries) - 1,
                },
            )
            continue
        records, diag = canonicalize(combined)
        prices = _normalise_ohlcv(
            pd.read_parquet(ROOT / f"data/tri_exchange/cache/{ticker}.parquet")
        )
        features = build_price_features(prices)
        news = build_causal_news_features(features.index, ticker, records)
        features = features.join(news)
        for name, begin, end in (
            ("train", "2019-01-01", "2023-01-01"),
            ("validation", "2023-01-01", "2025-01-01"),
        ):
            mask = (features.index >= begin) & (features.index < end)
            n = sum(begin <= r["t_available"][:10] < end for r in records)
            fraction = float((news.loc[mask, "news_headline_count_7d"] > 0).mean())
            coverage.append(
                {
                    "ticker": ticker,
                    "partition": name,
                    "articles": n,
                    "sessions": int(mask.sum()),
                    "coverage_7d": fraction,
                    "passed": n >= 100 and fraction >= 0.2,
                }
            )
        write_json(args.output_dir / f"{ticker}_diagnostics.json", diag)
        frames[ticker] = (prices, features)
    if args.collect_only:
        write_json(
            args.output_dir / "collection_complete.json",
            {"tickers": len(tickers), "test_scored": False, "models_fit": 0},
        )
        return
    write_json(args.output_dir / "coverage.json", coverage)
    if not all(r["passed"] for r in coverage):
        raise ValueError("Coverage insufficient; retained complete collection, no fitting")
    fit_and_score(frames, args.output_dir, estimator=args.estimator)


if __name__ == "__main__":
    main()
