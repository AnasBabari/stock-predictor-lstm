"""Fixed seven-session cross-sectional Ridge experiment; development validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
)
from research.price_forecasting.paired_validation import hac_mean  # noqa: E402


def partitions(table):
    """Global date boundaries; labels must end strictly before the next block."""
    dates = np.sort(table.date.unique())
    validation_start, reserve_start = dates[int(len(dates) * 0.70)], dates[int(len(dates) * 0.85)]
    train = (table.date < validation_start) & (table.label_end < validation_start)
    validation = (
        (table.date >= validation_start)
        & (table.date < reserve_start)
        & (table.label_end < reserve_start)
    )
    return train, validation, str(validation_start), str(reserve_start)


def centered_rank(values):
    # Average ties; all-equal values map to zero (no arbitrary ticker ordering).
    return (values.rank(method="average") - (len(values) + 1) / 2) / len(values)


def daily_scores(table):
    records = []
    for (market, date), group in table.groupby(["market", "date"], sort=True):
        truth = centered_rank(group.target)
        record = {"market": market, "date": date, "assets": len(group)}
        for name in ("ridge", "momentum", "equal"):
            rank = centered_rank(group[name])
            record[name + "_ic"] = (
                float(rank.corr(truth)) if rank.std() > 0 and truth.std() > 0 else 0.0
            )
            record[name + "_rank_mae"] = float(abs(rank - truth).mean())
        records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.baseline_dir / "protocol.json").read_text())
    rows = []
    for ticker, digest in original["file_hashes"].items():
        path = ROOT / "data/tri_exchange/cache" / f"{ticker}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cache identity changed: {ticker}")
        frame = _normalise_ohlcv(pd.read_parquet(path))
        features = build_price_features(frame)
        close = frame.Close
        features["target"] = np.log(close.shift(-7) / close).reindex(features.index)
        features["label_end"] = (
            pd.Series(frame.index, index=frame.index).shift(-7).reindex(features.index)
        )
        features["date"] = features.index
        features["stock_id"] = ticker
        features["market"] = "UK" if ticker.endswith(".L") else "US"
        features = features.loc[frame.Volume.reindex(features.index) > 0]
        rows.append(features.dropna())
    table = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", "market", "stock_id"])
        .reset_index(drop=True)
    )
    if (
        table.duplicated(["date", "stock_id"]).any()
        or not np.isfinite(table[list(FEATURE_NAMES) + ["target"]]).all().all()
    ):
        raise ValueError("Duplicate or nonfinite input")
    # Enforce identical outcome endpoint within each market/date basket.
    groups = table.groupby(["market", "date"])
    eligible = (groups.stock_id.transform("size") >= 20) & (
        groups.label_end.transform("nunique") == 1
    )
    excluded_rows = int((~eligible).sum())
    table = table.loc[eligible].reset_index(drop=True)
    train, validation, val_start, reserve_start = partitions(table)
    if not train.any() or not validation.any():
        raise ValueError("Empty development partition")
    protocol = {
        "hypothesis": "25 causal features predict within-market seven-session return ranks",
        "alpha": 100.0,
        "horizon": 7,
        "minimum_assets": 20,
        "validation_start": val_start,
        "reserve_start": reserve_start,
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "excluded_basket_rows": excluded_rows,
        "feature_names": list(FEATURE_NAMES),
        "source_hashes": original["file_hashes"],
        "caveats": [
            "Current survivor basket, not point-in-time membership",
            "Previously explored historical cache; reserve is not claimed pristine",
            "No transaction costs or executable trading strategy",
            "Global chronological 70/15/15 date split with label-overlap purge",
            "No reserve scoring or production changes",
        ],
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    target_rank = table.loc[train].groupby(["market", "date"]).target.transform(centered_rank)
    columns = list(FEATURE_NAMES)
    scaler = StandardScaler().fit(table.loc[train, columns])
    model = Ridge(alpha=100.0, solver="cholesky").fit(
        scaler.transform(table.loc[train, columns]), target_rank
    )
    evaluated = table.loc[validation, ["stock_id", "market", "date", "label_end", "target"]].copy()
    evaluated["ridge"] = model.predict(scaler.transform(table.loc[validation, columns]))
    evaluated["momentum"] = table.loc[validation, "return_20d"]
    evaluated["equal"] = 0.0
    evaluated.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    daily = daily_scores(evaluated)
    daily.to_parquet(args.output_dir / "daily_scores.parquet", index=False)
    results = {}
    for scope in ("ALL", "US", "UK"):
        subset = daily if scope == "ALL" else daily[daily.market == scope]
        dated = subset.groupby("date").mean(numeric_only=True)
        comparisons = {}
        for reference in ("momentum", "equal"):
            difference = dated.ridge_ic - dated[reference + "_ic"]
            comparisons[reference] = [hac_mean(difference.to_numpy(), lag) for lag in (6, 12, 18)]
        results[scope] = {
            "dates": len(dated),
            "mean_metrics": dated.mean().to_dict(),
            "ridge_ic_improvement_hac": comparisons,
        }
    report = {
        "status": "validation_complete",
        "test_scored": False,
        "deployment": False,
        "results": results,
        "inference_note": "Exploratory unadjusted intervals; ALL equally averages markets present per date. IC zero denotes no ranking for tied predictions.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output_dir / "linear_model.json").write_text(
        json.dumps(
            {
                "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
        )
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
