"""Beta-residual cross-sectional rank experiment; development validation only.

Target: trailing-60d OLS beta (45-session gate, pairwise-complete sessions,
no forward fill into covariance) against SPY (US) / ^FTSE (UK, GBp native),
forward residual epsilon_{t->t+h} = R_{t->t+h} - beta_t * Rm_{t->t+h} for
h = 1..7. Outcomes require actual traded closes at both endpoints (holiday
masked, never forward-filled). Ridge(alpha=100) maps 25 causal price
features to residuals on train; date-level Spearman IC per horizon on
validation. Reserve never scored.
"""

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

HORIZONS = (1, 2, 3, 4, 5, 6, 7)
BETA_WINDOW = 60
MIN_BETA_SESSIONS = 45
MIN_ASSETS = 20


def partitions(dates):
    dates = np.sort(np.unique(dates))
    validation_start = dates[int(len(dates) * 0.70)]
    reserve_start = dates[int(len(dates) * 0.85)]
    return validation_start, reserve_start


def centered_rank(values):
    return (values.rank(method="average") - (len(values) + 1) / 2) / len(values)


def trailing_beta(asset_logret: pd.Series, bench_logret: pd.Series) -> pd.Series:
    """OLS beta over trailing 60 asset sessions on pairwise-complete data.

    No forward fill: benchmark is reindexed to asset sessions with NaN where
    untraded (holiday masking). Requires >=45 valid nonzero asset sessions
    and positive benchmark variance in-window.
    """
    pair = pd.DataFrame({"a": asset_logret, "m": bench_logret.reindex(asset_logret.index)})
    a, m = pair["a"], pair["m"]
    window = a.rolling(BETA_WINDOW, min_periods=BETA_WINDOW)
    cov = window.cov(m)
    var = m.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()
    n_nonzero = ((a.notna()) & (m.notna()) & (a != 0.0)).rolling(
        BETA_WINDOW, min_periods=BETA_WINDOW
    ).sum()
    beta = cov / var
    ok = n_nonzero.ge(MIN_BETA_SESSIONS) & var.gt(1e-12)
    return beta.where(ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.baseline_dir / "protocol.json").read_text())

    spy = pd.read_parquet(ROOT / "data/macro/market_dailies.parquet")["SPY"]
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    ftse = pd.read_parquet(ROOT / "data/macro/ftse_dailies.parquet")["FTSE"]
    ftse.index = pd.to_datetime(ftse.index).tz_localize(None)
    bench = {"US": np.log(spy / spy.shift(1)), "UK": np.log(ftse / ftse.shift(1))}
    bench_close = {"US": spy, "UK": ftse}

    rows = []
    for ticker, digest in original["file_hashes"].items():
        path = ROOT / "data/tri_exchange/cache" / f"{ticker}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cache identity changed: {ticker}")
        market = "UK" if ticker.endswith(".L") else "US"
        frame = _normalise_ohlcv(pd.read_parquet(path))
        features = build_price_features(frame)
        aret = np.log(frame.Close / frame.Close.shift(1))
        beta = trailing_beta(aret, bench[market])
        close, bclose = frame.Close, bench_close[market]
        base = features.copy()
        base["stock_id"] = ticker
        base["market"] = market
        base["beta"] = beta.reindex(features.index)
        base["date"] = base.index
        base = base.loc[frame.Volume.reindex(base.index) > 0]
        for h in HORIZONS:
            end_dates = pd.Series(frame.index, index=frame.index).shift(-h)
            fwd_asset = np.log(close.shift(-h) / close)
            # Benchmark span uses actual closes at the asset's endpoint dates only.
            b_t = bclose.reindex(base.index)
            b_end = bclose.reindex(end_dates.reindex(base.index))
            fwd_bench = np.log(b_end.to_numpy(float) / b_t.to_numpy(float))
            sub = base.copy()
            sub["horizon"] = h
            sub["label_end"] = end_dates.reindex(base.index).to_numpy()
            sub["residual"] = (
                fwd_asset.reindex(base.index).to_numpy(float)
                - sub["beta"].to_numpy(float) * fwd_bench
            )
            rows.append(sub)
    table = pd.concat(rows, ignore_index=True)
    n_before_beta = len(table)
    cols = list(FEATURE_NAMES) + ["residual", "beta"]
    table = table.dropna(subset=cols + ["label_end"])
    beta_gated_rows = int(n_before_beta - len(table))
    if (
        table.duplicated(["market", "date", "horizon", "stock_id"]).any()
        or not np.isfinite(table[list(FEATURE_NAMES) + ["residual"]].to_numpy(float)).all()
    ):
        raise ValueError("Duplicate or nonfinite input")
    # Eligibility per (market, date, horizon): >=20 assets, one shared endpoint.
    groups = table.groupby(["market", "date", "horizon"])
    eligible = (groups.stock_id.transform("size") >= MIN_ASSETS) & (
        groups.label_end.transform("nunique") == 1
    )
    excluded_rows = int((~eligible).sum())
    table = table.loc[eligible].reset_index(drop=True)
    val_start, reserve_start = partitions(table.date.to_numpy())
    train = (table.date < val_start) & (table.label_end < val_start)
    validation = (
        (table.date >= val_start) & (table.date < reserve_start) & (table.label_end < reserve_start)
    )
    if not train.any() or not validation.any():
        raise ValueError("Empty development partition")
    protocol = {
        "hypothesis": "25 causal features predict beta-residual ranks, h=1..7",
        "alpha": 100.0,
        "horizons": list(HORIZONS),
        "beta_window": BETA_WINDOW,
        "min_beta_sessions": MIN_BETA_SESSIONS,
        "benchmarks": {"US": "SPY", "UK": "^FTSE"},
        "minimum_assets": MIN_ASSETS,
        "validation_start": str(val_start),
        "reserve_start": str(reserve_start),
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "excluded_basket_rows": excluded_rows,
        "beta_or_outcome_gated_rows": beta_gated_rows,
        "feature_names": list(FEATURE_NAMES),
        "source_hashes": original["file_hashes"],
        "caveats": [
            "Current survivor basket, not point-in-time membership",
            "Beta estimated with error; estimation noise absorbs variance",
            "No transaction costs or executable trading strategy",
            "Global chronological 70/15/15 date split with label-overlap purge",
            "No reserve scoring or production changes",
        ],
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    columns = list(FEATURE_NAMES)
    scaler = StandardScaler().fit(table.loc[train, columns].to_numpy(float))
    models = {}
    for h in HORIZONS:
        mh = train & (table.horizon == h)
        models[h] = Ridge(alpha=100.0, solver="cholesky").fit(
            scaler.transform(table.loc[mh, columns].to_numpy(float)),
            table.loc[mh, "residual"].to_numpy(float),
        )
    evaluated = table.loc[
        validation, ["stock_id", "market", "date", "horizon", "label_end", "residual"]
    ].copy()
    evaluated["ridge"] = np.nan
    for h, model in models.items():
        mh = evaluated.horizon == h
        evaluated.loc[mh, "ridge"] = model.predict(
            scaler.transform(table.loc[validation & (table.horizon == h), columns].to_numpy(float))
        )
    evaluated["momentum"] = table.loc[validation, "return_20d"].to_numpy(float)
    evaluated["equal"] = 0.0
    evaluated.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)

    # Baskets: within-market primary + pooled ALL (residuals are excess returns).
    records = []
    for scope in ("within", "pooled"):
        grp_cols = ["date", "horizon"] if scope == "pooled" else ["market", "date", "horizon"]
        for keys, group in evaluated.groupby(grp_cols, sort=True):
            truth = centered_rank(group.residual)
            rec = {"scope": scope, "assets": len(group)}
            if scope == "pooled":
                rec.update({"market": "ALL", "date": keys[0], "horizon": keys[1]})
            else:
                rec.update({"market": keys[0], "date": keys[1], "horizon": keys[2]})
            for name in ("ridge", "momentum", "equal"):
                rank = centered_rank(group[name])
                rec[name + "_ic"] = (
                    float(rank.corr(truth)) if rank.std() > 0 and truth.std() > 0 else 0.0
                )
            records.append(rec)
    daily = pd.DataFrame(records)
    daily.to_parquet(args.output_dir / "daily_scores.parquet", index=False)

    results = {}
    for scope in ("within-US", "within-UK", "pooled-ALL"):
        s, m = scope.split("-")
        subset = daily[daily.scope == s]
        if m != "ALL":
            subset = subset[subset.market == m]
        else:
            subset = subset[subset.market == "ALL"]
        by_h = {}
        for h in HORIZONS:
            dated = subset[subset.horizon == h].groupby("date").mean(numeric_only=True)
            if len(dated) < 5:
                continue
            lags = sorted({lag for lag in (h - 1, 6, 12, 18) if 0 <= lag < len(dated)})
            comps = {}
            for reference in ("momentum", "equal"):
                diff = (dated.ridge_ic - dated[reference + "_ic"]).to_numpy()
                comps[reference] = [hac_mean(diff, lag) for lag in lags]
            by_h[h] = {
                "dates": len(dated),
                "mean_ridge_ic": float(dated.ridge_ic.mean()),
                "mean_momentum_ic": float(dated.momentum_ic.mean()),
                "ridge_ic_vs": comps,
            }
        results[scope] = by_h
    report = {
        "status": "validation_complete",
        "test_scored": False,
        "deployment": False,
        "results": results,
        "inference_note": "Primary band L=h-1; 6/12/18 saved as sensitivity. Exploratory unadjusted intervals.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output_dir / "linear_model.json").write_text(
        json.dumps({str(h): m.coef_.tolist() for h, m in models.items()})
    )
    for scope, by_h in results.items():
        for h, r in by_h.items():
            lags = sorted({lag for lag in (h - 1, 6, 12, 18) if 0 <= lag < r["dates"]})
            primary = r["ridge_ic_vs"]["equal"][0]
            print(
                f"{scope} h={h} dates={r['dates']} IC={r['mean_ridge_ic']:.5f} "
                f"(mom={r['mean_momentum_ic']:.5f}) HAC@L{lags[0]} "
                f"mean={primary['mean_loss_improvement']:.5f} "
                f"ci95=[{primary['ci95'][0]:.5f},{primary['ci95'][1]:.5f}] "
                f"p={primary['p_two_sided']}",
                flush=True,
            )
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
