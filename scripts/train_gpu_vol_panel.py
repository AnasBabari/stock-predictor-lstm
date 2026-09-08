"""GPU panel volatility study: global XGBoost across all stocks (validation only).

Arms (frozen):
  G0  rolling champion (recomputed; gated against the frozen A numbers).
  G1  Yang-Zhang standalone candidate (recomputed).
  G2  global XGB, QLIKE objective, no base margin (learns log-variance).
  G3  global XGB, QLIKE objective, base_margin=log(rolling) (learns the
      log-multiplicative correction delta; Vhat = B*exp(delta) > 0 always).
  G4  G3 plus market/one-hot and SPY context columns.
G5 (neural) stays gated behind these results.

G3 feature set excludes context columns; G4 adds market_US/market_UK,
spy_rv_20, spy_ret_20. XGB params frozen below. Fixed 500 rounds, no early
stopping. Test partition never scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from research.price_forecasting.gpu_pipeline import _normalise_ohlcv  # noqa: E402
from research.price_forecasting.paired_validation import hac_mean  # noqa: E402
from research.volatility_structure import panel  # noqa: E402
from research.volatility_structure.gpu_panel import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    build_ticker_features,
    qlike_objective,
)
from research.volatility_structure.range_estimators import range_variances  # noqa: E402

XGB_PARAMS = {
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "seed": 42,
    "device": "cuda",
}
N_ROUNDS = 500
G3_FEATURES = [
    c for c in FEATURE_COLUMNS if c not in ("market_US", "market_UK", "spy_rv_20", "spy_ret_20")
]
G3_IDX = [FEATURE_COLUMNS.index(c) for c in G3_FEATURES]


def _load_frames():
    frames = {}
    for path in sorted((REPO_ROOT / "data" / "tri_exchange" / "cache").glob("*.parquet")):
        try:
            frames[path.stem] = _normalise_ohlcv(pd.read_parquet(path))
        except Exception as exc:
            print(f"skip {path.stem}: {exc}", flush=True)
    if not frames:
        raise RuntimeError("No cached OHLCV frames available")
    return frames


def _load_spy():
    spy = pd.read_parquet(REPO_ROOT / "data" / "macro" / "market_dailies.parquet")[["SPY"]]
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    return spy.sort_index()


def build_panel(frames, spy):
    """One row per (ticker, session) with frozen features, targets, base."""
    records = []
    for ticker, frame in frames.items():
        market = "UK" if ticker.endswith(".L") else "US"
        feats = build_ticker_features(frame, market, spy=spy)
        base = panel.arm_a_forecasts(frame["Close"])
        estimated = range_variances(frame["Open"], frame["High"], frame["Low"], frame["Close"])
        yz = estimated["yang_zhang_var"]
        yz_mean20 = yz.rolling(20, min_periods=20).mean()
        rows = pd.DataFrame({"ticker": ticker, "origin_date": frame.index})
        for column in FEATURE_COLUMNS:
            rows[column] = feats[column].to_numpy()
        for horizon in panel.TARGET_HORIZONS:
            rows[f"target_h{horizon}"] = panel.forward_realized_variance(
                frame["Close"], horizon
            ).to_numpy()
            rows[f"base_h{horizon}"] = base[f"f_h{horizon}"].to_numpy()
            rows[f"yz_h{horizon}"] = yz_mean20.to_numpy() * horizon
        rows["target_end_h5"] = frame.index[
            np.minimum(np.arange(len(frame)) + 5, len(frame) - 1)
        ].to_numpy()
        rows["target_end_h10"] = frame.index[
            np.minimum(np.arange(len(frame)) + 10, len(frame) - 1)
        ].to_numpy()
        rows["target_end_h20"] = frame.index[
            np.minimum(np.arange(len(frame)) + 20, len(frame) - 1)
        ].to_numpy()
        records.append(rows)
    return pd.concat(records, ignore_index=True)


def _finite_rows(table, columns):
    return np.isfinite(table[list(columns)].to_numpy(dtype=float)).all(axis=1)


def train_xgb(x_train, y_train, base_margin=None, base_score=None):
    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, label=y_train)
    if base_margin is not None:
        dtrain.set_base_margin(np.asarray(base_margin, dtype=np.float64))
    params = dict(XGB_PARAMS)
    # Optimizer init only (not model structure): raw margins live near
    # log-label scale (~-9), so the default base_score=0.5 starts Newton
    # steps astronomically far out and diverges. No base margin is added.
    if base_margin is None:
        params["base_score"] = float(np.log(max(np.mean(y_train), panel.QLIKE_FLOOR)))
    booster = xgb.train(
        {**params, "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=N_ROUNDS,
        obj=qlike_objective(),
        verbose_eval=False,
    )
    return booster


def predict_xgb(booster, x, base_margin=None):
    import xgboost as xgb

    dtest = xgb.DMatrix(x)
    if base_margin is not None:
        dtest.set_base_margin(np.asarray(base_margin, dtype=np.float64))
    raw = booster.predict(dtest)
    return np.exp(np.clip(np.asarray(raw, dtype=np.float64), -30.0, 10.0))


def score_predictions(table):
    """Attach QLIKE/MAE/SE for one arm's scored rows (already floored upstream)."""
    out = table.copy()
    out["qlike"] = [panel.qlike(y, f) for y, f in zip(out.realized, out.forecast, strict=True)]
    out["mae_comp"] = (out.realized - out.forecast).abs()
    out["se_comp"] = (out.realized - out.forecast) ** 2
    return out


def compare_arms(scored, ref_arm, cand_arm, horizon):
    ref = scored[(scored.arm == ref_arm) & (scored.horizon == horizon)]
    cand = scored[(scored.arm == cand_arm) & (scored.horizon == horizon)]
    joined = ref.merge(cand, on=["ticker", "origin_date"], suffixes=("_ref", "_cand"), how="inner")
    coverage = {
        "ref_origins": int(len(ref)),
        "cand_origins": int(len(cand)),
        "common_origins": int(len(joined)),
        "common_tickers": int(joined.ticker.nunique()) if len(joined) else 0,
    }
    if not len(joined):
        return {"coverage": coverage, "insufficient_overlap": True}
    joined["diff"] = joined.qlike_ref - joined.qlike_cand
    daily = joined.groupby("origin_date")["diff"].mean().sort_index()
    lags = [horizon - 1] + sorted(
        {lag for lag in (6, 12) if lag != horizon - 1 and 0 <= lag < len(daily)}
    )
    lags = [lag for lag in lags if 0 <= lag < len(daily)]
    hac = [hac_mean(daily.to_numpy(), lag) for lag in lags]
    mean_delta = float(joined["diff"].mean())
    primary = hac[0]
    base_mean = float(joined.qlike_ref.mean())
    return {
        "coverage": coverage,
        "mean_delta_qlike": mean_delta,
        "relative_improvement": mean_delta / base_mean if base_mean > 0 else 0.0,
        "hac_lags": lags,
        "hac": hac,
        "ci_excludes_zero_favoring_candidate": bool(float(primary["ci95"][0]) > 0),
        "suspicious_magnitude": bool(abs(mean_delta / base_mean) > 0.30)
        if base_mean > 0
        else False,
        "own_mae": float(joined.mae_comp_cand.mean()),
        "own_rmse": float(np.sqrt(joined.se_comp_cand.mean())),
        "mean_forecast_cand": float(joined.forecast_cand.mean()),
        "mean_realized_common": float(joined.realized_cand.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="G0,G1,G2,G3,G4")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    frozen = {"G0", "G1", "G2", "G3", "G4"}
    unknown = [a for a in requested if a not in frozen]
    if unknown:
        raise ValueError(f"Unknown arms (frozen: {sorted(frozen)}): {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    frames = _load_cache_frames(args.tickers)
    spy = _load_spy()
    print(f"tickers: {len(frames)}", flush=True)
    panel_df = build_panel(frames, spy)

    scored_parts = []
    for horizon in panel.TARGET_HORIZONS:
        cols = list(FEATURE_COLUMNS) + [f"target_h{horizon}", f"base_h{horizon}"]
        sub = panel_df[["ticker", "origin_date", f"target_end_h{horizon}"] + cols].copy()
        sub = sub.rename(
            columns={
                f"target_h{horizon}": "realized",
                f"base_h{horizon}": "base",
                f"target_end_h{horizon}": "target_end_date",
            }
        )
        usable = _finite_rows(sub, list(FEATURE_COLUMNS) + ["realized", "base"])
        sub = sub.loc[usable].reset_index(drop=True)
        usable = (
            np.isfinite(sub[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(sub.realized.to_numpy(dtype=float))
            & np.isfinite(sub.base.to_numpy(dtype=float))
        )
        sub = sub.loc[usable].reset_index(drop=True)
        train, _, _, _ = panel.partitions(
            sub.origin_date.to_numpy(), sub.target_end_date.to_numpy()
        )
        train_idx = np.where(train)[0]
        if len(train_idx) < 1000:
            raise RuntimeError(f"Too few training rows at h={horizon}")
        x_train = sub.loc[train_idx, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        y_train = sub.loc[train_idx, "realized"].to_numpy(dtype=float)
        x_all = sub.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        base_all = np.maximum(sub.base.to_numpy(dtype=float), panel.QLIKE_FLOOR)

        arm_frames = {}
        arm_frames["G0"] = base_all.copy()
        g1 = sub[[c for c in FEATURE_COLUMNS if c.startswith("range_yz_20")]].to_numpy(dtype=float)[
            :, 0
        ]
        arm_frames["G1"] = np.maximum(g1, 0.0) * horizon
        log_base = np.log(np.maximum(base_all, panel.QLIKE_FLOOR))
        arm_frames["G2"] = predict_xgb(train_xgb(x_train, y_train), x_all)
        arm_frames["G3"] = predict_xgb(
            train_xgb(x_train[:, G3_IDX], y_train, base_margin=log_base[train_idx]),
            x_all[:, G3_IDX],
            base_margin=log_base,
        )
        arm_frames["G4"] = predict_xgb(
            train_xgb(x_train, y_train, base_margin=log_base[train_idx]),
            x_all,
            base_margin=log_base,
        )
        for arm, values in arm_frames.items():
            scored_parts.append(
                pd.DataFrame(
                    {
                        "ticker": sub.ticker,
                        "horizon": horizon,
                        "origin_date": sub.origin_date,
                        "target_end_date": sub.target_end_date,
                        "arm": arm,
                        "forecast": values,
                        "realized": sub.realized.to_numpy(),
                    }
                )
            )
        print(f"h={horizon} train_rows={len(train_idx)} scored={len(sub)}", flush=True)

    scored = score_predictions(pd.concat(scored_parts, ignore_index=True))
    scored.to_parquet(args.output_dir / "predictions.parquet", index=False)

    results = {}
    for horizon in panel.TARGET_HORIZONS:
        sub = scored[(scored.arm == "G0") & (scored.horizon == horizon)]
        train, validation, val_start, reserve_start = panel.partitions(
            sub.origin_date.to_numpy(), sub.target_end_date.to_numpy()
        )
        val = sub.loc[validation]
        daily = val.groupby("origin_date")["qlike"].mean().sort_index()
        lags = [horizon - 1] + sorted(
            {lag for lag in (6, 12) if lag != horizon - 1 and 0 <= lag < len(daily)}
        )
        lags = [lag for lag in lags if 0 <= lag < len(daily)]
        results[horizon] = {
            "origins": int(len(val)),
            "dates": int(len(daily)),
            "mean_qlike": float(val.qlike.mean()),
            "hac_qlike": [hac_mean(daily.to_numpy(), lag) for lag in lags],
            "hac_lags": lags,
            "validation_start": val_start,
            "reserve_start": reserve_start,
        }
    if not args.tickers:
        # Run-validity gate: G0 recomputes the frozen champion through the
        # same deterministic function, so every shared origin must match
        # the frozen A predictions bit-for-bit (independent of split dates).
        frozen = pd.read_parquet(
            REPO_ROOT / "artifacts" / "vol_structure_v1" / "predictions.parquet"
        )
        live_a = scored[scored.arm == "G0"][["ticker", "horizon", "origin_date", "forecast"]]
        check = frozen.merge(
            live_a,
            on=["ticker", "horizon", "origin_date"],
            suffixes=("_frozen", "_live"),
            how="inner",
        )
        assert len(check) > 100000, "G0/frozen overlap too small to gate"
        max_abs_diff = float(
            np.max(np.abs(check.forecast_frozen.to_numpy() - check.forecast_live.to_numpy()))
        )
        print(f"G0 gate: {len(check)} shared origins, max abs diff={max_abs_diff:.2e}", flush=True)
        if max_abs_diff > 0.0:
            raise ValueError("G0 drifted from frozen Arm A predictions")
    else:
        print("G0 gate skipped (ticker subset)", flush=True)

    comparisons = {}
    for arm in requested:
        if arm == "G0":
            continue
        comparisons[arm] = {h: compare_arms(scored, "G0", arm, h) for h in panel.TARGET_HORIZONS}
    protocol = {
        "hypothesis": "frozen GPU ladder G0-G4; G5 gated",
        "arms": requested,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "g3_features": G3_FEATURES,
        "xgb_params": {
            **XGB_PARAMS,
            "objective": "custom_qlike_log_margin",
            "n_estimators": N_ROUNDS,
        },
        "note": "G2 learns log-variance (no base margin); G3/G4 learn the log-multiplicative correction with base_margin=log(rolling).",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    report = {
        "status": "validation_complete",
        "test_scored": False,
        "results": results,
        "comparisons_vs_G0": comparisons,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for arm, comparison in comparisons.items():
        for horizon in panel.TARGET_HORIZONS:
            cell = comparison[horizon]
            if cell.get("insufficient_overlap"):
                print(f"{arm} h={horizon}: INSUFFICIENT OVERLAP", flush=True)
                continue
            hac0 = cell["hac"][0]
            print(
                f"{arm} h={horizon} common={cell['coverage']['common_origins']} "
                f"dQLIKE={cell['mean_delta_qlike']:+.5f} rel={cell['relative_improvement']:+.4f} "
                f"HAC@{cell['hac_lags'][0]} ci95=[{hac0['ci95'][0]:+.5f},{hac0['ci95'][1]:+.5f}] "
                f"p={hac0['p_two_sided']:.3g} MAE={cell['own_mae']:.6f}",
                flush=True,
            )
    print(f"wrote {args.output_dir}", flush=True)
    return 0


def _load_cache_frames(tickers_arg: str) -> dict[str, pd.DataFrame]:
    from research.price_forecasting.gpu_pipeline import _normalise_ohlcv as _norm

    frames = {}
    for path in sorted((REPO_ROOT / "data" / "tri_exchange" / "cache").glob("*.parquet")):
        try:
            frames[path.stem] = _norm(pd.read_parquet(path))
        except Exception as exc:
            print(f"skip {path.stem}: {exc}", flush=True)
    if tickers_arg:
        wanted = {t.strip().upper() for t in tickers_arg.split(",") if t.strip()}
        frames = {t: f for t, f in frames.items() if t in wanted}
    return frames


if __name__ == "__main__":
    raise SystemExit(main())
