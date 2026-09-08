"""Frozen A-F volatility-structure ablation runner (validation only).

Executes only the requested arms against the parquet cache and writes
protocol.json, predictions.parquet, and report.json. Arm semantics live in
research.volatility_structure.panel and must not be redefined here.
Test partition is never scored.
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
# The HAR wrapper reuses the production engine, which lives under backend/.
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from research.price_forecasting.gpu_pipeline import _normalise_ohlcv  # noqa: E402
from research.price_forecasting.paired_validation import hac_mean  # noqa: E402
from research.volatility_structure import panel  # noqa: E402
from research.volatility_structure.har_wrapper import har_forecasts  # noqa: E402
from research.volatility_structure.range_estimators import range_variances  # noqa: E402

B_ARMS = ("B1-parkinson", "B1-garman_klass", "B1-rogers_satchell", "B1-yang_zhang", "B2")
EXECUTABLE = ("A",) + B_ARMS + ("C", "D")
D_COLUMNS = tuple(list(panel.B2_COLUMNS) + list(panel.D_EXTRA_COLUMNS))


def _load_cache() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted((REPO_ROOT / "data" / "tri_exchange" / "cache").glob("*.parquet")):
        try:
            frames[path.stem] = _normalise_ohlcv(pd.read_parquet(path))
        except Exception as exc:
            print(f"skip {path.stem}: {exc}", flush=True)
    if not frames:
        raise RuntimeError("No cached OHLCV frames available")
    return frames


def run_arm_a(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Standalone rolling-champion forecasts and realized targets per origin."""
    records = []
    for ticker, frame in frames.items():
        close = frame["Close"]
        forecasts = panel.arm_a_forecasts(close)
        for horizon in panel.TARGET_HORIZONS:
            realized = panel.forward_realized_variance(close, horizon)
            table = pd.DataFrame(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "origin_date": close.index,
                    "target_end_date": close.index[
                        np.minimum(np.arange(len(close)) + horizon, len(close) - 1)
                    ],
                    "arm": "A",
                    "forecast": forecasts[f"f_h{horizon}"].to_numpy(),
                    "realized": realized.to_numpy(),
                }
            )
            records.append(table)
    out = pd.concat(records, ignore_index=True)
    return out.dropna(subset=["forecast", "realized"]).reset_index(drop=True)


def run_arm_b1(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Standalone range forecasts; raw estimator values, floor at scoring."""
    records = []
    for ticker, frame in frames.items():
        estimated = range_variances(frame["Open"], frame["High"], frame["Low"], frame["Close"])
        arms = panel.arm_b1_forecasts(estimated)
        for arm, table in arms.items():
            for horizon in panel.TARGET_HORIZONS:
                realized = panel.forward_realized_variance(frame["Close"], horizon)
                rows = pd.DataFrame(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "origin_date": frame.index,
                        "target_end_date": frame.index[
                            np.minimum(np.arange(len(frame)) + horizon, len(frame) - 1)
                        ],
                        "arm": arm,
                        "forecast": table[f"f_h{horizon}"].to_numpy(),
                        "realized": realized.to_numpy(),
                    }
                )
                records.append(rows)
    out = pd.concat(records, ignore_index=True)
    return out.dropna(subset=["forecast", "realized"]).reset_index(drop=True)


def run_arm_b2(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pooled Ridge per horizon on the frozen four-column range block."""
    rows = []
    for ticker, frame in frames.items():
        estimated = range_variances(frame["Open"], frame["High"], frame["Low"], frame["Close"])
        block = panel.range_block_features(estimated)
        for horizon in panel.TARGET_HORIZONS:
            realized = panel.forward_realized_variance(frame["Close"], horizon)
            table = pd.DataFrame(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "origin_date": frame.index,
                    "target_end_date": frame.index[
                        np.minimum(np.arange(len(frame)) + horizon, len(frame) - 1)
                    ],
                    "realized": realized.to_numpy(),
                }
            )
            for column in panel.B2_COLUMNS:
                table[column] = block[column].to_numpy()
            rows.append(table)
    pooled = pd.concat(rows, ignore_index=True)
    predictions = []
    training_windows = {}
    for horizon in panel.TARGET_HORIZONS:
        sub = pooled[pooled.horizon == horizon].reset_index(drop=True)
        finite = sub[list(panel.B2_COLUMNS) + ["realized"]].apply(
            lambda col: np.isfinite(col.to_numpy())
        )
        usable = finite.all(axis=1).to_numpy()
        train, _, _, _ = panel.partitions(
            sub.origin_date.to_numpy(), sub.target_end_date.to_numpy()
        )
        train_idx = np.where(train & usable)[0]
        if len(train_idx) < 100:
            raise RuntimeError(f"B2 has too few training rows at h={horizon}")
        scaler, model = panel.fit_ridge_direct(
            sub.loc[train_idx, list(panel.B2_COLUMNS)].to_numpy(dtype=float),
            sub.loc[train_idx, "realized"].to_numpy(dtype=float),
        )
        scored = usable.copy()
        pred = np.full(len(sub), np.nan)
        pred[scored] = model.predict(
            scaler.transform(sub.loc[scored, list(panel.B2_COLUMNS)].to_numpy(dtype=float))
        )
        training_windows[horizon] = (
            str(sub.loc[train_idx, "origin_date"].min()),
            str(sub.loc[train_idx, "origin_date"].max()),
        )
        predictions.append(
            pd.DataFrame(
                {
                    "ticker": sub.ticker,
                    "horizon": horizon,
                    "origin_date": sub.origin_date,
                    "target_end_date": sub.target_end_date,
                    "arm": "B2",
                    "forecast": pred,
                    "realized": sub.realized.to_numpy(),
                }
            )
        )
    out = pd.concat(predictions, ignore_index=True)
    out = out.dropna(subset=["forecast", "realized"]).reset_index(drop=True)
    return out, training_windows


def run_arm_d(frames: dict[str, pd.DataFrame]):
    """Pooled Ridge per horizon on HAR log components + frozen range block."""
    from research.volatility_structure.har_wrapper import har_log_components

    rows = []
    for ticker, frame in frames.items():
        estimated = range_variances(frame["Open"], frame["High"], frame["Low"], frame["Close"])
        block = panel.range_block_features(estimated)
        har = har_log_components(frame["Close"])
        feats = pd.concat([block, har], axis=1)
        for horizon in panel.TARGET_HORIZONS:
            realized = panel.forward_realized_variance(frame["Close"], horizon)
            table = pd.DataFrame(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "origin_date": frame.index,
                    "target_end_date": frame.index[
                        np.minimum(np.arange(len(frame)) + horizon, len(frame) - 1)
                    ],
                    "realized": realized.to_numpy(),
                }
            )
            for column in D_COLUMNS:
                table[column] = feats[column].to_numpy()
            rows.append(table)
    pooled = pd.concat(rows, ignore_index=True)
    predictions = []
    training_windows = {}
    fitted = {}
    for horizon in panel.TARGET_HORIZONS:
        sub = pooled[pooled.horizon == horizon].reset_index(drop=True)
        finite = sub[list(D_COLUMNS) + ["realized"]].apply(
            lambda col: np.isfinite(col.to_numpy())
        )
        usable = finite.all(axis=1).to_numpy()
        train, _, _, _ = panel.partitions(
            sub.origin_date.to_numpy(), sub.target_end_date.to_numpy()
        )
        train_idx = np.where(train & usable)[0]
        if len(train_idx) < 100:
            raise RuntimeError(f"D has too few training rows at h={horizon}")
        scaler, model = panel.fit_ridge_direct(
            sub.loc[train_idx, list(D_COLUMNS)].to_numpy(dtype=float),
            sub.loc[train_idx, "realized"].to_numpy(dtype=float),
        )
        scored = usable.copy()
        pred = np.full(len(sub), np.nan)
        pred[scored] = model.predict(
            scaler.transform(sub.loc[scored, list(D_COLUMNS)].to_numpy(dtype=float))
        )
        training_windows[horizon] = (
            str(sub.loc[train_idx, "origin_date"].min()),
            str(sub.loc[train_idx, "origin_date"].max()),
        )
        fitted[horizon] = {"intercept": float(model.intercept_[0]) if np.ndim(model.intercept_) else float(model.intercept_),
                           "standardized_coef": {name: float(w) for name, w in zip(D_COLUMNS, np.asarray(model.coef_).reshape(-1))}}
        predictions.append(
            pd.DataFrame(
                {
                    "ticker": sub.ticker,
                    "horizon": horizon,
                    "origin_date": sub.origin_date,
                    "target_end_date": sub.target_end_date,
                    "arm": "D",
                    "forecast": pred,
                    "realized": sub.realized.to_numpy(),
                }
            )
        )
    out = pd.concat(predictions, ignore_index=True)
    out = out.dropna(subset=["forecast", "realized"]).reset_index(drop=True)
    return out, training_windows, fitted


def run_arm_c(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Production recursive HAR forecasts; no retuning, no variants."""
    from services.volatility_snapshot import causal_log_har_forecasts

    from research.volatility_structure.har_wrapper import daily_proxy

    records = []
    parity_checked = False
    for ticker, frame in frames.items():
        table = har_forecasts(frame["Close"])
        if not parity_checked:
            # Runtime wiring check on the first ticker: wrapper output must
            # equal a direct production-engine call on identical inputs.
            direct = causal_log_har_forecasts(daily_proxy(frame["Close"]), [5, 10, 20])
            np.testing.assert_array_equal(
                table[[f"har_h{h}" for h in (5, 10, 20)]].to_numpy(), direct
            )
            parity_checked = True
        for horizon in panel.TARGET_HORIZONS:
            realized = panel.forward_realized_variance(frame["Close"], horizon)
            rows = pd.DataFrame(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "origin_date": frame.index,
                    "target_end_date": frame.index[
                        np.minimum(np.arange(len(frame)) + horizon, len(frame) - 1)
                    ],
                    "arm": "C",
                    "forecast": table[f"har_h{horizon}"].to_numpy(),
                    "realized": realized.to_numpy(),
                }
            )
            records.append(rows)
    out = pd.concat(records, ignore_index=True)
    return out.dropna(subset=["forecast", "realized"]).reset_index(drop=True)


def _with_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Floor all forecasts at zero (the defined boundary; no-op for Arm A,
    whose trailing variances are already non-negative) and attach
    QLIKE/MAE/SE components."""
    out = predictions.copy()
    out["forecast"] = np.maximum(out.forecast.to_numpy(dtype=float), 0.0)
    out["qlike"] = [panel.qlike(y, f) for y, f in zip(out.realized, out.forecast, strict=True)]
    out["mae_comp"] = (out.realized - out.forecast).abs()
    out["se_comp"] = (out.realized - out.forecast) ** 2
    return out


def score_joined(joined, horizon, *, base_origins, cand_origins):
    """Metrics on an already-joined, already-purged frame.

    Shared with the rolling-origin study so both runners compute identical
    cells. Positive means favor the candidate (diff = base - candidate).
    """
    coverage = {
        "base_origins": int(base_origins),
        "cand_origins": int(cand_origins),
        "common_origins": int(len(joined)),
        "common_tickers": int(joined.ticker.nunique()) if len(joined) else 0,
    }
    if not len(joined):
        return {"coverage": coverage, "insufficient_overlap": True}
    joined = joined.copy()
    joined["diff"] = joined.qlike_a - joined.qlike
    daily = joined.groupby("origin_date")["diff"].mean().sort_index()
    lags = [horizon - 1] + sorted(
        {lag for lag in (6, 12) if lag != horizon - 1 and 0 <= lag < len(daily)}
    )
    lags = [lag for lag in lags if 0 <= lag < len(daily)]
    hac = [hac_mean(daily.to_numpy(), lag) for lag in lags]
    mean_delta = float(joined["diff"].mean())
    primary = hac[0]
    ci_low, ci_high = (float(v) for v in primary["ci95"])
    rel = mean_delta / float(joined.qlike_a.mean()) if float(joined.qlike_a.mean()) > 0 else 0.0
    return {
        "coverage": coverage,
        "mean_delta_qlike": mean_delta,
        "relative_improvement": rel,
        "hac_lags": lags,
        "hac": hac,
        "ci_excludes_zero_favoring_candidate": bool(ci_low > 0),
        "suspicious_magnitude": bool(abs(rel) > 0.30),
        "own_mae": float(joined.mae_comp.mean()),
        "own_rmse": float(np.sqrt(joined.se_comp.mean())),
        "mean_forecast_base": float(joined.forecast_base.mean()),
        "mean_forecast_cand": float(joined.forecast_cand.mean()),
        "mean_realized_common": float(joined.realized_cand.mean()),
    }


def compare_on_common_sample(base: pd.DataFrame, cand: pd.DataFrame, arm: str) -> dict:
    """Pairwise ΔQLIKE vs Arm A on the inner-joined scored sample.

    Positive means favor the candidate. Coverage is reported, never silently
    aligned: different warm-ups must not masquerade as improvements.
    """
    comparison = {}
    for horizon in panel.TARGET_HORIZONS:
        base_h = base[base.horizon == horizon].copy()
        # Validation-only on BOTH sides: recompute the purge mask on the
        # base arm and join candidates to validated base keys. Label
        # endpoints are frame-determined, so shared keys are equally clean.
        _, validation, _, _ = panel.partitions(
            base_h.origin_date.to_numpy(), base_h.target_end_date.to_numpy()
        )
        base_h = base_h.loc[validation]
        left = base_h[["ticker", "origin_date", "qlike", "forecast", "realized"]].rename(
            columns={"qlike": "qlike_a", "forecast": "forecast_base", "realized": "realized_base"}
        )
        right = cand[(cand.horizon == horizon) & (cand.arm == arm)][
            ["ticker", "origin_date", "qlike", "mae_comp", "se_comp", "forecast", "realized"]
        ].rename(columns={"forecast": "forecast_cand", "realized": "realized_cand"})
        joined = left.merge(right, on=["ticker", "origin_date"], how="inner")
        if len(joined) and not np.allclose(
            joined.realized_base.to_numpy(), joined.realized_cand.to_numpy(), rtol=0, atol=0
        ):
            raise ValueError(f"Realized-target mismatch on common sample for {arm} h={horizon}")
        comparison[horizon] = score_joined(
            joined,
            horizon,
            base_origins=int((base.horizon == horizon).sum()),
            cand_origins=int(((cand.horizon == horizon) & (cand.arm == arm)).sum()),
        )
    return comparison


def score_panel(predictions: pd.DataFrame) -> dict:
    results: dict = {}
    scored = _with_scores(predictions)
    for horizon in panel.TARGET_HORIZONS:
        sub = scored[scored.horizon == horizon].copy()
        train, validation, val_start, reserve_start = panel.partitions(
            sub.origin_date.to_numpy(), sub.target_end_date.to_numpy()
        )
        val = sub.loc[validation].copy()
        daily = val.groupby("origin_date")["qlike"].mean().sort_index()
        # Primary band first: L = h-1 per the frozen spec, then sensitivities.
        lags = [horizon - 1] + sorted(
            {lag for lag in (6, 12) if lag != horizon - 1 and 0 <= lag < len(daily)}
        )
        lags = [lag for lag in lags if 0 <= lag < len(daily)]
        results[horizon] = {
            "origins": int(len(val)),
            "dates": int(len(daily)),
            "mean_qlike": float(val.qlike.mean()),
            "mean_mae": float(val.mae_comp.mean()),
            "mean_rmse": float(np.sqrt(val.se_comp.mean())),
            "hac_qlike": [hac_mean(daily.to_numpy(), lag) for lag in lags],
            "hac_lags": lags,
            "validation_start": val_start,
            "reserve_start": reserve_start,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="A", help="comma-separated subset of the frozen matrix")
    parser.add_argument(
        "--tickers", default="", help="comma-separated subset (default: full cache)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in requested if a not in panel.ARMS]
    if unknown:
        raise ValueError(f"Unknown arms (frozen matrix is {sorted(panel.ARMS)}): {unknown}")
    unsupported = set(requested) - set(EXECUTABLE)
    if unsupported:
        raise ValueError(
            f"Arms not executable in this runner revision: {sorted(unsupported)} "
            f"(executable: {list(EXECUTABLE)})"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    frames = _load_cache()
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        frames = {t: f for t, f in frames.items() if t in wanted}
    print(f"tickers: {len(frames)}", flush=True)

    # Arm A always runs alongside candidates: deterministic, bit-identical to
    # the frozen A run, and required for common-sample comparison. Arm D
    # additionally pulls its descriptive baselines (C, YZ) for context.
    run_requested = list(dict.fromkeys(["A"] + requested))
    if "D" in run_requested:
        run_requested = list(dict.fromkeys(run_requested + ["C", "B1-yang_zhang"]))
    tables = {"A": run_arm_a(frames)}
    training_windows = {}
    fitted_d = {}
    if "B2" in run_requested:
        b2_table, training_windows = run_arm_b2(frames)
        tables["B2"] = b2_table
    wanted_b1 = [arm for arm in run_requested if arm.startswith("B1-")]
    if wanted_b1:
        b1_all = run_arm_b1(frames)
        for arm in wanted_b1:
            tables[arm] = b1_all[b1_all.arm == arm].reset_index(drop=True)
    if "C" in run_requested:
        tables["C"] = run_arm_c(frames)
    if "D" in run_requested:
        d_table, d_windows, fitted_d = run_arm_d(frames)
        tables["D"] = d_table
        training_windows = {**training_windows, **{f"D_h{k}": v for k, v in d_windows.items()}}
    predictions = pd.concat([tables[arm] for arm in run_requested], ignore_index=True)
    predictions["feature_schema"] = panel.FEATURE_SCHEMA_VERSION
    predictions["partition"] = "validation-pending"
    comparisons = {
        arm: compare_on_common_sample(
            _with_scores(predictions[predictions.arm == "A"]),
            _with_scores(predictions),
            arm,
        )
        for arm in run_requested
        if arm != "A"
    }
    # Descriptive secondaries for D only (context, never selection).
    secondary = {}
    diagnostics_d = {}
    if "D" in run_requested:
        scored_all = _with_scores(predictions)
        for label, base_arm in (("D_vs_C", "C"), ("D_vs_YZ", "B1-yang_zhang")):
            secondary[label] = compare_on_common_sample(
                scored_all[scored_all.arm == base_arm], scored_all, "D"
            )
        d_rows = predictions[predictions.arm == "D"]
        a_rows = predictions[predictions.arm == "A"]
        for horizon in panel.TARGET_HORIZONS:
            base_h = a_rows[a_rows.horizon == horizon]
            _, validation, _, _ = panel.partitions(
                base_h.origin_date.to_numpy(), base_h.target_end_date.to_numpy()
            )
            keys = set(
                zip(
                    base_h.loc[validation, "ticker"], base_h.loc[validation, "origin_date"],
                    strict=True,
                )
            )
            sub = d_rows[d_rows.horizon == horizon]
            sub = sub[
                [(t, d) in keys for t, d in zip(sub.ticker, sub.origin_date, strict=True)]
            ]
            raw = sub.forecast.to_numpy(dtype=float)
            diagnostics_d[horizon] = {
                "intercept": fitted_d[horizon]["intercept"],
                "standardized_coef": fitted_d[horizon]["standardized_coef"],
                "pred_real_ratio": float(np.mean(raw) / np.mean(sub.realized.to_numpy(dtype=float))),
                "pred_quantiles": {
                    q: float(np.quantile(raw, float(q)))
                    for q in ("0.01", "0.05", "0.5", "0.95", "0.99")
                },
                "floor_hit_fraction": float(np.mean(raw < 0.0)),
                "origins": int(len(sub)),
            }
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    # Arm-level summaries are scored on each arm's own rows (never pooled
    # across arms); pairwise claims come only from compare_on_common_sample.
    arm_results = {arm: score_panel(predictions[predictions.arm == arm]) for arm in run_requested}
    results = arm_results["A"]
    # Run-validity gate: Arm A is deterministic; any drift from the frozen
    # numbers means the harness (not the market) changed. Abort loudly.
    # Skipped for --tickers subsets, whose panel means legitimately differ.
    if args.tickers:
        print("A-gate skipped (ticker subset)", flush=True)
    else:
        frozen_a = json.loads(
            (REPO_ROOT / "artifacts" / "vol_structure_v1" / "report.json").read_text()
        )
        for horizon in panel.TARGET_HORIZONS:
            frozen_value = frozen_a["results"][str(horizon)]["mean_qlike"]
            live_value = results[horizon]["mean_qlike"]
            if abs(frozen_value - live_value) > 1e-9:
                raise ValueError(
                    f"Arm A drifted at h={horizon}: frozen={frozen_value} live={live_value}"
                )
    # Coverage detail: scored origins/tickers and first-valid-origin
    # distribution per executed arm (warm-up/fallbacks recorded, not dropped).
    coverage_detail = {}
    for arm in run_requested:
        sub = predictions[predictions.arm == arm]
        firsts = sub.groupby("ticker")["origin_date"].min()
        coverage_detail[arm] = {
            "scored_origins": int(len(sub)),
            "scored_tickers": int(sub.ticker.nunique()),
            "first_valid_origin_min": str(firsts.min()),
            "first_valid_origin_median": str(firsts.median()),
        }
    protocol = {
        "hypothesis": "frozen A-F matrix; this run executes " + ",".join(run_requested),
        "arms_requested": requested,
        "arms_executed": run_requested,
        "arm_matrix": panel.ARMS,
        "comparison_rule": "pairwise vs Arm A on inner-joined scored origins; "
        "forecasts floored at zero at scoring (no-op for A)",
        "target": "forward cumulative realized variance (sum of squared log returns)",
        "target_horizons": list(panel.TARGET_HORIZONS),
        "primary_metric": "QLIKE",
        "feature_schema": panel.FEATURE_SCHEMA_VERSION,
        "b2_training_windows": {str(k): v for k, v in training_windows.items()},
        "d_fitted": fitted_d if "D" in run_requested else {},
        "production_observation": (
            "Production raw HAR parity confirmed; validation study indicates "
            "substantial cumulative-variance underforecasting under recursive "
            "log-median multi-step forecasting. No production change made "
            "within this study."
        ),
        "coverage_detail": coverage_detail,
        "tickers": len(frames),
        "origins_total": int(len(predictions)),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    report = {
        "status": "validation_complete",
        "test_scored": False,
        "results": results,
        "comparisons_vs_A": comparisons,
        "secondary_descriptive": secondary,
        "diagnostics_D": diagnostics_d,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for horizon, summary in results.items():
        primary = summary["hac_qlike"][0]
        print(
            f"A h={horizon} origins={summary['origins']} dates={summary['dates']} "
            f"QLIKE={summary['mean_qlike']:.5f} MAE={summary['mean_mae']:.6f} "
            f"HAC@{summary['hac_lags'][0]} ci95=[{primary['ci95'][0]:.5f},{primary['ci95'][1]:.5f}]",
            flush=True,
        )
    for arm, comparison in comparisons.items():
        for horizon in panel.TARGET_HORIZONS:
            cell = comparison[horizon]
            if cell.get("insufficient_overlap"):
                print(f"{arm} h={horizon}: INSUFFICIENT OVERLAP {cell['coverage']}", flush=True)
                continue
            hac0 = cell["hac"][0]
            print(
                f"{arm} h={horizon} common={cell['coverage']['common_origins']} "
                f"dQLIKE={cell['mean_delta_qlike']:+.5f} rel={cell['relative_improvement']:+.4f} "
                f"HAC@{cell['hac_lags'][0]} ci95=[{hac0['ci95'][0]:+.5f},{hac0['ci95'][1]:+.5f}] "
                f"p={hac0['p_two_sided']:.3g} cand_improves={cell['ci_excludes_zero_favoring_candidate']} "
                f"suspicious={cell['suspicious_magnitude']} MAE={cell['own_mae']:.6f} "
                f"meanF_base={cell['mean_forecast_base']:.6f} meanF_cand={cell['mean_forecast_cand']:.6f} "
                f"meanY={cell['mean_realized_common']:.6f}",
                flush=True,
            )
    for label, comparison in secondary.items():
        for horizon in panel.TARGET_HORIZONS:
            cell = comparison[horizon]
            if cell.get("insufficient_overlap"):
                continue
            hac0 = cell["hac"][0]
            print(
                f"{label} h={horizon} dQLIKE={cell['mean_delta_qlike']:+.5f} "
                f"p={hac0['p_two_sided']:.3g} (descriptive only)",
                flush=True,
            )
    for horizon in sorted(diagnostics_d):
        diag = diagnostics_d[horizon]
        coefs = ", ".join(f"{k}={v:+.4f}" for k, v in diag["standardized_coef"].items())
        print(
            f"D-diag h={horizon} intercept={diag['intercept']:+.6f} [{coefs}] "
            f"pred/real={diag['pred_real_ratio']:.3f} "
            f"q01={diag['pred_quantiles']['0.01']:.6f} q50={diag['pred_quantiles']['0.5']:.6f} "
            f"floor_frac={diag['floor_hit_fraction']:.4f} n={diag['origins']}",
            flush=True,
        )
    print(f"wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
