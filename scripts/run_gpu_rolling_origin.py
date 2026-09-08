"""Rolling-origin regime replication for G3 vs G0 (validation only).

Frozen folds BEFORE any fold result is observed: calendar-year validation
blocks 2019-2024 with expanding train from 2015 and the same purge
semantics (labels may not cross the next boundary). G3 configuration is
frozen (params, 22 features, QLIKE objective, base_margin, 500 rounds).
Test partition never touched. See DECISION.md for the promotion rule.
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

from research.volatility_structure import panel  # noqa: E402
from research.volatility_structure.gpu_panel import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
)
from scripts.run_vol_structure_ablation import score_joined  # noqa: E402
from scripts.train_gpu_vol_panel import (  # noqa: E402
    G3_IDX,
    _load_cache_frames,
    _load_spy,
    build_panel,
    predict_xgb,
    score_predictions,
    train_xgb,
)

FOLDS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024)


def fold_masks(dates: pd.Series, ends: pd.Series, year: int):
    """Expanding-train / calendar-year-validation masks with purge on both."""
    start = np.datetime64(f"{year}-01-01")
    stop = np.datetime64(f"{year + 1}-01-01")
    dates = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    ends = pd.to_datetime(ends).to_numpy(dtype="datetime64[ns]")
    train = (dates < start) & (ends < start)
    validation = (dates >= start) & (dates < stop) & (ends < stop)
    return train, validation


def tail_report(joined: pd.DataFrame) -> dict:
    """Tail dependence: is the edge broad or a few catastrophic dates?"""
    daily = joined.groupby("origin_date")["diff"].mean().sort_index()
    total = float(np.abs(daily.to_numpy()).sum())
    worst = daily.sort_values()
    out = {}
    for frac in ("0.01", "0.05", "0.10"):
        k = max(1, int(len(daily) * float(frac)))
        share = float(np.abs(worst.iloc[:k].to_numpy()).sum() / total) if total > 0 else 0.0
        out[frac] = {
            "share_of_abs_delta": share,
            "worst_dates": [str(d.date()) for d in worst.index[:k]],
        }
    return {
        "frac_dates_won": float((daily > 0).mean()) if len(daily) else 0.0,
        "tail_shares": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default=",".join(str(y) for y in FOLDS))
    parser.add_argument("--tickers", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="ONE-SHOT untouched-test scoring: train final G3 on all non-test "
        "rows and score held-out test origins once. Refuses to run twice.",
    )
    args = parser.parse_args()
    folds = [int(y.strip()) for y in args.folds.split(",") if y.strip()]
    unknown = [y for y in folds if y not in FOLDS]
    if unknown:
        raise ValueError(f"Unknown folds (frozen: {list(FOLDS)}): {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    frames = _load_cache_frames(args.tickers)
    spy = _load_spy()
    print(f"tickers: {len(frames)}", flush=True)
    full = build_panel(frames, spy)

    all_scored = []
    fold_meta = {}
    for year in folds:
        for horizon in panel.TARGET_HORIZONS:
            cols = ["ticker", "origin_date", f"target_end_h{horizon}"]
            sub = full[
                cols + list(FEATURE_COLUMNS) + [f"target_h{horizon}", f"base_h{horizon}"]
            ].copy()
            sub = sub.rename(
                columns={
                    f"target_h{horizon}": "realized",
                    f"base_h{horizon}": "base",
                    f"target_end_h{horizon}": "target_end_date",
                }
            )
            usable = (
                np.isfinite(sub[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
                & np.isfinite(sub.realized.to_numpy(dtype=float))
                & np.isfinite(sub.base.to_numpy(dtype=float))
            )
            sub = sub.loc[usable].reset_index(drop=True)
            train_mask, val_mask = fold_masks(sub.origin_date, sub.target_end_date, year)
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            if len(train_idx) < 1000 or len(val_idx) < 100:
                raise RuntimeError(f"Fold {year} h={horizon} too small to score")
            x_train = sub.loc[train_idx, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            y_train = sub.loc[train_idx, "realized"].to_numpy(dtype=float)
            x_all = sub.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            log_base_all = np.log(np.maximum(sub.base.to_numpy(dtype=float), panel.QLIKE_FLOOR))
            booster = train_xgb(
                x_train[:, G3_IDX],
                y_train,
                base_margin=log_base_all[train_idx],
            )
            scored = predict_xgb(booster, x_all[:, G3_IDX], base_margin=log_base_all)
            for arm, values in (
                ("G3", scored),
                ("G0", np.maximum(sub.base.to_numpy(dtype=float), panel.QLIKE_FLOOR)),
            ):
                all_scored.append(
                    pd.DataFrame(
                        {
                            "fold": year,
                            "ticker": sub.ticker,
                            "horizon": horizon,
                            "origin_date": sub.origin_date,
                            "target_end_date": sub.target_end_date,
                            "arm": arm,
                            "forecast": values,
                            "realized": sub.realized.to_numpy(),
                            "in_fold_val": val_mask,
                        }
                    )
                )
            fold_meta.setdefault(year, {})[horizon] = {
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
            }
            print(f"fold={year} h={horizon} train={len(train_idx)} val={len(val_idx)}", flush=True)

    scored = score_predictions(pd.concat(all_scored, ignore_index=True))
    scored.to_parquet(args.output_dir / "predictions.parquet", index=False)

    fold_reports = {}
    for year in folds:
        fold_reports[year] = {"train_val_rows": fold_meta[year], "horizons": {}}
        for horizon in panel.TARGET_HORIZONS:
            part = scored[(scored.fold == year) & (scored.horizon == horizon) & scored.in_fold_val]
            base = part[part.arm == "G0"][
                ["ticker", "origin_date", "qlike", "forecast", "realized"]
            ].rename(
                columns={
                    "qlike": "qlike_a",
                    "forecast": "forecast_base",
                    "realized": "realized_base",
                }
            )
            cand = part[part.arm == "G3"][
                ["ticker", "origin_date", "qlike", "mae_comp", "se_comp", "forecast", "realized"]
            ].rename(columns={"forecast": "forecast_cand", "realized": "realized_cand"})
            joined = base.merge(cand, on=["ticker", "origin_date"], how="inner")
            if len(joined) and not np.allclose(
                joined.realized_base.to_numpy(), joined.realized_cand.to_numpy(), rtol=0, atol=0
            ):
                raise ValueError(f"Realized-target mismatch in fold {year} h={horizon}")
            comparison = score_joined(
                joined,
                horizon,
                base_origins=int((part.arm == "G0").sum()),
                cand_origins=int((part.arm == "G3").sum()),
            )
            tail = tail_report(joined.assign(diff=joined.qlike_a - joined.qlike))
            fold_reports[year]["horizons"][horizon] = {
                "comparison_vs_G0": comparison,
                "tail": tail,
            }

    pooled = {}
    for horizon in panel.TARGET_HORIZONS:
        fold_parts = []
        for year in folds:
            part = scored[(scored.fold == year) & (scored.horizon == horizon) & scored.in_fold_val]
            fold_parts.append(part)
        pooled_part = pd.concat(fold_parts, ignore_index=True)
        base = pooled_part[pooled_part.arm == "G0"][
            ["ticker", "origin_date", "qlike", "forecast", "realized"]
        ].rename(
            columns={"qlike": "qlike_a", "forecast": "forecast_base", "realized": "realized_base"}
        )
        cand = pooled_part[pooled_part.arm == "G3"][
            ["ticker", "origin_date", "qlike", "mae_comp", "se_comp", "forecast", "realized"]
        ].rename(columns={"forecast": "forecast_cand", "realized": "realized_cand"})
        joined = base.merge(cand, on=["ticker", "origin_date"], how="inner")
        pooled[horizon] = score_joined(
            joined,
            horizon,
            base_origins=int((pooled_part.arm == "G0").sum()),
            cand_origins=int((pooled_part.arm == "G3").sum()),
        )

    promotion = _promotion_verdict(fold_reports)
    protocol = {
        "hypothesis": "rolling-origin regime replication: G3 vs G0",
        "folds": folds,
        "fold_rule": "calendar-year validation, expanding train from 2015, purge on both ends",
        "arm_matrix": {"G0": "rolling champion", "G3": "frozen GPU correction"},
        "target_horizons": list(panel.TARGET_HORIZONS),
        "primary_metric": "QLIKE",
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "promotion_rule": "pooled Δ>0 AND majority of folds positive AND no catastrophic "
        "fold AND secondaries supportive; no per-fold p<.05 required",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    report = {
        "status": "validation_complete",
        "test_scored": False,
        "folds": fold_reports,
        "pooled_vs_G0": pooled,
        "promotion": promotion,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for year in folds:
        for horizon in panel.TARGET_HORIZONS:
            cell = fold_reports[year]["horizons"][horizon]["comparison_vs_G0"]
            if cell.get("insufficient_overlap"):
                print(f"fold={year} h={horizon}: INSUFFICIENT OVERLAP", flush=True)
                continue
            hac0 = cell["hac"][0]
            print(
                f"fold={year} h={horizon} common={cell['coverage']['common_origins']} "
                f"dQLIKE={cell['mean_delta_qlike']:+.5f} rel={cell['relative_improvement']:+.4f} "
                f"p={hac0['p_two_sided']:.3g}",
                flush=True,
            )
    print(f"promotion: {promotion}", flush=True)
    if args.evaluate_test:
        _evaluate_test_once(args.output_dir, frames, spy)
    print(f"wrote {args.output_dir}", flush=True)
    return 0


def _evaluate_test_once(output_dir: Path, frames: dict, spy) -> None:
    """Train final G3 on all non-test rows; score held-out test origins once.

    Runs at most once per output dir: refuses if a test report already
    exists, so the untouched set cannot be peeked at repeatedly.
    """
    marker = output_dir / "test_report.json"
    if marker.exists():
        raise RuntimeError(f"Refusing to re-score the untouched test set ({marker} exists)")
    full = build_panel(frames, spy)
    test_rows = []
    for horizon in panel.TARGET_HORIZONS:
        sub = full[
            ["ticker", "origin_date", f"target_end_h{horizon}"]
            + list(FEATURE_COLUMNS)
            + [f"target_h{horizon}", f"base_h{horizon}"]
        ].copy()
        sub = sub.rename(
            columns={
                f"target_h{horizon}": "realized",
                f"base_h{horizon}": "base",
                f"target_end_h{horizon}": "target_end_date",
            }
        )
        usable = (
            np.isfinite(sub[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(sub.realized.to_numpy(dtype=float))
            & np.isfinite(sub.base.to_numpy(dtype=float))
        )
        sub = sub.loc[usable].reset_index(drop=True)
        _, _, _, reserve_start = panel.partitions(
            sub.origin_date.to_numpy(), sub.target_end_date.to_numpy()
        )
        reserve = np.datetime64(reserve_start)
        dates = pd.to_datetime(sub.origin_date).to_numpy(dtype="datetime64[ns]")
        ends = pd.to_datetime(sub.target_end_date).to_numpy(dtype="datetime64[ns]")
        fit_mask = ends < reserve
        test_mask = (dates >= reserve) & (ends <= dates.max())
        fit_idx, test_idx = np.where(fit_mask)[0], np.where(test_mask)[0]
        if len(fit_idx) < 1000 or len(test_idx) < 100:
            raise RuntimeError(f"Test evaluation too small at h={horizon}")
        x_fit = sub.loc[fit_idx, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        y_fit = sub.loc[fit_idx, "realized"].to_numpy(dtype=float)
        x_test = sub.loc[test_idx, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        log_base_fit = np.log(
            np.maximum(sub.loc[fit_idx, "base"].to_numpy(dtype=float), panel.QLIKE_FLOOR)
        )
        log_base_test = np.log(
            np.maximum(sub.loc[test_idx, "base"].to_numpy(dtype=float), panel.QLIKE_FLOOR)
        )
        booster = train_xgb(x_fit[:, G3_IDX], y_fit, base_margin=log_base_fit)
        pred = predict_xgb(booster, x_test[:, G3_IDX], base_margin=log_base_test)
        rows = pd.DataFrame(
            {
                "ticker": sub.loc[test_idx, "ticker"].to_numpy(),
                "horizon": horizon,
                "origin_date": sub.loc[test_idx, "origin_date"].to_numpy(),
                "target_end_date": sub.loc[test_idx, "target_end_date"].to_numpy(),
                "arm": "G3",
                "forecast": pred,
                "realized": sub.loc[test_idx, "realized"].to_numpy(),
            }
        )
        base_rows = pd.DataFrame(
            {
                "ticker": sub.loc[test_idx, "ticker"].to_numpy(),
                "horizon": horizon,
                "origin_date": sub.loc[test_idx, "origin_date"].to_numpy(),
                "target_end_date": sub.loc[test_idx, "target_end_date"].to_numpy(),
                "arm": "G0",
                "forecast": np.maximum(
                    sub.loc[test_idx, "base"].to_numpy(dtype=float), panel.QLIKE_FLOOR
                ),
                "realized": sub.loc[test_idx, "realized"].to_numpy(),
            }
        )
        test_rows.append(pd.concat([rows, base_rows], ignore_index=True))
    scored = score_predictions(pd.concat(test_rows, ignore_index=True))
    out = {}
    for horizon in panel.TARGET_HORIZONS:
        part = scored[scored.horizon == horizon]
        base = part[part.arm == "G0"][
            ["ticker", "origin_date", "qlike", "forecast", "realized"]
        ].rename(
            columns={"qlike": "qlike_a", "forecast": "forecast_base", "realized": "realized_base"}
        )
        cand = part[part.arm == "G3"][
            ["ticker", "origin_date", "qlike", "mae_comp", "se_comp", "forecast", "realized"]
        ].rename(columns={"forecast": "forecast_cand", "realized": "realized_cand"})
        joined = base.merge(cand, on=["ticker", "origin_date"], how="inner")
        out[horizon] = score_joined(
            joined,
            horizon,
            base_origins=int((part.arm == "G0").sum()),
            cand_origins=int((part.arm == "G3").sum()),
        )
    marker.write_text(
        json.dumps(
            {"status": "test_complete", "evaluations": 1, "results": out},
            indent=2,
            allow_nan=False,
        )
    )
    for horizon in panel.TARGET_HORIZONS:
        cell = out[horizon]
        hac0 = cell["hac"][0]
        print(
            f"TEST h={horizon} common={cell['coverage']['common_origins']} "
            f"dQLIKE={cell['mean_delta_qlike']:+.5f} rel={cell['relative_improvement']:+.4f} "
            f"p={hac0['p_two_sided']:.3g} MAE={cell['own_mae']:.6f}",
            flush=True,
        )


def _promotion_verdict(fold_reports: dict) -> dict:
    """Frozen promotion rule, computed — not judged — here."""
    pooled_delta, positive_folds, total_folds = 0.0, 0, 0
    worst_rel, catastrophic = 0.0, []
    for year, block in fold_reports.items():
        for horizon, cell in block["horizons"].items():
            comp = cell["comparison_vs_G0"]
            if comp.get("insufficient_overlap"):
                continue
            total_folds += 1
            pooled_delta += comp["mean_delta_qlike"]
            if comp["mean_delta_qlike"] > 0:
                positive_folds += 1
            rel = comp["relative_improvement"]
            if rel < worst_rel:
                worst_rel = rel
            if rel < -0.50:
                catastrophic.append({"fold": year, "horizon": horizon, "rel": rel})
    return {
        "pooled_delta_positive": bool(pooled_delta > 0),
        "majority_folds_positive": bool(total_folds and positive_folds > total_folds / 2),
        "worst_relative": worst_rel,
        "catastrophic_folds": catastrophic,
        "cells_evaluated": total_folds,
        "cells_positive": positive_folds,
    }


if __name__ == "__main__":
    raise SystemExit(main())
