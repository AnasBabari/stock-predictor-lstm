"""Matched Price-only vs Price+SPY-macro Ridge benchmark (US-only, frozen protocol).

Arms share identical rows/splits (asserted). Scaler fit on train only (via
fit_ridge_validation). Slices: A = US March-2023 validation origins,
B = full US validation. Ridge alpha=100.0, fit_intercept=True.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.price_forecasting.baselines import (  # noqa: E402
    evaluate_predictions,
    fit_ridge_validation,
    training_majority,
)
from research.price_forecasting.gpu_pipeline import (  # noqa: E402
    PriceTrainingConfig,
    build_global_price_dataset,
)
from research.price_forecasting.news_archive import (  # noqa: E402
    MACRO_NEWS_FEATURE_NAMES,
    apply_revision_policy,
    load_news_archive,
)

OUT_DIR = REPO_ROOT / "artifacts" / f"macro_ridge_{datetime.now(UTC).date().isoformat()}"
ALPHA = 100.0


def slice_report(actual, predicted, majority, stocks, label):
    rep = evaluate_predictions(actual, predicted, majority, stocks)
    pooled = dict(rep["pooled"])
    per_h = pooled.pop("per_horizon")
    return {
        "label": label,
        "origins": int(len(actual)),
        "mae_percent": pooled["mae_percent"],
        "rmse_percent": pooled["rmse_percent"],
        "persistence_mae_percent": pooled["persistence_mae_percent"],
        "relative_mae_vs_persistence": pooled["relative_mae_vs_persistence"],
        "direction_accuracy": pooled["direction_accuracy"],
        "majority_direction_accuracy": pooled["majority_direction_accuracy"],
        "per_horizon": [
            {
                "day": h["day"],
                "mae_percent": h["mae_percent"],
                "rmse_percent": h["rmse_percent"],
                "direction_accuracy": h["direction_accuracy"],
                "relative_mae_vs_persistence": h["relative_mae_vs_persistence"],
            }
            for h in per_h
        ],
    }


def main() -> int:
    cache = REPO_ROOT / "data" / "tri_exchange" / "cache"
    frames = {}
    for p in sorted(cache.glob("*.parquet")):
        if p.stem.endswith(".L"):
            continue
        frames[p.stem] = pd.read_parquet(p)
    print(f"US tickers: {len(frames)}", flush=True)

    raw_macro = load_news_archive(REPO_ROOT / "data" / "news" / "macro_spy_full" / "SPY.jsonl")
    macro_events, gate_diag = apply_revision_policy(raw_macro)
    print(f"macro gated kept={gate_diag['kept']} discarded={gate_diag['discarded']}", flush=True)

    cfg = PriceTrainingConfig()
    print("building price-only dataset...", flush=True)
    d_price = build_global_price_dataset(frames, cfg)
    print("building price+macro dataset...", flush=True)
    d_macro = build_global_price_dataset(
        frames, cfg, feature_mode="price_plus_macro", macro_events=macro_events
    )
    assert d_price.sequences.shape[0] == d_macro.sequences.shape[0]
    assert (d_price.split_train == d_macro.split_train).all()
    assert (d_price.split_validation == d_macro.split_validation).all()
    assert (d_price.origin_dates == d_macro.origin_dates).all()
    assert (d_price.ticker_indices == d_macro.ticker_indices).all()
    print("rows/splits/origins identical across arms", flush=True)

    rep_price = fit_ridge_validation(d_price, alpha=ALPHA)
    rep_macro = fit_ridge_validation(d_macro, alpha=ALPHA)

    majority = training_majority(d_price.targets[d_price.split_train])
    val = d_price.split_validation
    val_dates = pd.to_datetime(d_price.origin_dates[val])
    in_march = (val_dates >= "2023-03-01") & (val_dates < "2023-04-01")
    idx_a = np.where(in_march)[0]
    stocks_val = np.asarray(d_price.ticker_names)[d_price.ticker_indices[val]]
    assert not any(s.endswith(".L") for s in stocks_val), "US-only violated"
    print(f"slice A (US March 2023): n={len(idx_a)}", flush=True)

    def preds(rep, ds):
        x = ds.sequences[val, -1, :]
        sc = (x - np.asarray(rep["scaler_mean"])) / np.asarray(rep["scaler_scale"])
        return sc @ np.asarray(rep["coefficients"]).T + np.asarray(rep["intercept"])

    p_price, p_macro = preds(rep_price, d_price), preds(rep_macro, d_macro)
    y = d_price.targets[val]
    assert (y == d_macro.targets[val]).all()

    out = {
        "protocol": "matched_ridge_macro_v1",
        "alpha": ALPHA,
        "fit_intercept": True,
        "feature_counts": {"price_only": 25, "price_plus_macro": 35},
        "rows": {"train": int(len(d_price.split_train)), "validation": int(len(val))},
        "macro_gate": gate_diag,
        "arms": {},
    }
    for arm, pr in (("price_only", p_price), ("price_plus_macro", p_macro)):
        out["arms"][arm] = {
            "slice_B_full_validation": slice_report(y, pr, majority, stocks_val, "B"),
            "slice_A_march2023": slice_report(
                y[idx_a], pr[idx_a], majority, stocks_val[idx_a], "A"
            ),
        }

    coef = np.asarray(rep_macro["coefficients"])
    assert coef.shape == (7, 35)
    price_n, macro_n = coef[:, :25], coef[:, 25:]
    out["beta_diagnostics"] = {
        "coef_shape": list(coef.shape),
        "frobenius_all": float(np.linalg.norm(coef)),
        "frobenius_price": float(np.linalg.norm(price_n)),
        "frobenius_macro": float(np.linalg.norm(macro_n)),
        "macro_share_of_norm_sq": float(
            np.square(macro_n).sum() / np.square(coef).sum()
        ),
        "per_horizon_macro_norm": [
            float(np.linalg.norm(macro_n[h])) for h in range(7)
        ],
        "per_horizon_price_norm": [
            float(np.linalg.norm(price_n[h])) for h in range(7)
        ],
        "top_macro_weights": [
            {"feature": MACRO_NEWS_FEATURE_NAMES[j], "horizon_day": int(h) + 1, "w": float(macro_n[h, j])}
            for h, j in zip(
                *np.unravel_index(
                    np.argpartition(-np.abs(macro_n), 5, axis=None)[:5], macro_n.shape
                )
            )
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUT_DIR / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for arm in out["arms"]:
        for sl in ("slice_A_march2023", "slice_B_full_validation"):
            s = out["arms"][arm][sl]
            print(
                f"{arm:16s} {sl:22s} n={s['origins']:6d} MAE={s['mae_percent']:.4f} "
                f"RMSE={s['rmse_percent']:.4f} rel={s['relative_mae_vs_persistence']:.6f} "
                f"dir={s['direction_accuracy']:.4f} (maj={s['majority_direction_accuracy']:.4f})",
                flush=True,
            )
    b = out["beta_diagnostics"]
    print(
        f"beta ||all||_F={b['frobenius_all']:.4f} ||price||_F={b['frobenius_price']:.4f} "
        f"||macro||_F={b['frobenius_macro']:.6f} macro_share={b['macro_share_of_norm_sq']:.6f}",
        flush=True,
    )
    print(f"wrote {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
