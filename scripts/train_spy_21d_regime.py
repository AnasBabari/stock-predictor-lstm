"""21-trading-day SPY regime replication (market-dailies-only, validation-only).

Target: y_t = ln(P_{t+21} / P_t) on dividend-adjusted SPY closes.
Splits are positional with purge-by-construction: train rows satisfy
origin + 21 < train_boundary; validation rows satisfy origin >=
train_boundary and origin + 21 < test_boundary. Test rows are never read.
Benchmarks: persistence (0), train-majority direction, Ridge(alpha=100),
depth-2 XGBoost with early stopping on a purged train-tail slice.
Inference: Bartlett HAC (L in {20, 40, 60}) on daily loss differentials.
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

from research.price_forecasting.paired_validation import hac_mean  # noqa: E402

HORIZON = 21
KMEANS_K = 4
OUT_DIR = REPO_ROOT / "artifacts" / f"spy_21d_regime_{datetime.now(UTC).date().isoformat()}"


def rget(df: pd.DataFrame, col: str, w: int) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(
            np.log(df[col].to_numpy(float) / df[col].shift(w).to_numpy(float)),
            index=df.index,
        )


def build_features(px: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    f: dict[str, pd.Series] = {}
    spy_ret = np.log(px["SPY"] / px["SPY"].shift(1))
    f["spy_ret_5"] = rget(px, "SPY", 5)
    f["spy_ret_21"] = rget(px, "SPY", 21)
    f["spy_ret_63"] = rget(px, "SPY", 63)
    f["spy_vol_21"] = spy_ret.rolling(21).std()
    f["spy_dd_63"] = px["SPY"] / px["SPY"].rolling(63).max() - 1.0
    f["vix_level"] = px["^VIX"]
    f["vix_chg_5"] = px["^VIX"] / px["^VIX"].shift(5) - 1.0
    f["vix_chg_21"] = px["^VIX"] / px["^VIX"].shift(21) - 1.0
    f["tnx_level"] = px["^TNX"]
    f["tnx_chg_21"] = px["^TNX"] / px["^TNX"].shift(21) - 1.0
    f["oil_ret_21"] = rget(px, "CL=F", 21)
    f["oil_ret_63"] = rget(px, "CL=F", 63)
    f["dxy_ret_21"] = rget(px, "DX-Y.NYB", 21)
    f["dxy_ret_63"] = rget(px, "DX-Y.NYB", 63)
    f["hyg_ret_21"] = rget(px, "HYG", 21)
    f["credit_stress_21"] = rget(px, "HYG", 21) - rget(px, "SPY", 21)
    f["hyg_dd_63"] = px["HYG"] / px["HYG"].rolling(63).max() - 1.0
    frame = pd.DataFrame(f, index=px.index)
    return frame, list(frame.columns)


REGIME_COLS = [
    "spy_ret_21",
    "spy_ret_63",
    "spy_vol_21",
    "spy_dd_63",
    "vix_chg_21",
    "tnx_chg_21",
    "credit_stress_21",
]


def pct_errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        err = 100 * (np.exp(predicted) - np.exp(actual))
        base = 100 * (1 - np.exp(actual))
    if not np.isfinite(err).all() or not np.isfinite(base).all():
        raise ValueError("Nonfinite 21-day errors")
    return err, base


def summarize(actual, predicted, majority_sign, label):
    err, base = pct_errors(actual, predicted)
    mae, rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))
    bmae = float(np.mean(np.abs(base)))
    correct = (np.sign(predicted) == np.sign(actual)) & (actual != 0)
    maj_ok = (np.sign(actual) == majority_sign) & (actual != 0)
    return {
        "label": label,
        "origins": int(len(actual)),
        "mae_percent": mae,
        "rmse_percent": rmse,
        "persistence_mae_percent": bmae,
        "relative_mae_vs_persistence": mae / bmae if bmae > 0 else None,
        "direction_accuracy": float(correct.mean()),
        "majority_direction_accuracy": float(maj_ok.mean()),
    }


def main() -> int:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    px = pd.read_parquet(REPO_ROOT / "data" / "macro" / "market_dailies.parquet")
    feats, feat_names = build_features(px)
    log_adj = np.log(px["SPY_ADJ"].to_numpy(dtype=float))
    target = np.array(
        [float(log_adj[i + HORIZON] - log_adj[i]) for i in range(len(px) - HORIZON)]
    )
    # Align features to origins that admit a full 21-day forward window.
    feats = feats.iloc[: len(px) - HORIZON]
    origin_dates = feats.index
    feats = feats.dropna()
    target = target[feats.index.map(lambda d: origin_dates.get_loc(d)).to_numpy()]
    if not np.isfinite(feats.to_numpy(dtype=float)).all():
        raise ValueError("NaN in feature matrix after warmup trim")
    n = len(feats)
    b70, b85 = int(n * 0.70), int(n * 0.85)

    train_idx = np.array([i for i in range(n) if i + HORIZON < b70])
    val_idx = np.array([i for i in range(n) if i >= b70 and i + HORIZON < b85])
    # Purged early-stopping slice: train tail whose labels end before b70.
    estop_idx = np.array([i for i in train_idx if i >= b70 - HORIZON - 63])
    fit_idx = np.array([i for i in train_idx if i < b70 - HORIZON - 63])
    assert len(train_idx) and len(val_idx) and len(estop_idx) and len(fit_idx)
    assert int(train_idx.max()) + HORIZON < b70, "purge violated on train tail"
    assert int(val_idx.min()) >= b70
    # Test rows (i >= b85) are never indexed below this line.

    # Regime discovery on train only.
    scaler_r = StandardScaler().fit(feats.iloc[train_idx][REGIME_COLS].to_numpy(float))
    km = KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10)
    km.fit(scaler_r.transform(feats.iloc[train_idx][REGIME_COLS].to_numpy(float)))
    regimes = km.predict(scaler_r.transform(feats[REGIME_COLS].to_numpy(float)))
    dummies = np.zeros((n, KMEANS_K))
    dummies[np.arange(n), regimes] = 1.0
    x_all = np.hstack([feats.to_numpy(float), dummies])
    all_names = feat_names + [f"regime_{k}" for k in range(KMEANS_K)]

    majority = 1 if (target[train_idx] > 0).sum() >= (target[train_idx] < 0).sum() else -1
    scaler = StandardScaler().fit(x_all[fit_idx])
    x_train, x_val, x_stop = scaler.transform(x_all[fit_idx]), scaler.transform(x_all[val_idx]), scaler.transform(x_all[estop_idx])
    y_train, y_val = target[fit_idx], target[val_idx]

    ridge = Ridge(alpha=100.0, fit_intercept=True, solver="cholesky").fit(x_train, y_train)
    pred_ridge = ridge.predict(x_val)

    import xgboost as xgb

    # Functional API: sklearn fit() no longer accepts early-stopping kwargs in 3.x.
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dstop = xgb.DMatrix(x_stop, label=target[estop_idx])
    bst = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "seed": 42,
            "nthread": -1,
        },
        dtrain,
        num_boost_round=300,
        evals=[(dstop, "estop")],
        callbacks=[xgb.callback.EarlyStopping(rounds=15, save_best=True)],
        verbose_eval=False,
    )
    pred_xgb = bst.predict(xgb.DMatrix(x_val))
    best_iter = int(bst.best_iteration)
    gain = bst.get_score(importance_type="gain")
    fi = np.array([float(gain.get(f"f{i}", 0.0)) for i in range(len(all_names))])

    preds = {
        "persistence": np.zeros_like(y_val),
        "ridge": pred_ridge,
        "xgboost": pred_xgb,
    }
    summaries = {
        name: summarize(y_val, p, majority, name) for name, p in preds.items()
    }
    err = {name: pct_errors(y_val, p)[0] for name, p in preds.items()}
    hac = {}
    for ref, cand in [("persistence", "ridge"), ("persistence", "xgboost"), ("ridge", "xgboost")]:
        diff = np.abs(err[ref]) - np.abs(err[cand])
        daily = pd.Series(diff, index=pd.to_datetime(origin_dates[val_idx])).groupby(level=0).mean().sort_index()
        hac[f"{ref}_vs_{cand}"] = {
            f"bandwidth_{L}": hac_mean(daily.to_numpy(), L) for L in (20, 40, 60)
        }

    ridge_coef = np.asarray(ridge.coef_)
    fi = np.asarray(fi, dtype=float)
    if fi.sum() <= 0:
        fi = np.zeros(len(all_names))
    out = {
        "protocol": "spy_21d_regime_market_only_v1",
        "horizon": HORIZON,
        "target": "ln(SPY_ADJ[t+21]/SPY_ADJ[t])",
        "macro_mode": "market_dailies_only",
        "monthly_revised_block": "excluded",
        "rows": {"features": n, "train_fit": int(len(fit_idx)), "early_stop": int(len(estop_idx)), "validation": int(len(val_idx))},
        "purge_check": {"max_train_origin_plus_21_lt_b70": bool(int(train_idx.max()) + HORIZON < b70), "test_scored": False},
        "majority_direction_train": majority,
        "kmeans_k": KMEANS_K,
        "feature_names": all_names,
        "summaries": summaries,
        "hac_loss_improvement_positive_favors_candidate": hac,
        "ridge_coef_norm": float(np.linalg.norm(ridge_coef)),
        "xgb_best_iteration": best_iter,
        "xgb_gain_top5": [
            {"feature": all_names[i], "gain": float(fi[i])}
            for i in np.argsort(-np.asarray(fi, dtype=float))[:5]
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUT_DIR / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, s in summaries.items():
        print(
            f"{name:12s} n={s['origins']:4d} MAE={s['mae_percent']:.4f} RMSE={s['rmse_percent']:.4f} "
            f"rel={s['relative_mae_vs_persistence']:.5f} dir={s['direction_accuracy']:.4f} "
            f"(maj={s['majority_direction_accuracy']:.4f})",
            flush=True,
        )
    for pair, bands in hac.items():
        b20 = bands["bandwidth_20"]
        print(
            f"HAC {pair}: L20 mean={b20['mean_loss_improvement']:.5f} "
            f"ci95=[{b20['ci95'][0]:.5f},{b20['ci95'][1]:.5f}] p={b20['p_two_sided']}",
            flush=True,
        )
    print(f"xgb best_iteration={best_iter} ridge|coef|={out['ridge_coef_norm']:.4f}", flush=True)
    print(f"wrote {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
