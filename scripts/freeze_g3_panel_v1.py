"""Freeze the validated G3 volatility correction so it can be served.

G3 is defined as::

    forecast(h) = base(h) * exp(delta)
    base(h)     = h * trailing-20d close-to-close variance   (the G0 champion)
    delta       = XGBoost correction fitted on log-variance scale
                  with a QLIKE objective and base_margin = log(base(h))

The study in ``artifacts/gpu_vol_panel_v1`` established G3 on validation and the
rolling-origin study in ``artifacts/gpu_rolling_origin_v1`` scored it on the
untouched test partition. Neither study persisted the boosters, so the winning
model does not currently exist as a loadable artifact.

This script retrains G3 on the *same* train partition using the frozen
hyperparameters, persists one booster per horizon, and then verifies the
persisted boosters reproduce the predictions the study stored.

Faithfulness is the whole point. A model that merely resembles the winner must
not be served as the winner, so the script compares its own predictions against
``predictions.parquet`` row for row and refuses to mark the freeze faithful
unless the maximum relative deviation is within tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_gpu_vol_panel as study  # noqa: E402

from research.volatility_structure import panel  # noqa: E402

# The study trained on GPU. Reproducing it requires the same device, and
# agreement is always measured rather than assumed.
TOLERANCE_REL = 1e-6

TARGET_HORIZONS = panel.TARGET_HORIZONS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sub_frame(panel_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rebuild the exact per-horizon modelling frame used by the study."""
    cols = list(study.FEATURE_COLUMNS) + [f"target_h{horizon}", f"base_h{horizon}"]
    sub = panel_df[["ticker", "origin_date", f"target_end_h{horizon}"] + cols].copy()
    sub = sub.rename(
        columns={
            f"target_h{horizon}": "realized",
            f"base_h{horizon}": "base",
            f"target_end_h{horizon}": "target_end_date",
        }
    )
    usable = study._finite_rows(sub, list(study.FEATURE_COLUMNS) + ["realized", "base"])
    sub = sub.loc[usable].reset_index(drop=True)
    usable = (
        np.isfinite(sub[list(study.FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(sub.realized.to_numpy(dtype=float))
        & np.isfinite(sub.base.to_numpy(dtype=float))
    )
    return sub.loc[usable].reset_index(drop=True)


def _train_g3(x_train: np.ndarray, y_train: np.ndarray, base_margin: np.ndarray, device: str):
    import xgboost as xgb

    params = dict(study.XGB_PARAMS)
    params["device"] = device
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtrain.set_base_margin(np.asarray(base_margin, dtype=np.float64))
    return xgb.train(
        {**params, "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=study.N_ROUNDS,
        obj=study.qlike_objective(),
        verbose_eval=False,
    )


def _predict_g3(booster, x: np.ndarray, base_margin: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    dtest = xgb.DMatrix(x)
    dtest.set_base_margin(np.asarray(base_margin, dtype=np.float64))
    raw = booster.predict(dtest)
    return np.exp(np.clip(np.asarray(raw, dtype=np.float64), -30.0, 10.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=study.XGB_PARAMS.get("device", "cuda"),
        help="XGBoost device. Must match the study to reproduce it exactly.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gpu_vol_panel_v1" / "predictions.parquet",
        help="Stored study predictions used to verify the freeze.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    print("building panel ...", flush=True)
    frames = study._load_cache_frames("")
    spy = study._load_spy()
    panel_df = study.build_panel(frames, spy)
    print(f"panel rows={len(panel_df)} tickers={panel_df.ticker.nunique()}", flush=True)

    models = {}
    verification = {}
    faithful = True

    for horizon in TARGET_HORIZONS:
        sub = _sub_frame(panel_df, horizon)
        train, _, val_start, reserve_start = panel.partitions(
            sub.origin_date.to_numpy(), sub.target_end_date.to_numpy()
        )
        train_idx = np.where(train)[0]
        if len(train_idx) < 1000:
            raise RuntimeError(f"Too few training rows at h={horizon}")

        x_all = sub.loc[:, list(study.FEATURE_COLUMNS)].to_numpy(dtype=float)
        base_all = np.maximum(sub.base.to_numpy(dtype=float), panel.QLIKE_FLOOR)
        log_base = np.log(np.maximum(base_all, panel.QLIKE_FLOOR))

        booster = _train_g3(
            x_train=x_all[train_idx][:, study.G3_IDX],
            y_train=sub.realized.to_numpy(dtype=float)[train_idx],
            base_margin=log_base[train_idx],
            device=args.device,
        )
        path = args.output_dir / f"model_h{horizon}.ubj"
        booster.save_model(path)

        preds = _predict_g3(booster, x_all[:, study.G3_IDX], log_base)
        models[horizon] = {
            "path": str(path),
            "sha256": _sha256(path),
            "train_rows": int(len(train_idx)),
            "validation_start": val_start,
            "reserve_start": reserve_start,
        }
        print(f"h={horizon} train_rows={len(train_idx)} saved={path.name}", flush=True)

        verification[horizon] = _verify(sub, horizon, preds, args.reference)
        if not verification[horizon]["within_tolerance"]:
            faithful = False

    manifest = {
        "schema": "g3-panel-v1-freeze",
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": "G3",
        "definition": "forecast(h) = base(h) * exp(delta); base(h) = h * trailing-20d variance",
        "feature_schema": study.FEATURE_SCHEMA_VERSION,
        "features": study.G3_FEATURES,
        "n_features": len(study.G3_FEATURES),
        "xgboost_params": {**study.XGB_PARAMS, "device": args.device},
        "n_rounds": study.N_ROUNDS,
        "objective": "qlike(log-variance)",
        "train_device": args.device,
        "study_train_device": study.XGB_PARAMS.get("device"),
        "horizons": list(TARGET_HORIZONS),
        "models": models,
        "verification": verification,
        "faithful": bool(faithful),
        "reference_predictions": str(args.reference),
        "study_code_sha256": _sha256(REPO_ROOT / "scripts" / "train_gpu_vol_panel.py"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"faithful": faithful, "verification": verification}, indent=2))
    print(f"wrote {args.output_dir}")
    return 0 if faithful else 1


def _verify(sub, horizon, preds, reference_path):
    """Compare refitted predictions against those stored by the study."""
    result: dict[str, object] = {"reference": str(reference_path)}
    if not reference_path.exists():
        result.update(compared=0, within_tolerance=False, reason="reference missing")
        return result

    ref = pd.read_parquet(
        reference_path, columns=["ticker", "origin_date", "horizon", "arm", "forecast"]
    )
    ref = ref[(ref.horizon == horizon) & (ref.arm == "G3")]
    mine = pd.DataFrame(
        {
            "ticker": sub.ticker.to_numpy(),
            "origin_date": sub.origin_date.to_numpy(),
            "forecast": np.asarray(preds, dtype=float),
        }
    )
    merged = mine.merge(
        ref[["ticker", "origin_date", "forecast"]].rename(columns={"forecast": "ref"}),
        on=["ticker", "origin_date"],
        how="inner",
    )
    if merged.empty:
        result.update(compared=0, within_tolerance=False, reason="no overlapping rows")
        return result

    a = merged.forecast.to_numpy(dtype=float)
    b = merged.ref.to_numpy(dtype=float)
    denom = np.maximum(np.abs(b), 1e-12)
    rel = np.abs(a - b) / denom
    result.update(
        compared=int(len(merged)),
        max_abs_deviation=float(np.max(np.abs(a - b))),
        max_rel_deviation=float(np.max(rel)),
        mean_rel_deviation=float(np.mean(rel)),
        within_tolerance=bool(np.max(rel) <= TOLERANCE_REL),
        tolerance=TOLERANCE_REL,
    )
    return result


if __name__ == "__main__":
    sys.exit(main())
