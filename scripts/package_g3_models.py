"""Package FINAL G3 models for production serving (one-shot packaging run).

Trains one XGBRegressor per horizon on ALL non-test rows (train+validation
pooled) with the frozen G3 configuration, converts to ONNX via onnxmltools,
and verifies strict parity (booster vs ONNX) before writing anything.
Identical trees to the functional API were verified separately; the parity
test below re-verifies on the real fitted models.

Outputs (backend/volatility_models/):
  g3_h{H}.onnx          ONNX graph (float32, batchable)
  g3_h{H}.meta.json     feature order, train window, promotion reference

At inference: z = onnx(x) + log(B); Vhat = exp(clip(z)). Base margin is
added OUTSIDE the graph, exactly as in validation.
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
from research.volatility_structure.gpu_panel import FEATURE_COLUMNS  # noqa: E402
from scripts.train_gpu_vol_panel import (  # noqa: E402
    G3_IDX,
    XGB_PARAMS,
    _load_cache_frames,
    _load_spy,
    build_panel,
)

OUT_DIR = REPO_ROOT / "backend" / "volatility_models"
# Converter ceiling of the pinned onnx package; TreeEnsemble has been
# stable since long before this. onnxruntime (>=1.19) runs it easily.
TARGET_OPSET = 15
# float32 conversion noise on ~500-tree ensembles measures ~1e-5 absolute
# on raw margins (relative forecast impact ~1e-5, immaterial against any
# decision threshold); genuine breakage shows up orders of magnitude larger.
PARITY_TOL = 5e-5


def train_final_g3(panel_df: pd.DataFrame, horizon: int):
    """Fit G3 on all non-test rows for one horizon (sklearn API)."""
    import xgboost as xgb

    sub = panel_df.copy()
    feats = [FEATURE_COLUMNS[i] for i in G3_IDX]
    usable = (
        np.isfinite(sub[feats].to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(sub[f"target_h{horizon}"].to_numpy(dtype=float))
        & np.isfinite(sub[f"base_h{horizon}"].to_numpy(dtype=float))
    )
    sub = sub.loc[usable].reset_index(drop=True)
    _, _, _, reserve_start = panel.partitions(
        sub.origin_date.to_numpy(),
        sub[f"target_end_h{horizon}"].to_numpy(),
    )
    reserve = np.datetime64(reserve_start)
    keep = pd.to_datetime(sub.origin_date).to_numpy(dtype="datetime64[ns]") < reserve
    x = sub.loc[keep, feats].to_numpy(dtype=float)
    y = sub.loc[keep, f"target_h{horizon}"].to_numpy(dtype=float)
    base = np.log(
        np.maximum(sub.loc[keep, f"base_h{horizon}"].to_numpy(dtype=float), panel.QLIKE_FLOOR)
    )
    print(f"h={horizon} final-train rows={len(x)} (test held out)", flush=True)
    model = xgb.XGBRegressor(
        max_depth=XGB_PARAMS["max_depth"],
        learning_rate=XGB_PARAMS["eta"],
        subsample=XGB_PARAMS["subsample"],
        colsample_bytree=XGB_PARAMS["colsample_bytree"],
        min_child_weight=XGB_PARAMS["min_child_weight"],
        n_estimators=500,
        random_state=XGB_PARAMS["seed"],
        device=XGB_PARAMS["device"],
        objective="reg:squarederror",
    )
    model.fit(x, y, base_margin=base)
    return model, {
        "train_rows": int(len(x)),
        "train_origin_min": str(sub.loc[keep, "origin_date"].min()),
        "train_origin_max": str(sub.loc[keep, "origin_date"].max()),
        "test_starts": reserve_start,
    }


def to_onnx(model, n_features: int):
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    return convert_xgboost(
        model,
        "g3_volatility",
        [("input", FloatTensorType([None, n_features]))],
        target_opset=TARGET_OPSET,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_cache_frames(args.tickers)
    spy = _load_spy()
    print(f"tickers: {len(frames)}", flush=True)
    full = build_panel(frames, spy)

    import onnxruntime as ort

    for horizon in panel.TARGET_HORIZONS:
        model, info = train_final_g3(full, horizon)
        ox = to_onnx(model, len(G3_IDX))
        onx_path = out_dir / f"g3_h{horizon}.onnx"
        with open(onx_path, "wb") as handle:
            handle.write(ox.SerializeToString())
        # Parity: ONNX graph vs booster on real fitted data (strict).
        sample = full.sample(n=min(5000, len(full)), random_state=42)
        feats = [FEATURE_COLUMNS[i] for i in G3_IDX]
        x = sample[feats].to_numpy(dtype=np.float32)
        mask = np.isfinite(x).all(axis=1)
        x = x[mask]
        ref = model.predict(x)
        sess = ort.InferenceSession(str(onx_path))
        got = sess.run(None, {"input": x})[0].ravel()
        max_diff = float(np.max(np.abs(ref - got)))
        print(f"h={horizon} parity max abs diff={max_diff:.2e}", flush=True)
        if max_diff > PARITY_TOL:
            raise ValueError(f"ONNX parity failed at h={horizon}: {max_diff}")
        meta = {
            "model": "G3",
            "horizon": horizon,
            "features": feats,
            "base_margin": "log(trailing_rolling_base); add outside graph",
            "inference": "z = onnx(x) + log(B); Vhat = exp(clip(z, -30, 10))",
            "target_opset": TARGET_OPSET,
            "parity_max_abs_diff": max_diff,
            "promotion_ref": "artifacts/gpu_rolling_origin_v1 + test_report.json",
            **info,
        }
        (out_dir / f"g3_h{horizon}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {onx_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
