"""Export the hash-verified, QLIKE-trained G3 boosters without retraining.

Graph output includes XGBoost's default intercept. Serving replaces that
intercept with log(rolling base), then clips the COMPLETE log variance.
Parity is measured on final variances with nonzero base margins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from services.g3_volatility import G3_FEATURES, build_g3_features, panel_rolling_base  # noqa: E402

PARITY_TOL = 5e-5


def export_verified(source: Path, frame: pd.DataFrame, output: Path) -> dict:
    import onnxruntime as ort
    import xgboost as xgb
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("faithful") is not True or manifest.get("objective") != "qlike(log-variance)":
        raise ValueError("A faithful QLIKE freeze is required")
    if manifest.get("features") != list(G3_FEATURES):
        raise ValueError("Feature order mismatch")
    features = build_g3_features(frame).dropna().to_numpy(dtype=np.float32)
    if len(features) < 20:
        raise ValueError("Insufficient parity rows")
    pending = []
    audit = {}
    for horizon in (5, 10, 20):
        info = manifest["models"][str(horizon)]
        path = source / f"model_h{horizon}.ubj"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            digest != info["sha256"]
            or not manifest["verification"][str(horizon)]["within_tolerance"]
        ):
            raise ValueError(f"Unverified booster h{horizon}")
        booster = xgb.Booster()
        booster.load_model(path)
        booster.set_param({"device": "cpu"})
        config = json.loads(booster.save_config())
        raw_score = config["learner"]["learner_model_param"]["base_score"]
        base_score = float(str(raw_score).strip("[]"))
        graph = convert_xgboost(
            booster,
            initial_types=[("input", FloatTensorType([None, len(G3_FEATURES)]))],
            target_opset=15,
        )
        graph_bytes = graph.SerializeToString()
        session = ort.InferenceSession(graph_bytes, providers=["CPUExecutionProvider"])
        got = session.run(None, {"input": features})[0].ravel().astype(float)
        # Vary B across realistic and extreme scales: catches doubled/omitted intercepts.
        bases = np.geomspace(1e-8, 0.1, len(features))
        dm = xgb.DMatrix(features, base_margin=np.log(bases))
        reference = np.exp(np.clip(booster.predict(dm, output_margin=True).astype(float), -30, 10))
        converted = np.exp(np.clip(got - base_score + np.log(bases), -30, 10))
        rel = float(np.max(np.abs(converted / reference - 1)))
        if not np.isfinite(rel) or rel > PARITY_TOL:
            raise ValueError(f"Final variance parity failed: {rel}")
        last_base = panel_rolling_base(frame, horizon)
        last = features[-1:]
        reference_last = float(
            np.exp(
                np.clip(
                    float(
                        booster.predict(
                            xgb.DMatrix(last, base_margin=[np.log(last_base)]), output_margin=True
                        )[0]
                    ),
                    -30,
                    10,
                )
            )
        )
        variance = float(np.exp(np.clip(float(got[-1]) - base_score + np.log(last_base), -30, 10)))
        meta = {
            "model": "G3",
            "horizon": horizon,
            "features": list(G3_FEATURES),
            "serving_contract": "g3-qlike-base-margin-v2",
            "graph_base_score": base_score,
            "onnx_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "booster_sha256": digest,
            "freeze_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "objective": manifest["objective"],
            "metric_source": "validation_panel",
            "source_study": "artifacts/gpu_vol_panel_v1",
            "source_freeze": "artifacts/g3_panel_v1",
            "train_rows": info["train_rows"],
            "validation_start": info["validation_start"],
            "reserve_start": info["reserve_start"],
            "parity_rows": len(features),
            "parity_max_relative_variance_error": rel,
            "inference": "exp(clip(onnx(x) - graph_base_score + log(B), -30, 10))",
        }
        audit[str(horizon)] = {
            "rolling_cumulative_variance": last_base,
            "original_booster_variance": reference_last,
            "onnx_variance": variance,
            "annualized_sigma": float(np.sqrt(variance * 252 / horizon)),
            "max_relative_variance_error": rel,
        }
        pending.append((horizon, graph_bytes, meta))
    # Nothing is published until ALL horizons pass parity.
    output.mkdir(parents=True, exist_ok=True)
    for horizon, graph_bytes, meta in pending:
        (output / f"g3_h{horizon}.onnx").write_bytes(graph_bytes)
        (output / f"g3_h{horizon}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/g3_panel_v1")
    parser.add_argument(
        "--ohlcv",
        type=Path,
        required=True,
        help="Local OHLCV CSV or Parquet used only for numerical parity; no fitting",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "backend/volatility_models")
    args = parser.parse_args()
    if args.ohlcv.suffix.lower() == ".parquet":
        frame = pd.read_parquet(args.ohlcv)
        frame.index = pd.to_datetime(frame.index)
    else:
        frame = pd.read_csv(args.ohlcv, index_col=0, parse_dates=True)
    print(json.dumps(export_verified(args.source, frame, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
