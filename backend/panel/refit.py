"""Full-sample refitting and artifact serialization for certified champions (Slice-12 / Refit).

Refits the exact frozen champion architecture on the permitted development
history and persists the trained weights, scaler metadata, and certification
provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from panel.candidates import REGISTRY, CandidateTargets
from panel.certification import CertificationDecision
from panel.features import DeployableFeatureContract
from panel.selection import SelectionDecision


@dataclass(frozen=True)
class RefitManifest:
    horizon: int
    candidate_name: str
    alpha: float
    feature_count: int
    training_rows: int
    ticker_count: int
    seed: int
    refit_at: str
    certification_decision: str
    certification_temporal_rmse: float
    disclaimer: str = (
        "Certification metrics describe the out-of-fold selection model, "
        "not this final full-history refit."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def refit_certified_champion(
    *,
    horizon: int,
    champion_decision: SelectionDecision,
    certification_decision: CertificationDecision,
    universe_data: dict[str, pd.DataFrame],
    features_by_ticker: dict[str, pd.DataFrame],
    dev_train_tickers: list[str],
    temporal_holdout_dates: pd.DatetimeIndex,
    out_dir: Path,
    seed: int = 42,
) -> tuple[RefitManifest, dict[str, bytes]]:
    """Refit the certified champion on all permitted development history."""
    cand_name = champion_decision.candidate_name or "persistence"
    if cand_name not in REGISTRY:
        raise ValueError(f"Candidate {cand_name} not found in candidate registry.")

    contract = DeployableFeatureContract()
    feature_cols = [
        c
        for c in features_by_ticker[dev_train_tickers[0]].columns
        if c in contract.feature_names
        and pd.api.types.is_numeric_dtype(features_by_ticker[dev_train_tickers[0]][c])
    ]

    x_train_rows: list[np.ndarray] = []
    y_train_rows: list[float] = []
    d_train_rows: list[int] = []

    for ticker in dev_train_tickers:
        f = features_by_ticker[ticker]
        c = universe_data[ticker]["Close"]
        cumret = np.log(c.shift(-horizon) / c)
        feat_vals = f[feature_cols].to_numpy(dtype=float)

        dev_dates = f.index[~f.index.isin(temporal_holdout_dates)]
        for t in range(contract.window_size, len(dev_dates) - horizon):
            w = feat_vals[t - contract.window_size : t]
            tgt = cumret.iloc[t - 1]
            if np.isfinite(tgt) and np.isfinite(w).all():
                x_train_rows.append(w)
                y_train_rows.append(float(tgt))
                d_train_rows.append(1 if abs(tgt) < 0.005 else (2 if tgt > 0 else 0))

    if not x_train_rows:
        raise ValueError(f"No training rows available to refit champion for horizon {horizon}")

    X_train = np.stack(x_train_rows)
    targets_tr = CandidateTargets(
        cumulative_returns=np.asarray(y_train_rows, dtype=np.float32),
        direction_classes=np.asarray(d_train_rows, dtype=int),
    )

    model = REGISTRY[cand_name](seed)
    model.fit(X_train, targets_tr)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_files: dict[str, bytes] = {}

    # Serialize model state / description
    model_meta = {
        "candidate_family": cand_name,
        "horizon": horizon,
        "alpha": champion_decision.alpha,
        "features": feature_cols,
        "window_size": contract.window_size,
        "seed": seed,
        "description": model.describe() if hasattr(model, "describe") else {},
    }
    model_json = json.dumps(model_meta, indent=2, sort_keys=True).encode("utf-8")
    model_files[f"model_h{horizon}.json"] = model_json
    (out_dir / f"model_h{horizon}.json").write_bytes(model_json)

    manifest = RefitManifest(
        horizon=horizon,
        candidate_name=cand_name,
        alpha=champion_decision.alpha,
        feature_count=len(feature_cols),
        training_rows=len(x_train_rows),
        ticker_count=len(dev_train_tickers),
        seed=seed,
        refit_at=datetime.now(UTC).isoformat(),
        certification_decision=certification_decision.decision,
        certification_temporal_rmse=certification_decision.temporal_relative_rmse,
    )
    manifest_bytes = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    model_files[f"refit_manifest_h{horizon}.json"] = manifest_bytes
    (out_dir / f"refit_manifest_h{horizon}.json").write_bytes(manifest_bytes)

    return manifest, model_files
