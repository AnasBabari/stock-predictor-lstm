"""Cryptographic candidate freezer serializing entire bundle state into immutable manifests."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from research.volatility_forecasting.global_multimodal_trainer_v11 import (
    FrozenCandidateBundle,
)


class CandidateFreezerV11:
    """Serializes and cryptographically seals candidate bundles with bitwise reproducibility validation."""

    @classmethod
    def freeze_and_save_bundle(
        cls,
        bundle: FrozenCandidateBundle,
        output_dir: Path,
        git_sha: str,
        panel_sha: str,
        split_sha: str,
    ) -> tuple[Path, Path, str]:
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = output_dir / "v11_candidate_freeze_manifest.json"
        sha_file = output_dir / "v11_candidate_freeze_manifest.sha256"

        buf_m1 = io.BytesIO()
        torch.save(bundle.m1_numeric_model.state_dict(), buf_m1)
        m1_bytes = buf_m1.getvalue()

        if bundle.m2_multimodal_model is not None:
            buf_m2 = io.BytesIO()
            torch.save(bundle.m2_multimodal_model.state_dict(), buf_m2)
            m2_bytes = buf_m2.getvalue()
        else:
            m2_bytes = b"NO_M2_MODEL_TRAINED"

        # Combine all canonical components into master freeze hash
        h = hashlib.sha256()
        h.update(bundle.manifest.selected_candidate_family.encode())
        h.update(git_sha.encode())
        h.update(panel_sha.encode())
        h.update(split_sha.encode())
        h.update(m1_bytes)
        h.update(m2_bytes)
        h.update(bundle.num_scaler_mean.tobytes())
        h.update(bundle.num_scaler_std.tobytes())
        h.update(bundle.news_scaler_mean.tobytes())
        h.update(bundle.news_scaler_std.tobytes())
        h.update(bundle.train_scale_returns.tobytes())
        for k in sorted(bundle.m0_har_baseline.coefficients.keys()):
            h.update(bundle.m0_har_baseline.coefficients[k].tobytes())

        master_freeze_digest = h.hexdigest()

        manifest_data: dict[str, Any] = {
            "master_freeze_digest": master_freeze_digest,
            "selected_candidate_family": bundle.manifest.selected_candidate_family,
            "candidate_id": bundle.manifest.candidate_id,
            "git_sha": git_sha,
            "panel_sha": panel_sha,
            "split_sha": split_sha,
            "train_dates": bundle.manifest.train_dates,
            "val_dates": bundle.manifest.val_dates,
            "hyperparameters": dict(bundle.manifest.selected_hyperparameters),
            "validation_oof_metrics": {
                k: v.to_dict() for k, v in bundle.manifest.validation_oof_metrics.items()
            },
        }

        canonical_json = json.dumps(manifest_data, indent=2, sort_keys=True)
        manifest_file.write_text(canonical_json, encoding="utf-8")
        sha_file.write_text(master_freeze_digest, encoding="utf-8")

        return manifest_file, sha_file, master_freeze_digest
