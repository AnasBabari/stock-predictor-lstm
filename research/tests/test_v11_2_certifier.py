"""Focused one-shot V11.2 certification checks using a synthetic sealed panel."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from research.volatility_forecasting.model import RobustSequenceScaler
from research.volatility_forecasting.v11_2_freezer import V112Route, freeze_routing_bundle
from research.volatility_forecasting.v11_2_protocol import (
    V112Protocol,
    canonical_json_digest,
    feature_schema_digest,
    protocol_manifest,
)
from research.volatility_forecasting.v11_2_sealed_store import seal_v112_dataset
from research.volatility_forecasting.v11_2_split import create_v112_split
from scripts.certify_v11_2_candidate import certify


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    protocol = V112Protocol()
    dates = [
        (dt.date(2020, 1, 1) + dt.timedelta(days=index)).isoformat()
        for index in range(500)
        for _ in range(2)
    ]
    security_ids = [f"SEC-{index % 2}" for index in range(len(dates))]
    rng = np.random.default_rng(42)
    features = np.zeros(
        (len(dates), protocol.window_size, len(protocol.feature_names)), dtype=np.float32
    )
    features[:, :, 0] = 0.01
    features[:, :, 13] = 0.05
    features[:, :, 15] = 0.06
    features[:, :, 16] = 0.07
    features += rng.normal(0.0, 1e-4, size=features.shape).astype(np.float32)
    returns = np.zeros((len(dates), len(protocol.horizons)), dtype=np.float32)
    rv = np.full_like(returns, 0.01)
    split = create_v112_split(dates, security_ids)
    dataset = tmp_path / "dataset"
    key_path = tmp_path / "private" / "v11_2.key"
    metadata = seal_v112_dataset(
        dates=dates,
        features=features,
        returns=returns,
        rv=rv,
        split=split,
        output_dir=dataset,
        panel_sha256="a" * 64,
        schema_sha256=feature_schema_digest(protocol),
        key_path=key_path,
        repository_root=tmp_path / "repository",
    )
    (dataset / "manifests" / "protocol.json").write_text(
        json.dumps(protocol_manifest(protocol), sort_keys=True), encoding="utf-8"
    )
    universe_body = {
        "protocol_id": protocol.protocol_id,
        "universe_version": "synthetic-v11.2",
        "selection_method": "synthetic_fixture",
        "membership_sources": ["synthetic_fixture"],
        "securities": [{"security_id": f"SECURITY-{index:03d}"} for index in range(64)],
    }
    universe_payload = {
        **universe_body,
        "universe_size": 64,
        "manifest_sha256": canonical_json_digest(universe_body),
    }
    (dataset / "manifests" / "universe.json").write_text(
        json.dumps(universe_payload, sort_keys=True), encoding="utf-8"
    )
    (dataset / "manifests" / "split.json").write_text(
        json.dumps(split.to_dict(), sort_keys=True), encoding="utf-8"
    )
    results = tmp_path / "results"
    results.mkdir()
    scaler = RobustSequenceScaler.fit(features[: split.train_rows])
    scaler_payload = scaler.to_dict()
    (results / "numeric_scaler.json").write_text(
        json.dumps(scaler_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    routes: list[V112Route] = []
    evidence_digests: list[str] = []
    for horizon in protocol.horizons:
        artifact = results / "baselines" / f"har_horizon_{horizon}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps({"family": "M0_HAR_BASELINE", "horizon": horizon}), encoding="utf-8"
        )
        selection = results / f"selection_horizon_{horizon}.json"
        selection.write_text(
            json.dumps({"horizon": horizon, "selected_family": "M0_HAR_BASELINE"}), encoding="utf-8"
        )
        for seed in protocol.seeds:
            body = {"horizon": horizon, "seed": seed, "family": "M1_NUMERIC_RESIDUAL"}
            body["evidence_sha256"] = canonical_json_digest({k: v for k, v in body.items()})
            evidence_path = results / "seed_evidence" / f"horizon_{horizon}" / f"seed_{seed}.json"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
            evidence_digests.append(body["evidence_sha256"])
        routes.append(
            V112Route(
                horizon=horizon,
                family="M0_HAR_BASELINE",
                model_digest=__import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
                scaler_digest=__import__("hashlib")
                .sha256((results / "numeric_scaler.json").read_bytes())
                .hexdigest(),
                selection_record_digest=__import__("hashlib")
                .sha256(selection.read_bytes())
                .hexdigest(),
                learned_promotion=False,
                artifact_path=f"baselines/har_horizon_{horizon}.json",
            )
        )
    comparison_body = {
        "protocol_id": protocol.protocol_id,
        "sealed_test_status": "LOCKED_UNOPENED",
    }
    comparison_body["report_sha256"] = canonical_json_digest(comparison_body)
    comparison_path = results / "v11_2_development_model_comparison.json"
    comparison_path.write_text(json.dumps(comparison_body, sort_keys=True), encoding="utf-8")
    freeze_routing_bundle(
        protocol=protocol,
        universe_sha256=universe_payload["manifest_sha256"],
        panel_sha256="a" * 64,
        schema_sha256=feature_schema_digest(protocol),
        split_sha256=split.split_sha256,
        development_evidence_sha256=comparison_body["report_sha256"],
        routes=routes,
        seed_evidence_sha256=evidence_digests,
        sealed_ciphertext_sha256=metadata.ciphertext_sha256,
        output_dir=results,
        git_sha="d" * 40,
        git_dirty=False,
    )
    return dataset, results, key_path


def test_certifier_scores_frozen_baseline_routes_once(tmp_path: Path) -> None:
    dataset, results, key_path = _make_inputs(tmp_path)
    report = certify(
        dataset_dir=dataset,
        results_dir=results,
        key_path=key_path,
        output_dir=tmp_path / "certification",
        repository_root=tmp_path / "repository",
    )
    assert report["status"] == "passed"
    assert report["metric_source"] == "sealed_holdout_once"
    assert report["sealed_test_status"] == "OPENED_ONCE"
    assert len(report["routes"]) == 4
