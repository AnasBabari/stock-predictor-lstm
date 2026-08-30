"""V11.2 release-adapter contracts and signed bundle assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from volatility_forecasting.export_v11_2 import V112ProductionGraph, assemble_v11_2_release
from volatility_forecasting.model import RobustSequenceScaler
from volatility_forecasting.v11_2_freezer import V112Route, freeze_routing_bundle
from volatility_forecasting.v11_2_protocol import (
    V11_2_HORIZONS,
    V112Protocol,
    canonical_json_digest,
    feature_schema_digest,
)

from backend.services.volatility_runtime.contracts import (
    RUNTIME_SCHEMA_VERSION,
    VolatilityRuntimeContract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v112_graph_supports_all_baseline_routes() -> None:
    graph = V112ProductionGraph(
        models={},
        scaler=RobustSequenceScaler(np.zeros(26), np.ones(26)),
    ).eval()
    outputs = graph(torch.zeros(2, 60, 26), torch.ones(2, 6))
    variance, location, probabilities, return_variance = outputs
    assert tuple(variance.shape) == (2, 6)
    assert tuple(location.shape) == (2, 6)
    assert tuple(probabilities.shape) == (2, 6, 3)
    assert tuple(return_variance.shape) == (2, 6)
    torch.testing.assert_close(variance, torch.ones(2, 6))
    torch.testing.assert_close(location, torch.zeros(2, 6))
    torch.testing.assert_close(probabilities, torch.full((2, 6, 3), 1.0 / 3.0))
    torch.testing.assert_close(return_variance, variance)


def test_runtime_contract_accepts_sealed_holdout_source_and_subset() -> None:
    metadata = {
        "runtime_schema": RUNTIME_SCHEMA_VERSION,
        "model_id": "v11-2-fixture",
        "model_version": "v11.2-numeric-residual-v1",
        "protocol_version": "v11.2-numeric-pit64",
        "feature_names": [],
        "window_size": 60,
        "horizons": [1, 3, 5, 7, 14, 30],
        "news_feature_count": 0,
        "news_feature_names": [],
        "members": [{"seed": 42, "file": "members/model.onnx"}],
        "certified_horizons": [1, 3, 5, 7],
        "metric_source": "sealed_holdout_once",
        "certification_scope": "sealed_holdout_once",
        "news_status": "not_certified",
    }
    # Use the canonical deployable order instead of relying on a dataclass
    # default (which is intentionally absent for this field).
    from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5

    metadata["feature_names"] = list(DEPLOYABLE_FEATURE_COLUMNS_V5)
    contract = VolatilityRuntimeContract.from_release_metadata(
        metadata,
        {"members/model.onnx"},
    )
    assert contract.metric_source == "sealed_holdout_once"
    assert contract.certification_scope == "sealed_holdout_once"
    assert contract.certified_horizon_list() == (1, 3, 5, 7)


def _signing_keys(tmp_path: Path) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def _baseline_fixture(tmp_path: Path) -> tuple[Path, Path]:
    protocol = V112Protocol()
    results = tmp_path / "results"
    results.mkdir()
    scaler_path = results / "numeric_scaler.json"
    scaler_path.write_text(
        json.dumps(RobustSequenceScaler(np.zeros(26), np.ones(26)).to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    routes: list[V112Route] = []
    evidence_digests: list[str] = []
    for horizon in V11_2_HORIZONS:
        artifact = results / "baselines" / f"har_horizon_{horizon}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps({"family": "M0_HAR_BASELINE", "horizon": horizon}, sort_keys=True),
            encoding="utf-8",
        )
        selection = results / f"selection_horizon_{horizon}.json"
        selection.write_text(
            json.dumps({"horizon": horizon, "selected_family": "M0_HAR_BASELINE"}, sort_keys=True),
            encoding="utf-8",
        )
        for seed in protocol.seeds:
            evidence_body = {"horizon": horizon, "seed": seed, "family": "M0_HAR_BASELINE"}
            evidence_body["evidence_sha256"] = canonical_json_digest(
                {key: value for key, value in evidence_body.items()}
            )
            evidence_digests.append(str(evidence_body["evidence_sha256"]))
        routes.append(
            V112Route(
                horizon=horizon,
                family="M0_HAR_BASELINE",
                model_digest=_sha256(artifact),
                scaler_digest=_sha256(scaler_path),
                selection_record_digest=_sha256(selection),
                learned_promotion=False,
                artifact_path=f"baselines/{artifact.name}",
            )
        )
    freeze_routing_bundle(
        protocol=protocol,
        universe_sha256="a" * 64,
        panel_sha256="b" * 64,
        schema_sha256=feature_schema_digest(protocol),
        split_sha256="c" * 64,
        development_evidence_sha256="d" * 64,
        routes=routes,
        seed_evidence_sha256=evidence_digests,
        sealed_ciphertext_sha256="e" * 64,
        output_dir=results,
        git_sha="f" * 40,
        git_dirty=False,
    )
    routing_digest = (results / "v11_2_routing_bundle.sha256").read_text(encoding="ascii").strip()
    report_routes = [
        {
            "horizon": horizon,
            "family": "M0_HAR_BASELINE",
            "learned_promotion": False,
            "metrics": {"crps_mean": 0.1},
            "gate": {"passed": True, "decision": "frozen_baseline_route"},
        }
        for horizon in V11_2_HORIZONS
    ]
    report_body = {
        "protocol_id": protocol.protocol_id,
        "candidate_digest": routing_digest,
        "metric_source": "sealed_holdout_once",
        "sealed_test_status": "OPENED_ONCE",
        "status": "passed",
        "m0_adequacy_passed": True,
        "m0_adequacy": {
            "constant": [{"passed": True}],
            "persistence": [{"passed": True}],
        },
        "routes": report_routes,
        "test_stock_origin_observations": 100,
        "test_unique_sessions": 50,
        "test_sessions": ["2025-01-01", "2025-03-11"],
    }
    report = {**report_body, "report_sha256": canonical_json_digest(report_body)}
    certification = tmp_path / "certification"
    certification.mkdir()
    (certification / "v11_2_holdout_certification.json").write_text(
        json.dumps(report, sort_keys=True),
        encoding="utf-8",
    )
    return results, certification


def test_assemble_v112_baseline_release_is_signed_and_withholds_14_30(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    from release.bundle import verify_release

    results, certification = _baseline_fixture(tmp_path)
    private_path, public_path = _signing_keys(tmp_path)
    output = tmp_path / "release"
    summary = assemble_v11_2_release(
        results_dir=results,
        certification_dir=certification,
        output_dir=output,
        private_key_path=private_path,
        public_key_path=public_path,
        parity_rows=2,
    )
    assert summary["certified_horizons"] == [1, 3, 5, 7]
    assert summary["learned_horizons"] == []
    assert summary["baseline_horizons"] == [1, 3, 5, 7]
    manifest = verify_release(output, public_key_path=public_path)
    metadata = manifest["metadata"]
    assert metadata["metric_source"] == "sealed_holdout_once"
    assert metadata["certified_horizons"] == [1, 3, 5, 7]
    assert metadata["uncertified_horizons"] == [14, 30]
