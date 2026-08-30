"""V11.2 release-adapter contracts and signed bundle assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from volatility_forecasting.export_v11_2 import V112ProductionGraph, assemble_v11_2_release
from volatility_forecasting.model import RobustSequenceScaler
from volatility_forecasting.v11_2_freezer import V112Route, freeze_routing_bundle
from volatility_forecasting.v11_2_model import (
    V11_2_RESIDUAL_ARCHITECTURE_VERSION,
    build_v11_2_residual_model,
)
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


def _release_fixture(
    tmp_path: Path,
    *,
    route_family_overrides: dict[int, str] | None = None,
) -> tuple[Path, Path]:
    protocol = V112Protocol()
    family_overrides = route_family_overrides or {}
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
        family = family_overrides.get(horizon, "M0_HAR_BASELINE")
        learned = family not in {
            "ZERO_RETURN_CONST_VAR",
            "ZERO_RETURN_PERSISTENCE_VOL",
            "M0_HAR_BASELINE",
        }
        if family == "M1_NUMERIC_RESIDUAL":
            artifact = results / "models" / f"horizon_{horizon}" / "seed_42.pt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            model = build_v11_2_residual_model(
                feature_count=len(protocol.feature_names),
                window_size=protocol.window_size,
            )
            with torch.no_grad():
                model.log_variance_residual_head.bias.fill_(0.25)
                model.return_location_head.bias.fill_(0.10)
            torch.save(model.state_dict(), artifact)
        elif learned:
            # The unsupported-family test only needs an immutable artifact;
            # the release adapter must reject the route before deserialization.
            artifact = results / "models" / f"horizon_{horizon}_research.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"research-only-route")
        else:
            artifact = results / "baselines" / f"har_horizon_{horizon}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps({"family": family, "horizon": horizon}, sort_keys=True),
                encoding="utf-8",
            )
        selection = results / f"selection_horizon_{horizon}.json"
        selection.write_text(
            json.dumps(
                {
                    "horizon": horizon,
                    "selected_family": family,
                    "learned_promotion": learned,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        for seed in protocol.seeds:
            evidence_body = {"horizon": horizon, "seed": seed, "family": family}
            evidence_body["evidence_sha256"] = canonical_json_digest(
                {key: value for key, value in evidence_body.items()}
            )
            evidence_digests.append(str(evidence_body["evidence_sha256"]))
        routes.append(
            V112Route(
                horizon=horizon,
                family=family,
                model_digest=_sha256(artifact),
                scaler_digest=_sha256(scaler_path),
                selection_record_digest=_sha256(selection),
                learned_promotion=learned,
                artifact_path=artifact.relative_to(results).as_posix(),
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
            "family": family_overrides.get(horizon, "M0_HAR_BASELINE"),
            "learned_promotion": family_overrides.get(horizon, "M0_HAR_BASELINE")
            not in {
                "ZERO_RETURN_CONST_VAR",
                "ZERO_RETURN_PERSISTENCE_VOL",
                "M0_HAR_BASELINE",
            },
            "metrics": {"crps_mean": 0.1},
            "gate": {"passed": True, "decision": "passed"},
        }
        for horizon in V11_2_HORIZONS
    ]
    report_body = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.digest(),
        "feature_schema_sha256": feature_schema_digest(protocol),
        "candidate_digest": routing_digest,
        "metric_source": "sealed_holdout_once",
        "sealed_test_status": "OPENED_ONCE",
        "status": "passed",
        "m0_adequacy_passed": True,
        "m0_adequacy": {
            "har_vs_constant": [
                {
                    "horizon": horizon,
                    "candidate": "M0_HAR_BASELINE",
                    "comparator": "ZERO_RETURN_CONST_VAR",
                    "passed": True,
                }
                for horizon in V11_2_HORIZONS
            ],
            "har_vs_persistence": [
                {
                    "horizon": horizon,
                    "candidate": "M0_HAR_BASELINE",
                    "comparator": "ZERO_RETURN_PERSISTENCE_VOL",
                    "passed": True,
                }
                for horizon in V11_2_HORIZONS
            ],
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

    results, certification = _release_fixture(tmp_path)
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
    assert metadata["architecture_version"] == V11_2_RESIDUAL_ARCHITECTURE_VERSION
    assert metadata["architecture"]["feature_count"] == 26
    assert metadata["architecture"]["window_size"] == 60
    assert metadata["certified_heads"] == {
        "volatility": True,
        "return_distribution": False,
        "direction": False,
    }
    assert metadata["return_distribution_horizons"] == []
    assert metadata["return_distribution"] == {
        "family": "zero_location_normal",
        "degrees_of_freedom": None,
        "location_output": None,
        "variance_output": None,
    }
    assert metadata["certified_horizons"] == [1, 3, 5, 7]
    assert metadata["uncertified_horizons"] == [14, 30]


def test_assemble_v112_m1_release_round_trips_through_production_runtime(
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnxruntime")
    from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5
    from backend.services.volatility_runtime import VolatilityOnnxRuntime

    results, certification = _release_fixture(
        tmp_path,
        route_family_overrides={3: "M1_NUMERIC_RESIDUAL"},
    )
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
    assert summary["learned_horizons"] == [3]
    assert summary["baseline_horizons"] == [1, 5, 7]

    runtime = VolatilityOnnxRuntime.from_release_bundle(
        output,
        public_key_path=public_path,
    )
    baseline = np.full(6, 0.04, dtype=np.float32)
    snapshot = SimpleNamespace(
        feature_names=DEPLOYABLE_FEATURE_COLUMNS_V5,
        features=np.zeros((60, 26), dtype=np.float32),
        causal_har_variance=baseline,
    )
    forecast = runtime.forecast(snapshot)
    assert runtime.certified_horizon_list() == (1, 3, 5, 7)
    assert runtime.return_distribution_horizon_list() == (3,)
    assert runtime.is_return_distribution_horizon(1) is False
    assert runtime.is_return_distribution_horizon(3) is True
    assert forecast.forecast_variance.shape == (6,)
    assert forecast.direction_probabilities.shape == (6, 3)
    assert forecast.forecast_variance[1] > baseline[1]
    np.testing.assert_allclose(
        forecast.forecast_variance[[0, 2, 3, 4, 5]],
        baseline[[0, 2, 3, 4, 5]],
        rtol=1e-6,
        atol=1e-8,
    )


def test_assemble_v112_rejects_unexportable_research_route(tmp_path: Path) -> None:
    results, certification = _release_fixture(
        tmp_path,
        route_family_overrides={1: "RIDGE_LOCATION_HAR_SCALE"},
    )
    with pytest.raises(ValueError, match="not exportable"):
        assemble_v11_2_release(
            results_dir=results,
            certification_dir=certification,
            output_dir=tmp_path / "release",
            private_key_path=tmp_path / "unused-private-key.pem",
        )
    assert not (tmp_path / "release").exists()


def test_assemble_v112_rejects_changed_selection_record(tmp_path: Path) -> None:
    results, certification = _release_fixture(tmp_path)
    (results / "selection_horizon_1.json").write_text(
        '{"horizon": 1, "selected_family": "M1_NUMERIC_RESIDUAL"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selection record changed"):
        assemble_v11_2_release(
            results_dir=results,
            certification_dir=certification,
            output_dir=tmp_path / "release",
            private_key_path=tmp_path / "unused-private-key.pem",
        )
    assert not (tmp_path / "release").exists()


def test_assemble_v112_rejects_incomplete_m0_adequacy_family(tmp_path: Path) -> None:
    results, certification = _release_fixture(tmp_path)
    report_path = certification / "v11_2_holdout_certification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["m0_adequacy"]["har_vs_constant"] = report["m0_adequacy"]["har_vs_constant"][:1]
    report_body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_json_digest(report_body)
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="M0 adequacy evidence is incomplete"):
        assemble_v11_2_release(
            results_dir=results,
            certification_dir=certification,
            output_dir=tmp_path / "release",
            private_key_path=tmp_path / "unused-private-key.pem",
        )
    assert not (tmp_path / "release").exists()
