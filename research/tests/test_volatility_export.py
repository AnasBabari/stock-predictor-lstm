from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from volatility_forecasting.baselines import AdaptiveBaselineHorizon, AdaptiveBaselineSelection
from volatility_forecasting.export import (
    MarketOnlyProductionGraph,
    NewsProductionGraph,
    ProductionVolatilityGraph,
    load_frozen_candidate_member,
    load_locked_v8_candidate_member,
    load_prospective_candidate_member,
    load_prospective_v8_candidate_member,
    production_graph,
)
from volatility_forecasting.model import (
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
    RobustSequenceScaler,
    TrainingResult,
    VolatilityLossWeights,
)
from volatility_forecasting.refit import FrozenCandidate, candidate_identity


def _candidate(*, news_features: int = 0) -> FrozenCandidate:
    config = BaselineResidualTCNConfig(
        feature_count=4,
        horizon_count=2,
        window_size=60,
        channels=8,
        dilations=(1, 2),
        news_feature_count=news_features,
        news_channels=4,
    )
    model = BaselineResidualTCN(config).eval()
    scaler = RobustSequenceScaler(median=np.zeros(4), iqr=np.ones(4))
    news_scaler = (
        RobustSequenceScaler(median=np.zeros(news_features), iqr=np.ones(news_features))
        if news_features
        else None
    )
    training = TrainingResult(
        model=model,
        scaler=scaler,
        news_scaler=news_scaler,
        best_epoch=1,
        history=(),
        device="cpu",
        duration_seconds=0.0,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    comparison = AdaptiveBaselineSelection(
        horizons=tuple(
            AdaptiveBaselineHorizon(
                horizon=horizon,
                family="causal_log_har",
                blend_alpha=0.0,
                multiplicative_scale=1.0,
                calibration_qlike=0.1,
                har_calibration_qlike=0.1,
            )
            for horizon in (1, 7)
        )
    )
    candidate = FrozenCandidate(
        training=training,
        architecture=config,
        fit_split=None,
        seed=41,
        epoch_budget=1,
        variance_scale=np.asarray((2.0, 3.0)),
        return_variance_scale=np.asarray((0.5, 0.25)),
        comparison_baseline=comparison,
        baseline_return_variance_scale=np.ones(2),
        model_identity="fixture",
    )
    identity = candidate_identity(
        candidate.training,
        architecture=candidate.architecture,
        seed=candidate.seed,
        epoch_budget=candidate.epoch_budget,
        variance_scale=candidate.variance_scale,
        return_variance_scale=candidate.return_variance_scale,
        comparison_baseline=candidate.comparison_baseline,
        baseline_return_variance_scale=candidate.baseline_return_variance_scale,
    )
    return FrozenCandidate(**{**candidate.__dict__, "model_identity": identity})


def test_production_graph_embeds_calibration_and_normalized_probabilities() -> None:
    candidate = _candidate()
    graph = production_graph(candidate)
    assert isinstance(graph, MarketOnlyProductionGraph)
    features = torch.zeros(3, 60, 4)
    baseline = torch.ones(3, 2)
    variance, location, probabilities, return_variance = graph(features, baseline)
    assert tuple(variance.shape) == (3, 2)
    assert torch.allclose(variance, torch.tensor([[2.0, 3.0]]).repeat(3, 1))
    assert torch.allclose(return_variance, torch.tensor([[1.0, 0.75]]).repeat(3, 1))
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3, 2))
    assert tuple(location.shape) == (3, 2)


def test_production_graph_embeds_train_only_feature_scaling() -> None:
    candidate = _candidate()
    scaler = RobustSequenceScaler(
        median=np.full(4, 2.0),
        iqr=np.full(4, 4.0),
        clip=3.0,
    )
    training = TrainingResult(
        **{**candidate.training.__dict__, "scaler": scaler}
    )
    candidate = FrozenCandidate(**{**candidate.__dict__, "training": training})
    graph = production_graph(candidate)
    raw_features = torch.full((2, 60, 4), 2.0)
    baseline = torch.ones(2, 2)
    with torch.no_grad():
        actual = graph(raw_features, baseline)
        variance, location, logits, _residual = candidate.training.model(
            torch.zeros_like(raw_features), baseline
        )
    expected = (
        variance * torch.tensor(candidate.variance_scale, dtype=torch.float32),
        location,
        torch.softmax(logits, dim=-1),
        variance
        * torch.tensor(candidate.variance_scale, dtype=torch.float32)
        * torch.tensor(candidate.return_variance_scale, dtype=torch.float32),
    )
    for actual_values, expected_values in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_values, expected_values)


def test_export_signature_is_explicit_about_news_input() -> None:
    market = ProductionVolatilityGraph(_candidate())
    with pytest.raises(ValueError, match="news-enabled"):
        NewsProductionGraph(market)
    news_candidate = _candidate(news_features=3)
    news_graph = production_graph(news_candidate)
    assert isinstance(news_graph, NewsProductionGraph)
    outputs = news_graph(torch.zeros(2, 60, 4), torch.ones(2, 2), torch.zeros(2, 3))
    assert all(tuple(output.shape[:2]) == (2, 2) for output in outputs)
    with pytest.raises(ValueError, match="market-only"):
        MarketOnlyProductionGraph(ProductionVolatilityGraph(news_candidate))


def test_candidate_loader_verifies_weights_metadata_and_content_identity(tmp_path) -> None:
    candidate = _candidate()
    weights = tmp_path / "seed-41.pt"
    torch.save(candidate.training.model.state_dict(), weights)
    member = {
        "seed": 41,
        "model_identity": candidate.model_identity,
        "weights_file": weights.name,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "epoch_budget": 1,
        "best_epoch": 1,
        "market_scaler": candidate.training.scaler.to_dict(),
        "news_scaler": None,
        "variance_scale": candidate.variance_scale.tolist(),
        "return_variance_scale": candidate.return_variance_scale.tolist(),
        "baseline_return_variance_scale": candidate.baseline_return_variance_scale.tolist(),
        "comparison_baseline": [asdict(value) for value in candidate.comparison_baseline.horizons],
    }
    manifest = {
        "artifact_role": "locked_certification_candidate",
        "model_identity": "ensemble-fixture",
        "protocol": {"horizons": [1, 7]},
        "architecture": asdict(candidate.architecture),
        "members": [member],
    }
    (tmp_path / "candidate-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_frozen_candidate_member(tmp_path, 41)
    assert loaded.model_identity == candidate.model_identity
    assert loaded.training.scaler.to_dict() == candidate.training.scaler.to_dict()
    with pytest.raises(ValueError, match="role"):
        load_prospective_candidate_member(tmp_path, 41)

    manifest["artifact_role"] = "prospective_development_candidate"
    (tmp_path / "candidate-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    prospective = load_prospective_candidate_member(tmp_path, 41)
    assert prospective.model_identity == candidate.model_identity
    with pytest.raises(ValueError, match="role"):
        load_frozen_candidate_member(tmp_path, 41)

    manifest["artifact_role"] = "prospective_v8_development_candidate"
    (tmp_path / "candidate-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_prospective_v8_candidate_member(tmp_path, 41).model_identity == candidate.model_identity
    with pytest.raises(ValueError, match="role"):
        load_locked_v8_candidate_member(tmp_path, 41)

    manifest["artifact_role"] = "locked_v8_certification_candidate"
    (tmp_path / "candidate-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_locked_v8_candidate_member(tmp_path, 41).model_identity == candidate.model_identity

    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_locked_v8_candidate_member(tmp_path, 41)


def test_candidate_identity_binds_explicit_loss_weights() -> None:
    candidate = _candidate()
    arguments = {
        "architecture": candidate.architecture,
        "seed": candidate.seed,
        "epoch_budget": candidate.epoch_budget,
        "variance_scale": candidate.variance_scale,
        "return_variance_scale": candidate.return_variance_scale,
        "comparison_baseline": candidate.comparison_baseline,
        "baseline_return_variance_scale": candidate.baseline_return_variance_scale,
    }
    legacy_identity = candidate_identity(candidate.training, **arguments)
    explicit_identity = candidate_identity(
        candidate.training,
        **arguments,
        loss_weights=VolatilityLossWeights(),
    )
    challenger_identity = candidate_identity(
        candidate.training,
        **arguments,
        loss_weights=VolatilityLossWeights(
            qlike=0.70,
            variance_crps=0.25,
            return_location=0.0,
            direction=0.0,
            baseline_regularization=0.05,
        ),
    )
    assert explicit_identity != legacy_identity
    assert challenger_identity != explicit_identity


# --- Release assembly -------------------------------------------------------


def _certified_member(seed: int = 41) -> FrozenCandidate:
    """A fixture at the full certified deployable_v5 schema (26 features, 6 horizons)."""
    from volatility_forecasting.contracts import DEPLOYABLE_FEATURE_COLUMNS_V5

    horizons = (1, 3, 5, 7, 14, 30)
    config = BaselineResidualTCNConfig(
        feature_count=len(DEPLOYABLE_FEATURE_COLUMNS_V5),
        horizon_count=len(horizons),
        window_size=60,
        channels=8,
        dilations=(1, 2),
    )
    model = BaselineResidualTCN(config).eval()
    scaler = RobustSequenceScaler(
        median=np.zeros(len(DEPLOYABLE_FEATURE_COLUMNS_V5)),
        iqr=np.ones(len(DEPLOYABLE_FEATURE_COLUMNS_V5)),
    )
    training = TrainingResult(
        model=model,
        scaler=scaler,
        news_scaler=None,
        best_epoch=1,
        history=(),
        device="cpu",
        duration_seconds=0.0,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    comparison = AdaptiveBaselineSelection(
        horizons=tuple(
            AdaptiveBaselineHorizon(
                horizon=horizon,
                family="causal_log_har",
                blend_alpha=0.0,
                multiplicative_scale=1.0,
                calibration_qlike=0.1,
                har_calibration_qlike=0.1,
            )
            for horizon in horizons
        )
    )
    ones = np.ones(len(horizons))
    candidate = FrozenCandidate(
        training=training,
        architecture=config,
        fit_split=None,
        seed=seed,
        epoch_budget=1,
        variance_scale=2.0 * ones,
        return_variance_scale=0.5 * ones,
        comparison_baseline=comparison,
        baseline_return_variance_scale=ones.copy(),
        model_identity="fixture",
    )
    identity = candidate_identity(
        candidate.training,
        architecture=candidate.architecture,
        seed=candidate.seed,
        epoch_budget=candidate.epoch_budget,
        variance_scale=candidate.variance_scale,
        return_variance_scale=candidate.return_variance_scale,
        comparison_baseline=candidate.comparison_baseline,
        baseline_return_variance_scale=candidate.baseline_return_variance_scale,
    )
    return FrozenCandidate(**{**candidate.__dict__, "model_identity": identity})


def _write_candidate_directory(directory, seeds: tuple[int, ...]) -> str:
    """Persist verified seed members exactly like the certification CLI does."""
    members = [_certified_member(seed) for seed in seeds]
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for member in members:
        filename = f"seed-{member.seed}.pt"
        path = directory / filename
        torch.save(member.training.model.state_dict(), path)
        rows.append(
            {
                "seed": member.seed,
                "model_identity": member.model_identity,
                "weights_file": filename,
                "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "epoch_budget": member.epoch_budget,
                "best_epoch": member.training.best_epoch,
                "market_scaler": member.training.scaler.to_dict(),
                "news_scaler": None,
                "variance_scale": member.variance_scale.tolist(),
                "return_variance_scale": member.return_variance_scale.tolist(),
                "baseline_return_variance_scale": member.baseline_return_variance_scale.tolist(),
                "comparison_baseline": [
                    asdict(value) for value in member.comparison_baseline.horizons
                ],
            }
        )
    manifest = {
        "artifact_role": "locked_certification_candidate",
        "model_identity": f"global-volatility-ensemble:{len(members)}",
        "protocol": {"horizons": [1, 3, 5, 7, 14, 30]},
        "architecture": asdict(members[0].architecture),
        "members": rows,
    }
    (directory / "candidate-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(manifest["model_identity"])


def _signing_keys(tmp_path) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private_pem = tmp_path / "signing.pem"
    private_pem.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_pem = tmp_path / "verify.pem"
    public_pem.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_pem, public_pem


def test_assemble_release_bundle_signs_verified_runtime_metadata(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    from volatility_forecasting.export import assemble_release_bundle

    directory = tmp_path / "candidate"
    expected_model_id = _write_candidate_directory(directory, seeds=(41, 42))
    private_pem, public_pem = _signing_keys(tmp_path)

    summary = assemble_release_bundle(
        directory,
        tmp_path / "release",
        private_key_path=private_pem,
        public_key_path=public_pem,
    )
    assert summary["model_id"] == expected_model_id
    assert summary["member_seeds"] == [41, 42]
    assert summary["max_parity_error"] < 1e-4

    from backend.release.bundle import verify_release

    manifest = verify_release(tmp_path / "release", public_key_path=public_pem)
    metadata = manifest["metadata"]
    assert metadata["runtime_schema"] == "volatility-runtime-v1"
    assert metadata["model_id"] == expected_model_id
    assert metadata["window_size"] == 60
    assert metadata["horizons"] == [1, 3, 5, 7, 14, 30]
    assert metadata["news_feature_count"] == 0
    assert len(metadata["feature_names"]) == 26
    assert [row["seed"] for row in metadata["members"]] == [41, 42]
    assert set(manifest["files"]) == {"members/seed-41.onnx", "members/seed-42.onnx"}


def test_assemble_refuses_candidates_off_the_certified_schema(tmp_path) -> None:
    directory = tmp_path / "candidate"
    candidate = _candidate()  # four-feature research fixture, not deployable_v5
    weights = directory / "seed-41.pt"
    directory.mkdir(parents=True)
    torch.save(candidate.training.model.state_dict(), weights)
    manifest = {
        "artifact_role": "locked_certification_candidate",
        "model_identity": "ensemble-fixture",
        "protocol": {"horizons": [1, 7]},
        "architecture": asdict(candidate.architecture),
        "members": [
            {
                "seed": 41,
                "model_identity": candidate.model_identity,
                "weights_file": weights.name,
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            }
        ],
    }
    (directory / "candidate-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    from volatility_forecasting.export import assemble_release_bundle

    with pytest.raises(ValueError, match="deployable_v5"):
        assemble_release_bundle(
            directory, tmp_path / "release", private_key_path=tmp_path / "k.pem"
        )


def test_assemble_refuses_non_locked_roles(tmp_path) -> None:
    directory = tmp_path / "candidate"
    directory.mkdir(parents=True)
    (directory / "candidate-manifest.json").write_text(
        json.dumps({"artifact_role": "development_snapshot"}),
        encoding="utf-8",
    )
    from volatility_forecasting.export import assemble_release_bundle

    with pytest.raises(ValueError, match="locked-certification"):
        assemble_release_bundle(
            directory, tmp_path / "release", private_key_path=tmp_path / "k.pem"
        )
