from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from volatility_forecasting.baselines import AdaptiveBaselineHorizon, AdaptiveBaselineSelection
from volatility_forecasting.export import (
    MarketOnlyProductionGraph,
    NewsProductionGraph,
    ProductionVolatilityGraph,
    load_frozen_candidate_member,
    production_graph,
)
from volatility_forecasting.model import (
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
    RobustSequenceScaler,
    TrainingResult,
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

    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_frozen_candidate_member(tmp_path, 41)
