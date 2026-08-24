from __future__ import annotations

import numpy as np
import pytest
import torch
from volatility_forecasting.export import (
    MarketOnlyProductionGraph,
    NewsProductionGraph,
    ProductionVolatilityGraph,
    production_graph,
)
from volatility_forecasting.model import (
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
    RobustSequenceScaler,
    TrainingResult,
)
from volatility_forecasting.refit import FrozenCandidate


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
    return FrozenCandidate(
        training=training,
        architecture=config,
        fit_split=None,
        seed=41,
        epoch_budget=1,
        variance_scale=np.asarray((2.0, 3.0)),
        return_variance_scale=np.asarray((0.5, 0.25)),
        comparison_baseline=None,
        baseline_return_variance_scale=np.ones(2),
        model_identity="fixture",
    )


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
