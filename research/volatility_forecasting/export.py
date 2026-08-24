"""ONNX export and parity checks for a certified volatility ensemble.

Only the compact inference graph is exported. Robust scaling and the matched
adaptive baseline are explicit signed metadata inputs; model-specific variance
calibration is embedded into each member graph so serving cannot accidentally
omit it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .refit import FrozenCandidate


class ProductionVolatilityGraph(nn.Module):
    """Stable tensor-only wrapper used by both PyTorch and ONNX parity tests."""

    def __init__(self, candidate: FrozenCandidate) -> None:
        super().__init__()
        self.model = candidate.training.model.eval()
        self.news_feature_count = candidate.architecture.news_feature_count
        self.register_buffer(
            "variance_scale",
            torch.from_numpy(np.asarray(candidate.variance_scale, dtype=np.float32)),
        )
        self.register_buffer(
            "return_variance_scale",
            torch.from_numpy(np.asarray(candidate.return_variance_scale, dtype=np.float32)),
        )

    def forward(
        self,
        features: torch.Tensor,
        baseline_variance: torch.Tensor,
        news_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        variance, location, logits, _residual = self.model(
            features,
            baseline_variance,
            news_features,
        )
        calibrated_variance = variance * self.variance_scale
        return_variance = calibrated_variance * self.return_variance_scale
        probabilities = torch.softmax(logits, dim=-1)
        return calibrated_variance, location, probabilities, return_variance


class MarketOnlyProductionGraph(nn.Module):
    """Two-input ONNX signature without a misleading empty news tensor."""

    def __init__(self, graph: ProductionVolatilityGraph) -> None:
        super().__init__()
        if graph.news_feature_count:
            raise ValueError("market-only wrapper cannot export a news-enabled candidate")
        self.graph = graph

    def forward(
        self,
        features: torch.Tensor,
        baseline_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.graph(features, baseline_variance)


class NewsProductionGraph(nn.Module):
    """Three-input ONNX signature for a news-ablation-certified candidate."""

    def __init__(self, graph: ProductionVolatilityGraph) -> None:
        super().__init__()
        if not graph.news_feature_count:
            raise ValueError("news wrapper requires a news-enabled candidate")
        self.graph = graph

    def forward(
        self,
        features: torch.Tensor,
        baseline_variance: torch.Tensor,
        news_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.graph(features, baseline_variance, news_features)


def production_graph(candidate: FrozenCandidate) -> nn.Module:
    graph = ProductionVolatilityGraph(candidate).eval()
    return (
        NewsProductionGraph(graph).eval()
        if graph.news_feature_count
        else MarketOnlyProductionGraph(graph).eval()
    )


def export_candidate_onnx(
    candidate: FrozenCandidate,
    output_path: Path,
    *,
    opset_version: int = 18,
) -> Path:
    """Export one immutable seed member with a dynamic batch dimension."""
    if opset_version < 17:
        raise ValueError("production export requires ONNX opset 17 or newer")
    path = output_path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ONNX member: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    config = candidate.architecture
    graph = production_graph(candidate)
    features = torch.zeros(1, config.window_size, config.feature_count, dtype=torch.float32)
    baseline = torch.ones(1, config.horizon_count, dtype=torch.float32)
    inputs: tuple[torch.Tensor, ...]
    input_names = ["features", "baseline_variance"]
    dynamic_axes: dict[str, dict[int, str]] = {
        "features": {0: "batch"},
        "baseline_variance": {0: "batch"},
    }
    if config.news_feature_count:
        news = torch.zeros(1, config.news_feature_count, dtype=torch.float32)
        inputs = (features, baseline, news)
        input_names.append("news_features")
        dynamic_axes["news_features"] = {0: "batch"}
    else:
        inputs = (features, baseline)
    output_names = [
        "forecast_variance",
        "return_location",
        "direction_probabilities",
        "return_variance",
    ]
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    with torch.no_grad():
        torch.onnx.export(
            graph,
            inputs,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("ONNX export did not create a non-empty artifact")
    return path


def verify_onnx_parity(
    candidate: FrozenCandidate,
    onnx_path: Path,
    *,
    rows: int = 7,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> dict[str, float]:
    """Compare all production outputs using deterministic finite inputs."""
    if rows < 2 or absolute_tolerance <= 0 or relative_tolerance <= 0:
        raise ValueError("parity settings are invalid")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for production parity verification") from error
    config = candidate.architecture
    rng = np.random.default_rng(20260824 + candidate.seed)
    features = rng.normal(size=(rows, config.window_size, config.feature_count)).astype(np.float32)
    baseline = rng.lognormal(mean=-7.0, sigma=0.4, size=(rows, config.horizon_count)).astype(
        np.float32
    )
    feed = {"features": features, "baseline_variance": baseline}
    tensors: tuple[torch.Tensor, ...]
    if config.news_feature_count:
        news = rng.normal(size=(rows, config.news_feature_count)).astype(np.float32)
        feed["news_features"] = news
        tensors = (torch.from_numpy(features), torch.from_numpy(baseline), torch.from_numpy(news))
    else:
        tensors = (torch.from_numpy(features), torch.from_numpy(baseline))
    graph = production_graph(candidate)
    with torch.no_grad():
        expected = [value.cpu().numpy() for value in graph(*tensors)]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, feed)
    names = ("forecast_variance", "return_location", "direction_probabilities", "return_variance")
    maximum_errors: dict[str, float] = {}
    for name, expected_values, actual_values in zip(names, expected, actual, strict=True):
        if expected_values.shape != actual_values.shape or not np.isfinite(actual_values).all():
            raise RuntimeError(f"ONNX parity output is invalid: {name}")
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            err_msg=f"ONNX parity failed for {name}",
        )
        maximum_errors[name] = float(np.max(np.abs(actual_values - expected_values)))
    return maximum_errors
