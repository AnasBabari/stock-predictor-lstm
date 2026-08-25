"""ONNX export and parity checks for a certified volatility ensemble.

Only the compact inference graph is exported. Robust scaling and the matched
adaptive baseline are explicit signed metadata inputs; model-specific variance
calibration is embedded into each member graph so serving cannot accidentally
omit it.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .baselines import AdaptiveBaselineHorizon, AdaptiveBaselineSelection
from .contracts import DEPLOYABLE_FEATURE_COLUMNS_V5
from .model import (
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
    RobustSequenceScaler,
    TrainingResult,
)
from .refit import FrozenCandidate, candidate_identity


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


def load_frozen_candidate_member(candidate_dir: Path, seed: int) -> FrozenCandidate:
    """Reconstruct and verify one local candidate member for release conversion."""
    directory = candidate_dir.resolve()
    try:
        manifest = json.loads((directory / "candidate-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("candidate manifest is missing or invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("artifact_role") != "locked_certification_candidate"
    ):
        raise ValueError("candidate manifest role is incompatible")
    architecture_payload = manifest.get("architecture")
    protocol_payload = manifest.get("protocol")
    members = manifest.get("members")
    if (
        not isinstance(architecture_payload, dict)
        or not isinstance(protocol_payload, dict)
        or not isinstance(members, list)
    ):
        raise ValueError("candidate manifest is incomplete")
    rows = [row for row in members if isinstance(row, dict) and row.get("seed") == seed]
    if len(rows) != 1:
        raise ValueError(f"candidate manifest does not contain exactly one seed {seed}")
    row = rows[0]
    filename = row.get("weights_file")
    expected_digest = row.get("weights_sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".pt")
    ):
        raise ValueError("candidate weights path is not allowed")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("candidate weights checksum is invalid")
    weights_path = directory / filename
    try:
        weights_bytes = weights_path.read_bytes()
    except OSError as error:
        raise ValueError("candidate weights are missing") from error
    if hashlib.sha256(weights_bytes).hexdigest() != expected_digest:
        raise ValueError("candidate weights checksum does not match")
    architecture = BaselineResidualTCNConfig(**architecture_payload)
    model = BaselineResidualTCN(architecture)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as error:
        raise ValueError("candidate weights are incompatible") from error
    market_scaler = row.get("market_scaler")
    news_scaler_payload = row.get("news_scaler")
    comparison_rows = row.get("comparison_baseline")
    if not isinstance(market_scaler, dict) or not isinstance(comparison_rows, list):
        raise ValueError("candidate preprocessing metadata is incomplete")
    training = TrainingResult(
        model=model.eval(),
        scaler=RobustSequenceScaler.from_dict(market_scaler),
        news_scaler=(
            RobustSequenceScaler.from_dict(news_scaler_payload)
            if isinstance(news_scaler_payload, dict)
            else None
        ),
        best_epoch=int(row.get("best_epoch", 0)),
        history=(),
        device="cpu",
        duration_seconds=0.0,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    comparison = AdaptiveBaselineSelection(
        horizons=tuple(AdaptiveBaselineHorizon(**value) for value in comparison_rows)
    )
    candidate = FrozenCandidate(
        training=training,
        architecture=architecture,
        fit_split=None,  # type: ignore[arg-type] -- release conversion does not retrain
        seed=seed,
        epoch_budget=int(row.get("epoch_budget", 0)),
        variance_scale=np.asarray(row.get("variance_scale"), dtype=np.float64),
        return_variance_scale=np.asarray(row.get("return_variance_scale"), dtype=np.float64),
        comparison_baseline=comparison,
        baseline_return_variance_scale=np.asarray(
            row.get("baseline_return_variance_scale"),
            dtype=np.float64,
        ),
        model_identity=str(row.get("model_identity", "")),
    )
    horizon_shape = (architecture.horizon_count,)
    if candidate.epoch_budget < 1 or candidate.training.best_epoch < 1:
        raise ValueError("candidate epoch metadata is invalid")
    if len(candidate.training.scaler.median) != architecture.feature_count:
        raise ValueError("candidate market scaler dimension is incompatible")
    if architecture.news_feature_count:
        if (
            candidate.training.news_scaler is None
            or len(candidate.training.news_scaler.median) != architecture.news_feature_count
        ):
            raise ValueError("candidate news scaler dimension is incompatible")
    elif candidate.training.news_scaler is not None:
        raise ValueError("market-only candidate unexpectedly contains a news scaler")
    scales = (
        candidate.variance_scale,
        candidate.return_variance_scale,
        candidate.baseline_return_variance_scale,
    )
    if any(
        values.shape != horizon_shape or not np.isfinite(values).all() or (values <= 0).any()
        for values in scales
    ):
        raise ValueError("candidate calibration scales are incompatible")
    if tuple(item.horizon for item in comparison.horizons) != tuple(
        int(value) for value in protocol_payload.get("horizons", [])
    ):
        raise ValueError("candidate comparison baseline horizons are incompatible")
    actual_identity = candidate_identity(
        candidate.training,
        architecture=candidate.architecture,
        seed=candidate.seed,
        epoch_budget=candidate.epoch_budget,
        variance_scale=candidate.variance_scale,
        return_variance_scale=candidate.return_variance_scale,
        comparison_baseline=candidate.comparison_baseline,
        baseline_return_variance_scale=candidate.baseline_return_variance_scale,
    )
    if candidate.model_identity != actual_identity:
        raise ValueError("candidate content identity does not match weights and metadata")
    return candidate


def assemble_release_bundle(
    candidate_dir: Path,
    output_dir: Path,
    *,
    private_key_path: Path,
    public_key_path: Path | None = None,
    opset_version: int = 18,
    parity_rows: int = 7,
) -> dict[str, object]:
    """Convert one locked-certification candidate into one signed ONNX bundle.

    Every seed member is reloaded from verified persisted weights, exported to
    the calibrated production graph, and parity-checked against PyTorch before
    the bundle is signed. Serving metadata follows the frozen volatility
    runtime schema so the CPU runtime can load the release without inference.
    """
    directory = candidate_dir.resolve()
    try:
        manifest = json.loads((directory / "candidate-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("candidate manifest is missing or invalid") from error
    if manifest.get("artifact_role") != "locked_certification_candidate":
        raise ValueError("only locked-certification candidates may be released")
    model_id = manifest.get("model_identity")
    if not isinstance(model_id, str) or not model_id or len(model_id) > 128:
        raise ValueError("candidate model identity cannot serve as a release model id")
    protocol_payload = manifest.get("protocol")
    horizons = protocol_payload.get("horizons") if isinstance(protocol_payload, dict) else None
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons)
        or tuple(sorted(horizons)) != tuple(horizons)
    ):
        raise ValueError("certification protocol horizons are malformed")
    architecture_payload = manifest.get("architecture")
    members_payload = manifest.get("members")
    if not isinstance(architecture_payload, dict) or not isinstance(members_payload, list):
        raise ValueError("candidate manifest is incomplete")
    architecture = BaselineResidualTCNConfig(**architecture_payload)
    if architecture.feature_count != len(DEPLOYABLE_FEATURE_COLUMNS_V5):
        raise ValueError("release candidates must use the certified deployable_v5 feature schema")
    if architecture.horizon_count != len(horizons):
        raise ValueError("architecture horizon count does not match the certified protocol")
    seeds = sorted(
        int(row.get("seed")) for row in members_payload if isinstance(row, dict) and "seed" in row
    )
    if len(seeds) != len(members_payload) or len(set(seeds)) != len(seeds) or any(s < 1 for s in seeds):
        raise ValueError("candidate member table is malformed")

    try:
        from release.bundle import RUNTIME_SCHEMA_VERSION
    except ImportError:  # pragma: no cover - research harness runs from the repository root
        from backend.release.bundle import RUNTIME_SCHEMA_VERSION  # type: ignore[no-redef]

    files: dict[str, bytes] = {}
    parity_errors: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="volatility-release-") as temp:
        temp_dir = Path(temp)
        for seed in seeds:
            candidate = load_frozen_candidate_member(directory, seed)
            onnx_path = export_candidate_onnx(
                candidate,
                temp_dir / f"seed-{seed}.onnx",
                opset_version=opset_version,
            )
            member_errors = verify_onnx_parity(candidate, onnx_path, rows=parity_rows)
            parity_errors[f"seed-{seed}"] = max(member_errors.values())
            files[f"members/seed-{seed}.onnx"] = onnx_path.read_bytes()
    metadata = {
        "runtime_schema": RUNTIME_SCHEMA_VERSION,
        "model_id": model_id,
        "window_size": int(architecture.window_size),
        "horizons": [int(value) for value in horizons],
        "feature_names": list(DEPLOYABLE_FEATURE_COLUMNS_V5),
        "news_feature_count": int(architecture.news_feature_count),
        "members": [{"seed": seed, "file": f"members/seed-{seed}.onnx"} for seed in seeds],
    }
    locked = manifest.get("locked_certification")
    if isinstance(locked, dict):
        certified_horizons = locked.get("certified_horizons")
        if isinstance(certified_horizons, list) and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in certified_horizons
        ):
            metadata["certified_horizons"] = sorted(certified_horizons)
        horizon_decisions = locked.get("horizon_decisions")
        if isinstance(horizon_decisions, dict):
            certification_metrics: dict[str, dict] = {}
            for raw_horizon, summaries in horizon_decisions.items():
                if not isinstance(raw_horizon, str) or not raw_horizon.isdecimal():
                    continue
                if isinstance(summaries, dict):
                    certification_metrics[raw_horizon] = summaries
            if certification_metrics:
                metadata["certification_metrics"] = certification_metrics
    try:
        from release.bundle import build_release, verify_release
    except ImportError:  # pragma: no cover - research harness runs from the repository root
        from backend.release.bundle import build_release, verify_release  # type: ignore[no-redef]

    build_release(output_dir, files, metadata, private_key_path=Path(private_key_path))
    if public_key_path is not None:
        verify_release(output_dir, public_key_path=Path(public_key_path))
    return {
        "model_id": model_id,
        "bundle": str(Path(output_dir).resolve()),
        "member_seeds": seeds,
        "max_parity_error": max(parity_errors.values()) if parity_errors else 0.0,
        "parity_errors": parity_errors,
        "signed_release_metadata": metadata,
    }
