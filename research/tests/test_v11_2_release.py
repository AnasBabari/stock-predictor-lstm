"""V11.2 remains reproducible as research code but can never ship."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from volatility_forecasting.export_v11_2 import V112ProductionGraph, assemble_v11_2_release
from volatility_forecasting.model import RobustSequenceScaler
from volatility_forecasting.v11_2_protocol import V112CertificationRetiredError

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5
from backend.services.volatility_runtime.contracts import (
    RUNTIME_SCHEMA_VERSION,
    VolatilityRuntimeContract,
)


def test_v112_graph_supports_historical_baseline_reproduction() -> None:
    graph = V112ProductionGraph(
        models={},
        scaler=RobustSequenceScaler(np.zeros(26), np.ones(26)),
    ).eval()
    variance, location, probabilities, return_variance = graph(
        torch.zeros(2, 60, 26), torch.ones(2, 6)
    )
    assert tuple(variance.shape) == (2, 6)
    assert tuple(location.shape) == (2, 6)
    assert tuple(probabilities.shape) == (2, 6, 3)
    assert tuple(return_variance.shape) == (2, 6)
    torch.testing.assert_close(variance, torch.ones(2, 6))
    torch.testing.assert_close(location, torch.zeros(2, 6))
    torch.testing.assert_close(probabilities, torch.full((2, 6, 3), 1.0 / 3.0))
    torch.testing.assert_close(return_variance, variance)


def test_runtime_contract_rejects_retired_v112_generation() -> None:
    metadata = {
        "runtime_schema": RUNTIME_SCHEMA_VERSION,
        "model_id": "v11-2-fixture",
        "model_version": "v11.2-numeric-residual-v1",
        "protocol_version": "v11.2-numeric-pit64",
        "feature_names": list(DEPLOYABLE_FEATURE_COLUMNS_V5),
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
    with pytest.raises(ValueError, match="permanently retired"):
        VolatilityRuntimeContract.from_release_metadata(
            metadata,
            {"members/model.onnx"},
        )


def test_v112_release_assembly_is_permanently_retired(tmp_path: Path) -> None:
    with pytest.raises(V112CertificationRetiredError, match="INVALIDATED_OPENED"):
        assemble_v11_2_release(
            results_dir=tmp_path / "results",
            certification_dir=tmp_path / "certification",
            output_dir=tmp_path / "release",
            private_key_path=tmp_path / "private-key.pem",
        )
    assert not (tmp_path / "release").exists()
