"""Unit tests for ONNXReleaseExporter and numerical parity."""

import tempfile
from pathlib import Path

import torch

from research.volatility_forecasting.causal_models_v1 import CausalTCNModel
from research.volatility_forecasting.onnx_parity_export_v10 import ONNXReleaseExporter


def test_onnx_export_and_numerical_parity():
    in_features = 16
    seq_len = 60
    forecast_days = 7

    model = CausalTCNModel(
        in_features=in_features,
        num_channels=[16, 32],
        forecast_days=forecast_days,
        mode="return",
    )
    model.eval()

    dummy_input = torch.randn(2, seq_len, in_features, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model_return.onnx"
        result = ONNXReleaseExporter.export_and_verify_parity(
            model=model,
            dummy_input=dummy_input,
            export_path=onnx_path,
            tolerance=1e-4,
        )

        assert result.passed_tolerance is True
        assert result.max_absolute_difference < 1e-4
        assert len(result.onnx_file_sha256) == 64
        assert onnx_path.exists()
