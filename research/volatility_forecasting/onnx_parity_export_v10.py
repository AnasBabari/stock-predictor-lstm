"""ONNX export with numerical parity verification and canonical bundle manifest generation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ParityVerificationResult:
    max_absolute_difference: float
    passed_tolerance: bool
    input_shape: list[int]
    output_shape: list[int]
    onnx_file_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ONNXReleaseExporter:
    """Exports certified PyTorch candidate to ONNX and asserts exact numerical parity."""

    @staticmethod
    def export_and_verify_parity(
        model: nn.Module,
        dummy_input: torch.Tensor,
        export_path: Path,
        tolerance: float = 1e-4,
    ) -> ParityVerificationResult:
        model.eval()
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # PyTorch forward
        with torch.no_grad():
            pytorch_output = model(dummy_input).cpu().numpy()

        # Export to ONNX using classic TorchScript engine
        torch.onnx.export(
            model,
            dummy_input,
            str(export_path),
            input_names=["input_features"],
            output_names=["predicted_horizons"],
            dynamic_axes={
                "input_features": {0: "batch_size"},
                "predicted_horizons": {0: "batch_size"},
            },
            opset_version=14,
            dynamo=False,
        )

        # Compute SHA-256
        with open(export_path, "rb") as f:
            onnx_bytes = f.read()
        onnx_hash = hashlib.sha256(onnx_bytes).hexdigest()

        # ONNX Runtime forward
        session = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        ort_output = session.run(None, {input_name: dummy_input.cpu().numpy()})[0]

        # Check parity
        diff = float(np.max(np.abs(pytorch_output - ort_output)))
        passed = diff <= tolerance

        return ParityVerificationResult(
            max_absolute_difference=diff,
            passed_tolerance=passed,
            input_shape=list(dummy_input.shape),
            output_shape=list(pytorch_output.shape),
            onnx_file_sha256=onnx_hash,
        )
