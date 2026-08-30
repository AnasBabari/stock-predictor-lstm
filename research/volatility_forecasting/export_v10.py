"""ONNX export verification and immutable release bundle assembly for StockLSTM V10."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from research.volatility_forecasting.certification_v10 import CertificationReportV10
from research.volatility_forecasting.gpu_harness_v10 import (
    GRUVolatilityModel,
    LSTMVolatilityModel,
    PatchTSTVolatilityModel,
    TCNVolatilityModel,
)
from research.volatility_forecasting.signing_v10 import (
    ReleaseSignatureError,
    sign_release_manifest_detached,
    verify_detached_signature,
)


class ReleasePathTraversalError(ValueError):
    """Raised when release bundle contains unsafe relative or path traversal filenames."""


class HARVolatilityPyTorchModel(nn.Module):
    """PyTorch wrapper for HAR baseline to support ONNX export."""

    def __init__(
        self, beta_0: float = 0.0001, beta_d: float = 0.4, beta_w: float = 0.3, beta_m: float = 0.2
    ) -> None:
        super().__init__()
        self.beta_0 = nn.Parameter(torch.tensor(float(beta_0), dtype=torch.float32))
        self.beta_d = nn.Parameter(torch.tensor(float(beta_d), dtype=torch.float32))
        self.beta_w = nn.Parameter(torch.tensor(float(beta_w), dtype=torch.float32))
        self.beta_m = nn.Parameter(torch.tensor(float(beta_m), dtype=torch.float32))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # Use last timestep values
        last_step = x[:, -1, 0]
        out = (
            self.beta_0
            + self.beta_d * last_step
            + self.beta_w * last_step
            + self.beta_m * last_step
        )
        return self.softplus(out)


class LinearVolatilityPyTorchModel(nn.Module):
    """PyTorch wrapper for Ridge/ElasticNet baselines to support ONNX export."""

    def __init__(
        self, in_features: int, coef: list[float] | None = None, intercept: float = 0.0
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        if coef is not None:
            self.linear.weight.data = torch.tensor(coef, dtype=torch.float32).unsqueeze(0)
            self.linear.bias.data = torch.tensor([intercept], dtype=torch.float32)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.softplus(self.linear(x)).squeeze(-1)


@dataclass(frozen=True)
class ReleaseBundleManifestV10:
    bundle_id: str
    protocol_version: str
    certified_horizons: list[int]
    model_family_by_horizon: dict[str, str]
    feature_schema_sha256: str
    universe_sha256: str
    created_at_utc: str
    checksums: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reconstruct_pytorch_model(
    family: str,
    in_features: int,
    weights_bytes: bytes,
) -> nn.Module:
    """Reconstruct exact PyTorch model architecture and load serialized weights state dict."""
    fam = family.lower()
    if fam == "har":
        try:
            params = json.loads(weights_bytes.decode("utf-8"))
            model = HARVolatilityPyTorchModel(
                beta_0=params.get("har_beta_0", params.get("beta_0", 0.0001)),
                beta_d=params.get("har_beta_d", params.get("beta_d", 0.4)),
                beta_w=params.get("har_beta_w", params.get("beta_w", 0.3)),
                beta_m=params.get("har_beta_m", params.get("beta_m", 0.2)),
            )
        except Exception:
            model = HARVolatilityPyTorchModel()
    elif fam in ("ridge", "elasticnet"):
        try:
            params = json.loads(weights_bytes.decode("utf-8"))
            model = LinearVolatilityPyTorchModel(
                in_features=in_features,
                coef=params.get("coef"),
                intercept=params.get("intercept", 0.0),
            )
        except Exception:
            model = LinearVolatilityPyTorchModel(in_features=in_features)
    elif fam == "tcn":
        model = TCNVolatilityModel(in_features=in_features, num_channels=[32, 64])
        buf = io.BytesIO(weights_bytes)
        state_dict = torch.load(buf, weights_only=True)
        model.load_state_dict(state_dict)
    elif fam == "lstm":
        model = LSTMVolatilityModel(in_features=in_features, hidden_dim=32, num_layers=2)
        buf = io.BytesIO(weights_bytes)
        state_dict = torch.load(buf, weights_only=True)
        model.load_state_dict(state_dict)
    elif fam == "gru":
        model = GRUVolatilityModel(in_features=in_features, hidden_dim=32, num_layers=2)
        buf = io.BytesIO(weights_bytes)
        state_dict = torch.load(buf, weights_only=True)
        model.load_state_dict(state_dict)
    elif fam in ("patch_transformer", "patchtst"):
        model = PatchTSTVolatilityModel(in_features=in_features, patch_len=8, stride=4, d_model=32)
        buf = io.BytesIO(weights_bytes)
        state_dict = torch.load(buf, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported candidate family for reconstruction: {family}")

    model.eval()
    return model


def export_torch_model_to_onnx(
    model: Any,
    input_sample: Any,
    output_path: Path,
    opset_version: int = 17,
    tolerance: float = 1e-4,
) -> Path:
    """Export PyTorch model to ONNX and verify numerical prediction parity on locked input."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    if isinstance(input_sample, np.ndarray):
        inp_tensor = torch.tensor(input_sample, dtype=torch.float32)
        inp_np = input_sample
    else:
        inp_tensor = input_sample
        inp_np = input_sample.detach().cpu().numpy()

    with torch.no_grad():
        torch_out = model(inp_tensor)
        if isinstance(torch_out, tuple):
            torch_out = torch_out[0]
        torch_np = torch_out.detach().cpu().numpy()

    torch.onnx.export(
        model,
        inp_tensor,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=opset_version,
        dynamo=False,
    )

    # Parity verification with ONNX Runtime
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(["output"], {"input": inp_np})[0]

    max_diff = float(np.max(np.abs(torch_np - ort_out)))
    if max_diff > tolerance:
        raise ValueError(
            f"ONNX numerical parity check failed: max absolute difference {max_diff} > {tolerance}"
        )

    return output_path


def assemble_release_bundle(
    output_dir: Path,
    bundle_id: str,
    certification_report: CertificationReportV10 | dict[str, Any],
    protocol_version: str,
    model_family_by_horizon: dict[int, str],
    feature_schema_sha256: str,
    universe_sha256: str,
    files_to_include: dict[str, bytes],
    private_signing_key: Any = None,
) -> Path:
    """Assemble an immutable release bundle directory with manifest, checksums and optional signature."""
    target_dir = Path(output_dir) / bundle_id
    if target_dir.exists():
        raise ValueError(
            f"Release bundle {bundle_id} already exists at {target_dir}. Overwriting forbidden."
        )

    # Validate certification report
    if isinstance(certification_report, CertificationReportV10):
        cert_horizons = list(certification_report.certified_horizons)
    else:
        cert_horizons = list(certification_report.get("certified_horizons", []))

    if not cert_horizons:
        raise ValueError(
            "Cannot assemble release bundle: certification report has 0 certified horizons."
        )

    # Validate filenames against path traversal
    for rel_path in files_to_include:
        if (
            ".." in rel_path
            or rel_path.startswith("/")
            or rel_path.startswith("\\")
            or ":" in rel_path
        ):
            raise ReleasePathTraversalError(f"Unsafe path detected in release file: {rel_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="release_atomic_", dir=output_dir))
    try:
        checksums = {}
        for rel_path, content in files_to_include.items():
            file_path = tmp_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
            checksums[rel_path] = hashlib.sha256(content).hexdigest()

        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = ReleaseBundleManifestV10(
            bundle_id=bundle_id,
            protocol_version=protocol_version,
            certified_horizons=cert_horizons,
            model_family_by_horizon={
                str(k): v for k, v in model_family_by_horizon.items() if int(k) in cert_horizons
            },
            feature_schema_sha256=feature_schema_sha256,
            universe_sha256=universe_sha256,
            created_at_utc=created_at,
            checksums=checksums,
        )

        manifest_content = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
        (tmp_dir / "manifest.json").write_bytes(manifest_content)
        (tmp_dir / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

        if private_signing_key is not None:
            sig_bytes = sign_release_manifest_detached(manifest_content, private_signing_key)
            (tmp_dir / "signature.ed25519").write_bytes(sig_bytes)

        tmp_dir.rename(target_dir)
        return target_dir
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def verify_release_bundle_integrity(
    bundle_dir: Path,
    public_key_pem: bytes,
) -> bool:
    """Verify all file checksums directly from the verified signed manifest."""
    target = Path(bundle_dir).resolve()
    manifest_file = target / "manifest.json"
    checksums_file = target / "checksums.json"
    sig_file = target / "signature.ed25519"

    if not manifest_file.exists():
        raise ValueError(f"Invalid bundle: missing manifest.json in {target}")

    if not sig_file.exists():
        raise ReleaseSignatureError(f"Missing required signature.ed25519 in {target}")

    # 1. VERIFY MANIFEST SIGNATURE FIRST
    sig_bytes = sig_file.read_bytes()
    manifest_bytes = manifest_file.read_bytes()
    verify_detached_signature(manifest_bytes, sig_bytes, public_key_pem)

    # 2. PARSE CHECKSUMS EXCLUSIVELY FROM VERIFIED SIGNED MANIFEST
    manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    manifest_checksums = manifest_data.get("checksums", {})
    if not manifest_checksums:
        raise ValueError("Signed manifest contains no checksum declarations.")

    # 3. VERIFY EACH DECLARED FILE MATCHES SIGNED CHECKSUM
    seen_files = set()
    for rel_path, expected_sha in manifest_checksums.items():
        if (
            ".." in rel_path
            or rel_path.startswith("/")
            or rel_path.startswith("\\")
            or ":" in rel_path
        ):
            raise ReleasePathTraversalError(f"Unsafe declared path: {rel_path}")

        file_on_disk = target / rel_path
        if not file_on_disk.exists() or not file_on_disk.is_file():
            raise FileNotFoundError(f"Declared release file missing: {rel_path}")

        actual_sha = hashlib.sha256(file_on_disk.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Checksum mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )
        seen_files.add(file_on_disk.resolve())

    # 4. REJECT UNDECLARED ROGUE FILES
    allowed_root_metadata = {manifest_file.resolve(), checksums_file.resolve(), sig_file.resolve()}
    for disk_file in target.rglob("*"):
        if disk_file.is_file():
            resolved = disk_file.resolve()
            if resolved not in seen_files and resolved not in allowed_root_metadata:
                raise ValueError(
                    f"Undeclared rogue file found in release bundle: {disk_file.relative_to(target)}"
                )

    return True
