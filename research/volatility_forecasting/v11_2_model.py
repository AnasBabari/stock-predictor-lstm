"""Single source of truth for the V11.2 residual-LSTM architecture.

Training, one-shot certification, and production export must reconstruct the
same module before loading a frozen state dictionary.  Keeping that geometry
in one helper prevents a silent architecture drift between those boundaries.
"""

from __future__ import annotations

from typing import Any

from .model import BaselineResidualLSTM, BaselineResidualTCNConfig

V11_2_RESIDUAL_ARCHITECTURE_VERSION = "v11.2-residual-lstm-32x1-v1"


def v11_2_residual_architecture_manifest(*, feature_count: int, window_size: int) -> dict[str, Any]:
    """Return the portable architecture identity used by every V11.2 stage."""
    if feature_count < 1:
        raise ValueError("V11.2 residual model requires at least one feature")
    if window_size < 2:
        raise ValueError("V11.2 residual model requires at least two time steps")
    return {
        "architecture_version": V11_2_RESIDUAL_ARCHITECTURE_VERSION,
        "feature_count": int(feature_count),
        "horizon_count": 1,
        "encoder_family": "lstm",
        "window_size": int(window_size),
        "channels": 32,
        "lstm_hidden": 32,
        "lstm_layers": 1,
        "dropout": 0.15,
        "patch_length": 2,
        "patch_stride": 1,
    }


def v11_2_residual_config(*, feature_count: int, window_size: int) -> BaselineResidualTCNConfig:
    """Build the exact PyTorch configuration bound to V11.2 state files."""
    manifest = v11_2_residual_architecture_manifest(
        feature_count=feature_count,
        window_size=window_size,
    )
    return BaselineResidualTCNConfig(
        feature_count=int(manifest["feature_count"]),
        horizon_count=int(manifest["horizon_count"]),
        encoder_family="lstm",
        window_size=int(manifest["window_size"]),
        channels=int(manifest["channels"]),
        lstm_hidden=int(manifest["lstm_hidden"]),
        lstm_layers=int(manifest["lstm_layers"]),
        dropout=float(manifest["dropout"]),
        patch_length=int(manifest["patch_length"]),
        patch_stride=int(manifest["patch_stride"]),
    )


def build_v11_2_residual_model(*, feature_count: int, window_size: int) -> BaselineResidualLSTM:
    """Construct the exact V11.2 residual model used across all boundaries."""
    return BaselineResidualLSTM(
        v11_2_residual_config(
            feature_count=feature_count,
            window_size=window_size,
        )
    )
