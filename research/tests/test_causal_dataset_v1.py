"""Unit tests for strict causal dataset loader."""

import numpy as np
import pandas as pd
import pytest

from research.volatility_forecasting.causal_dataset_v1 import (
    STATIONARY_FEATURE_COLUMNS_V1,
    StrictCausalDatasetLoader,
)


def test_stationary_features_derivation():
    """Verify that stationary features derive correctly without NaNs after warmup."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
    rng = np.random.default_rng(42)

    # Synthetic OHLCV
    close = np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=100))) * 50.0
    high = close * (1.0 + rng.uniform(0.002, 0.02, size=100))
    low = close * (1.0 - rng.uniform(0.002, 0.02, size=100))
    open_p = (high + low) / 2.0
    vol = rng.uniform(1e6, 5e6, size=100)

    df_ohlcv = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": vol,
        },
        index=dates,
    )

    feats = StrictCausalDatasetLoader.compute_stationary_features(df_ohlcv)
    assert len(feats) == 80  # 100 - 20 warmup rows
    assert list(feats.columns) == list(STATIONARY_FEATURE_COLUMNS_V1)
    assert not feats.isna().any().any()
    assert np.isfinite(feats.to_numpy()).all()


def test_validate_panel_dataframe_duplicates_and_format():
    """Verify panel validation rejects duplicate rows and invalid SecurityID formats."""
    dates = ["2023-01-02", "2023-01-03", "2023-01-04"]
    rows = []
    for d in dates:
        row = {"Date": d, "SecurityID": "SEC_BP_001"}
        for f in STATIONARY_FEATURE_COLUMNS_V1:
            row[f] = 0.01
        rows.append(row)

    df = pd.DataFrame(rows)
    sorted_df, meta = StrictCausalDatasetLoader.validate_panel_dataframe(df)
    assert meta.security_count == 1
    assert meta.session_count == 3
    assert len(meta.snapshot_hash) == 64

    # Duplicate rejection
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        StrictCausalDatasetLoader.validate_panel_dataframe(df_dup)

    # Invalid SecurityID rejection
    df_bad_sec = df.copy()
    df_bad_sec["SecurityID"] = "BP"  # Missing SEC_ prefix and ID
    with pytest.raises(ValueError, match="does not match contract format"):
        StrictCausalDatasetLoader.validate_panel_dataframe(df_bad_sec)
