"""Unit tests for SplitManifestBuilderV11."""

import tempfile
from pathlib import Path

import pandas as pd

from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
)
from research.volatility_forecasting.split_manifest_v11 import (
    SplitManifestBuilderV11,
)


def test_canonical_split_manifest_construction():
    n_days = 200
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()
    security_ids = ["US.AMGN"] * n_days

    split = ChronologicalPartitionManager.create_70_15_15_split(
        dates=dates, max_horizon_days=7, embargo_sessions=30
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "v11_split_manifest.json"
        manifest = SplitManifestBuilderV11.build_and_save_manifest(
            dates=dates,
            security_ids=security_ids,
            split=split,
            target_path=manifest_path,
        )

        assert manifest.total_rows == 200
        assert len(manifest.partition_assignments_digest) == 64
        assert manifest_path.exists()
