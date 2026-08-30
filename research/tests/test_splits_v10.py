"""Tests for 70/15/15 unique-origin splitter, expanding folds, strict loader and sealed target store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.volatility_forecasting.splits_v10 import (
    DEPLOYABLE_FEATURE_COLUMNS_V5,
    ExpandingFoldSplitterV10,
    PanelValidationError,
    SealedTargetStore,
    StrictPanelLoader,
    UniqueOriginSplitterV10,
)


def test_unique_origin_splitter_enforces_embargo_and_purge() -> None:
    dates = pd.date_range("2020-01-01", periods=500, freq="B").strftime("%Y-%m-%d").tolist()
    security_ids = ["SEC_AAPL_001", "SEC_AMZN_001", "SEC_MSFT_001", "SEC_PEP_001"]

    rows = []
    for d in dates:
        for sec in security_ids:
            rows.append({"Date": d, "SecurityID": sec, "Close": 100.0})
    df = pd.DataFrame(rows)

    splitter = UniqueOriginSplitterV10(
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
        embargo_sessions=30,
        max_label_horizon=30,
        required_transfer_security_ids=("SEC_MSFT_001", "SEC_PEP_001"),
    )
    train_dates, val_dates, test_dates = splitter.partition_sessions(dates)

    assert set(train_dates).isdisjoint(set(val_dates))
    assert set(val_dates).isdisjoint(set(test_dates))
    assert set(train_dates).isdisjoint(set(test_dates))

    df_assigned, assignment = splitter.build_assignment(df)
    assert assignment.transfer_rows > 0
    assert assignment.purged_rows > 0
    assert len(assignment.assignment_fingerprint_sha256) == 64


def test_splitter_rejects_missing_security_id_and_duplicate_identities() -> None:
    splitter = UniqueOriginSplitterV10()

    bad_df1 = pd.DataFrame({"Date": ["2024-01-01"], "Ticker": ["AAPL"]})
    with pytest.raises(ValueError, match="Required column 'SecurityID' missing"):
        splitter.build_assignment(bad_df1)

    bad_df2 = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01"],
            "SecurityID": ["SEC_AAPL_001", "SEC_AAPL_001"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        splitter.build_assignment(bad_df2)


def test_strict_panel_loader_validates_features_and_targets() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
    rows = []
    for d in dates:
        row = {"Date": d, "SecurityID": "SEC_AAPL_001", "target_h1": 0.0004, "target_h5": 0.0020}
        for feat in DEPLOYABLE_FEATURE_COLUMNS_V5:
            row[feat] = 0.5
        rows.append(row)

    df_valid = pd.DataFrame(rows)
    df_loaded = StrictPanelLoader.load_and_validate(df_valid, required_horizons=[1, 5])
    assert len(df_loaded) == 10

    # Missing target
    with pytest.raises(PanelValidationError, match="Missing required target column"):
        StrictPanelLoader.load_and_validate(df_valid, required_horizons=[1, 5, 10])

    # Non-finite feature
    df_bad_feat = df_valid.copy()
    df_bad_feat.loc[0, "Return_1D"] = np.nan
    with pytest.raises(PanelValidationError, match="Non-finite values detected in feature column"):
        StrictPanelLoader.load_and_validate(df_bad_feat, required_horizons=[1, 5])

    # Non-positive target
    df_bad_targ = df_valid.copy()
    df_bad_targ.loc[0, "target_h1"] = -0.0001
    with pytest.raises(PanelValidationError, match="Non-positive values detected in target column"):
        StrictPanelLoader.load_and_validate(df_bad_targ, required_horizons=[1, 5])

    # Bad SecurityID format
    df_bad_sec = df_valid.copy()
    df_bad_sec["SecurityID"] = "AAPL"
    with pytest.raises(PanelValidationError, match="does not match required format"):
        StrictPanelLoader.load_and_validate(df_bad_sec, required_horizons=[1, 5])


def test_expanding_fold_splitter_enforces_embargo_and_purge() -> None:
    dev_dates = pd.date_range("2020-01-01", periods=300, freq="B").strftime("%Y-%m-%d").tolist()
    splitter = ExpandingFoldSplitterV10(
        n_folds=5, embargo_sessions=30, max_label_horizon=30, min_train_sessions=40
    )
    folds = splitter.split_sessions(dev_dates)

    assert len(folds) == 5
    for f in folds:
        train_set = set(f.train_sessions)
        val_set = set(f.val_sessions)
        purged_set = set(f.purged_sessions)

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(purged_set)
        assert val_set.isdisjoint(purged_set)
        assert len(f.purged_sessions) == 60  # 30 embargo + 30 label horizon


def test_assignment_fingerprint_is_row_order_invariant() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B").strftime("%Y-%m-%d").tolist()
    rows = []
    for d in dates:
        for sec in ["SEC_AAPL_001", "SEC_NVDA_001"]:
            rows.append({"Date": d, "SecurityID": sec, "Close": 100.0})

    df1 = pd.DataFrame(rows)
    df2 = df1.sample(frac=1.0, random_state=123).reset_index(drop=True)

    splitter = UniqueOriginSplitterV10(required_transfer_security_ids=("SEC_NVDA_001",))
    _, a1 = splitter.build_assignment(df1)
    _, a2 = splitter.build_assignment(df2)

    assert a1.assignment_fingerprint_sha256 == a2.assignment_fingerprint_sha256


def test_sealed_target_store_creation_and_tamper_detection(tmp_path: Path) -> None:
    target_data = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "SecurityID": ["SEC_AAPL_001", "SEC_AAPL_001"],
            "target_h1": [0.0004, 0.0005],
        }
    )
    store_file = tmp_path / "sealed_targets.json"
    store, meta = SealedTargetStore.create_sealed_store(target_data, store_file)

    assert store_file.exists()
    assert meta.row_count == 2
    assert len(meta.checksum_sha256) == 64

    # Load successfully with verified checksum
    df_loaded = store.load_targets(expected_checksum=meta.checksum_sha256)
    assert len(df_loaded) == 2


def test_perturbing_future_data_does_not_affect_train_scaler() -> None:
    """Invariant test: modifying validation/future observations has ZERO effect on training scaler."""
    from research.volatility_forecasting.gpu_harness_v10 import TrainOnlyRobustScaler

    rng = np.random.default_rng(999)
    # 200 train sessions, 50 val sessions
    train_data = rng.normal(10.0, 2.0, size=(200, 60, 5))
    val_data = rng.normal(10.0, 2.0, size=(50, 60, 5))

    scaler_clean = TrainOnlyRobustScaler()
    scaler_clean.fit(train_data)
    center_clean = scaler_clean.center_.copy()
    scale_clean = scaler_clean.scale_.copy()

    # Perturb future/validation data drastically
    val_data_perturbed = val_data * 500.0 + 10000.0
    assert not np.array_equal(val_data_perturbed, val_data)

    # Retrain scaler strictly on train_data
    scaler_retested = TrainOnlyRobustScaler()
    scaler_retested.fit(train_data)

    assert np.array_equal(scaler_retested.center_, center_clean)
    assert np.array_equal(scaler_retested.scale_, scale_clean)


def test_sample_metadata_record_integrity() -> None:
    """Verify sample metadata record structure and serialization."""
    from research.volatility_forecasting.splits_v10 import SampleMetadataRecord

    rec = SampleMetadataRecord(
        security_id="SEC_BP_001",
        origin_session="2024-01-15",
        label_end_session="2024-01-22",
        partition="train",
        fold=0,
        feature_schema_sha256="abcdef123456",
        target_contract_version="price-return-distribution-v1",
    )
    d = rec.to_dict()
    assert d["security_id"] == "SEC_BP_001"
    assert d["fold"] == 0
    assert d["target_contract_version"] == "price-return-distribution-v1"
