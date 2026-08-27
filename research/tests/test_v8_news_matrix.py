from __future__ import annotations

import numpy as np
import pytest

from research.volatility_forecasting.data import VolatilityPanelExamples
from research.volatility_forecasting.news import NewsFeatureMatrix
from research.volatility_forecasting.news_matrix_v8 import build_v8_aligned_news_matrix


def _examples() -> VolatilityPanelExamples:
    return VolatilityPanelExamples(
        features=np.ones((2, 2, 1), dtype=np.float32),
        baseline_variance=np.ones((2, 1), dtype=np.float32),
        realized_variance=np.ones((2, 1), dtype=np.float32),
        cumulative_returns=np.zeros((2, 1), dtype=np.float32),
        direction_classes=np.ones((2, 1), dtype=np.int64),
        tickers=np.asarray(["MSFT", "VOD.L"]),
        origin_dates=np.asarray(["2024-07-03", "2024-07-04"], dtype="datetime64[D]"),
        origin_closes=np.ones(2),
        horizons=(1,),
        feature_names=("x",),
    )


def _patch_dependencies(monkeypatch, *, returned_tickers=None) -> None:
    universe = {
        "members": [
            {"ticker": "MSFT", "primary_exchange_mic": "XNAS", "security_id": "SEC-MSFT"},
            {"ticker": "VOD.L", "primary_exchange_mic": "XLON", "security_id": "SEC-VOD"},
        ]
    }
    monkeypatch.setattr(
        "research.volatility_forecasting.news_matrix_v8.verify_universe_manifest",
        lambda value: universe,
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.news_matrix_v8.verify_v8_news_manifest",
        lambda *args, **kwargs: {
            "coverage_complete": True,
            "feature_lookback_days": 20,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.news_matrix_v8.load_news_snapshot",
        lambda path: ((), {}),
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.news_matrix_v8.universe_identity_maps",
        lambda value: ({"MSFT": "XNAS", "VOD.L": "XLON"}, {}),
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.news_matrix_v8.load_news_exposure_map",
        lambda *args, **kwargs: type(
            "Exposure", (), {"exposures": {}, "source_sha256": "b" * 64}
        )(),
    )
    tickers = (
        np.asarray(returned_tickers)
        if returned_tickers is not None
        else np.asarray(["MSFT", "VOD.L"])
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.news_matrix_v8.aggregate_news_for_market_rows",
        lambda *args, **kwargs: NewsFeatureMatrix(
            values=np.asarray([[1.0], [2.0]], dtype=np.float32),
            tickers=tickers,
            cutoffs=np.asarray(
                ["2024-07-03T17:00:00", "2024-07-04T15:30:00"],
                dtype="datetime64[ns]",
            ),
            feature_names=("news",),
        ),
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.news_matrix_v8.validate_news_coverage",
        lambda *args, **kwargs: None,
    )


def test_v8_news_matrix_binds_rows_snapshot_and_exposure(monkeypatch, tmp_path) -> None:
    _patch_dependencies(monkeypatch)
    matrix = build_v8_aligned_news_matrix(
        _examples(),
        news_snapshot_dir=tmp_path,
        news_manifest={},
        universe_manifest={},
        market_manifest={},
        ticker_aliases_path=tmp_path / "aliases.json",
        exposure_map_path=tmp_path / "exposures.json",
    )
    assert matrix.values.shape == (2, 1)
    assert matrix.snapshot_sha256 == "sha256:" + "a" * 64
    assert matrix.exposure_sha256 == "sha256:" + "b" * 64
    assert matrix.matrix_sha256.startswith("sha256:")


def test_v8_news_matrix_rejects_row_reordering(monkeypatch, tmp_path) -> None:
    _patch_dependencies(monkeypatch, returned_tickers=["VOD.L", "MSFT"])
    with pytest.raises(RuntimeError, match="row order"):
        build_v8_aligned_news_matrix(
            _examples(),
            news_snapshot_dir=tmp_path,
            news_manifest={},
            universe_manifest={},
            market_manifest={},
            ticker_aliases_path=tmp_path / "aliases.json",
            exposure_map_path=tmp_path / "exposures.json",
        )
