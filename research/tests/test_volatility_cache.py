from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from volatility_forecasting.cache import (
    ExampleCacheError,
    example_cache_key,
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
    save_example_cache,
)
from volatility_forecasting.contracts import VolatilityForecastProtocol
from volatility_forecasting.data import VolatilityPanelExamples


def _examples(protocol: VolatilityForecastProtocol) -> VolatilityPanelExamples:
    rows = 3
    horizon_count = len(protocol.horizons)
    variance = np.full((rows, horizon_count), 0.01, dtype=np.float32)
    return VolatilityPanelExamples(
        features=np.ones((rows, protocol.window_size, protocol.feature_count), dtype=np.float32),
        baseline_variance=variance,
        realized_variance=variance * 1.1,
        cumulative_returns=np.zeros((rows, horizon_count), dtype=np.float32),
        direction_classes=np.ones((rows, horizon_count), dtype=np.int64),
        tickers=np.array(["AAA", "BBB", "CCC"]),
        origin_dates=np.array(["2025-01-02", "2025-01-03", "2025-01-06"], dtype="datetime64[D]"),
        origin_closes=np.array([10.0, 20.0, 30.0]),
        horizons=protocol.horizons,
        feature_names=protocol.feature_names,
    )


def test_example_cache_round_trip_and_identity(tmp_path: Path) -> None:
    protocol = VolatilityForecastProtocol()
    checksum = "sha256:" + "a" * 64
    cache_dir = tmp_path / example_cache_key(checksum, protocol)
    original = _examples(protocol)
    save_example_cache(cache_dir, original, panel_checksum=checksum, protocol=protocol)
    restored = load_example_cache(
        cache_dir,
        panel_checksum=checksum,
        protocol=protocol,
        mmap_mode=None,
    )
    np.testing.assert_array_equal(restored.features, original.features)
    np.testing.assert_array_equal(restored.tickers, original.tickers)
    changed_model = replace(protocol, architecture_version="different-model")
    assert example_cache_key(checksum, changed_model) == example_cache_key(checksum, protocol)
    assert (
        find_compatible_example_cache(
            tmp_path,
            panel_checksum=checksum,
            protocol=changed_model,
        )
        == cache_dir.resolve()
    )


def test_example_cache_rejects_tampered_array(tmp_path: Path) -> None:
    protocol = VolatilityForecastProtocol()
    checksum = "sha256:" + "b" * 64
    cache_dir = tmp_path / "cache"
    save_example_cache(cache_dir, _examples(protocol), panel_checksum=checksum, protocol=protocol)
    with (cache_dir / "origin_closes.npy").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ExampleCacheError, match="checksum"):
        load_example_cache(cache_dir, panel_checksum=checksum, protocol=protocol)


def test_panel_fingerprint_reads_only_valid_manifest_checksum(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"pooled_checksum": "sha256:" + "c" * 64}),
        encoding="utf-8",
    )
    assert panel_fingerprint(tmp_path) == "sha256:" + "c" * 64
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ExampleCacheError, match="pooled checksum"):
        panel_fingerprint(tmp_path)
