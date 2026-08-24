from __future__ import annotations

import json
from pathlib import Path

import pytest
from volatility_forecasting.news import NewsValidationError
from volatility_forecasting.news_exposures import load_news_exposure_map


def test_frozen_exposures_cover_universe_and_encode_shipping_transmission() -> None:
    root = Path(__file__).resolve().parents[2]
    universe = {
        line.strip()
        for line in (root / "configs" / "volatility-universe-v1.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    loaded = load_news_exposure_map(
        root / "configs" / "news-ticker-exposures-v1.json",
        required_tickers=universe,
    )
    assert set(loaded.exposures) == universe
    assert loaded.exposures["NMM"]["shipping_disruption"] == 1.0
    assert loaded.exposures["NMM"]["oil_supply"] >= 0.8
    assert loaded.exposures["MSFT"]["regulation"] >= 0.7
    assert len(loaded.source_sha256) == 64


def test_exposure_loader_rejects_targets_missing_profiles_and_bad_weights(tmp_path: Path) -> None:
    payload = {
        "schema_version": "news-exposure-map-v1",
        "methodology": "not fit to returns",
        "profiles": {"shipping": {"shipping_disruption": 1.1}},
        "ticker_profiles": {"NMM": "shipping"},
        "overrides": {},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NewsValidationError, match=r"\[0, 1\]"):
        load_news_exposure_map(path)

    payload["profiles"]["shipping"]["shipping_disruption"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NewsValidationError, match="coverage mismatch"):
        load_news_exposure_map(path, required_tickers={"NMM", "MSFT"})
