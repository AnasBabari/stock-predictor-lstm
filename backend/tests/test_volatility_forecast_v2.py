from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app
from config import settings
from routes import volatility_v2
from services.volatility_runtime.contracts import VOLATILITY_HORIZONS

CLIENT = TestClient(app)

QUANTILE_KEYS = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")


class _FakeRuntime:
    model_id = "global-volatility-tcn-v1"
    member_seeds = (41, 42, 43)

    def __init__(self, variance: float = 4e-4) -> None:
        self.variance = variance
        self.calls = 0

    def forecast(self, snapshot):
        self.calls += 1
        return SimpleNamespace(
            forecast_variance=np.full(len(VOLATILITY_HORIZONS), self.variance, dtype=np.float32),
            return_location=np.zeros(len(VOLATILITY_HORIZONS), dtype=np.float32),
            direction_probabilities=np.full((len(VOLATILITY_HORIZONS), 3), 1 / 3, dtype=np.float32),
            return_variance=np.full(len(VOLATILITY_HORIZONS), self.variance * 1.5),
        )


def _fake_snapshot(ticker: str = "NMM"):
    return SimpleNamespace(
        ticker=ticker,
        snapshot_id="a" * 64,
        origin_date="2026-08-21",
        origin_close=88.78,
        feature_names=("f1", "f2"),
        features=None,
        causal_har_variance=np.full(len(VOLATILITY_HORIZONS), 2e-4),
        baseline_candidates={},
        historical_dates=(),
        historical_prices=np.array([88.0]),
    )


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    volatility_v2._reset_release_state()
    volatility_v2._response_cache = volatility_v2._ResponseCache()
    monkeypatch.setattr(settings, "volatility_release_dir", None)
    monkeypatch.setattr(settings, "volatility_public_key_path", None)
    monkeypatch.setattr(settings, "volatility_forecast_cache_ttl", 900)
    yield
    volatility_v2._reset_release_state()


def _install_release(
    monkeypatch, *, release_dir: str | None = "/signed/release", runtime=None
) -> None:
    if release_dir is not None:
        monkeypatch.setattr(settings, "volatility_release_dir", release_dir)
        monkeypatch.setattr(settings, "volatility_public_key_path", "/keys/public.pem")
    if runtime is not None:
        monkeypatch.setattr(volatility_v2, "_RELEASE_STATE", volatility_v2._ReleaseState())
        monkeypatch.setattr(volatility_v2._ReleaseState, "get", lambda self: (runtime, None))


def test_abstains_when_no_release_is_configured() -> None:
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "abstain_no_certified_model"
    assert "configured" in detail["reason"]


def test_abstains_when_release_verification_fails(monkeypatch) -> None:
    _install_release(monkeypatch)
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 7})
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "abstain_no_certified_model"


def test_serves_certified_volatility_cone(monkeypatch) -> None:
    runtime = _FakeRuntime(variance=4e-4)
    _install_release(monkeypatch, runtime=runtime)
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        lambda ticker: _fake_snapshot(ticker),
    )
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "nmm", "horizon": 7})
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "NMM"
    assert body["as_of"] == "2026-08-21"
    assert body["current_price"] == pytest.approx(88.78)
    quantiles = body["forecast"]["price_quantiles"]
    values = [quantiles[key][0] for key in QUANTILE_KEYS]
    assert values == sorted(values)
    assert all(value > 0 for value in values)
    assert quantiles["p05"][0] < body["current_price"] < quantiles["p95"][0]
    assert quantiles["p50"][0] == pytest.approx(88.78, rel=1e-9)
    expected_annualized = float(np.sqrt((4e-4 / 7) * 252))
    assert body["forecast"]["expected_annualized_volatility"] == pytest.approx(
        expected_annualized, rel=1e-6
    )
    assert body["forecast"]["expected_cumulative_variance"] == pytest.approx(4e-4)
    assert body["forecast"]["probability_up"] is None
    evidence = body["evidence"]
    assert evidence["model_id"] == "global-volatility-tcn-v1"
    assert evidence["member_seeds"] == [41, 42, 43]
    assert evidence["metric_source"] == "locked_purged_walk_forward"
    assert evidence["certified_heads"] == {
        "volatility": True,
        "return_distribution": False,
        "direction": False,
    }
    assert evidence["certified"] is True
    assert "no learned direction claim" in evidence["quantile_model"]


def test_repeated_requests_reuse_the_response_cache(monkeypatch) -> None:
    runtime = _FakeRuntime()
    _install_release(monkeypatch, runtime=runtime)

    def counting_snapshot(ticker: str):
        counting_snapshot.calls += 1
        return _fake_snapshot(ticker)

    counting_snapshot.calls = 0
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        counting_snapshot,
    )
    first = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    second = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    assert first.status_code == second.status_code == 200
    assert counting_snapshot.calls == 1
    other_ticker = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 7})
    assert other_ticker.status_code == 200
    assert counting_snapshot.calls == 2


def test_rejects_invalid_horizon_and_ticker(monkeypatch) -> None:
    _install_release(monkeypatch, runtime=_FakeRuntime())
    assert CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 4}).status_code == 400
    assert (
        CLIENT.get("/api/v2/forecast", params={"ticker": "DROP TABLE", "horizon": 7}).status_code
        == 400
    )


def test_maps_short_history_to_conflict_and_upstream_to_bad_gateway(monkeypatch) -> None:
    _install_release(monkeypatch, runtime=_FakeRuntime())

    def short_history(ticker: str):
        raise ValueError("market history is too short for volatility inference")

    def upstream_failure(ticker: str):
        raise OSError("network unreachable")

    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot", short_history
    )
    conflict = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    assert conflict.status_code == 409
    assert "cannot support a certified forecast" in conflict.json()["detail"]

    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot", upstream_failure
    )
    unavailable = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    assert unavailable.status_code == 502
    assert "temporarily unavailable" in unavailable.json()["detail"]


def test_maps_inference_failure_to_artifact_integrity(monkeypatch) -> None:
    class _BrokenRuntime(_FakeRuntime):
        def forecast(self, snapshot):
            raise RuntimeError("session backend crashed")

    _install_release(monkeypatch, runtime=_BrokenRuntime())
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        lambda ticker: _fake_snapshot(ticker),
    )
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "artifact_integrity_failure"
