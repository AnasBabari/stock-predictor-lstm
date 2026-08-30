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
    model_version = "global-volatility-tcn-v1"
    member_seeds = (41, 42, 43)
    metric_source = "locked_purged_walk_forward"
    certification_scope = "prospective_walk_forward"
    news_status = "not_certified"
    certified_heads = {
        "volatility": True,
        "return_distribution": False,
        "direction": False,
    }
    return_distribution_family = "zero_location_normal"
    return_distribution_degrees_of_freedom = None

    def __init__(self, variance: float = 4e-4, return_location: float = 0.0) -> None:
        self.variance = variance
        self.return_location = return_location
        self.calls = 0

    def is_certified_horizon(self, horizon: int) -> bool:
        return horizon in (1, 5, 7)

    def certification_summary(self, horizon: int):
        if horizon not in (1, 5, 7):
            return None
        return {
            "decision": "pass",
            "relative_qlike": 0.85 - 0.05 * (horizon == 7),
            "coverage_80": 0.79,
        }

    def forecast(self, snapshot):
        self.calls += 1
        return SimpleNamespace(
            forecast_variance=np.full(len(VOLATILITY_HORIZONS), self.variance, dtype=np.float32),
            return_location=np.full(
                len(VOLATILITY_HORIZONS), self.return_location, dtype=np.float32
            ),
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
        future_dates=tuple(
            f"2026-08-{22 + index:02d}" for index in range(len(VOLATILITY_HORIZONS) * 5)
        ),
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
    readiness = volatility_v2.volatility_release_readiness()
    assert readiness == {
        "configured": True,
        "status": "unavailable",
        "certified_horizons": [],
    }


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
    assert body["historical_dates"] == []
    assert body["historical_prices"] == [88.0]
    quantiles = body["forecast"]["price_quantiles"]
    values = [quantiles[key][-1] for key in QUANTILE_KEYS]
    assert values == sorted(values)
    assert all(value > 0 for value in values)
    assert all(len(quantiles[key]) == 7 for key in QUANTILE_KEYS)
    assert body["forecast"]["future_dates"] == list(_fake_snapshot().future_dates[:7])
    assert quantiles["p05"][0] < body["current_price"] < quantiles["p95"][0]
    assert quantiles["p50"][-1] == pytest.approx(88.78, rel=1e-9)
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
    summary = evidence["horizon_certification"]["7"]
    assert summary["decision"] == "pass"
    assert summary["coverage_80"] == pytest.approx(0.79)


def test_serves_certified_student_t_return_distribution(monkeypatch) -> None:
    class _DistributionRuntime(_FakeRuntime):
        certified_heads = {
            "volatility": True,
            "return_distribution": True,
            "direction": False,
        }
        return_distribution_family = "student_t"
        return_distribution_degrees_of_freedom = 5.0

    runtime = _DistributionRuntime(variance=4e-4, return_location=0.035)
    _install_release(monkeypatch, runtime=runtime)
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        lambda ticker: _fake_snapshot(ticker),
    )
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 7})
    assert response.status_code == 200
    body = response.json()
    quantiles = body["forecast"]["price_quantiles"]
    expected_terminal_median = body["current_price"] * np.exp(0.035)
    assert quantiles["p50"][-1] == pytest.approx(expected_terminal_median, rel=1e-6)
    assert quantiles["p50"][0] == pytest.approx(body["current_price"] * np.exp(0.035 / 7), rel=1e-6)
    assert [quantiles[key][-1] for key in QUANTILE_KEYS] == sorted(
        quantiles[key][-1] for key in QUANTILE_KEYS
    )
    assert body["forecast"]["expected_cumulative_return"] == pytest.approx(0.035)
    assert body["forecast"]["return_distribution_variance"] == pytest.approx(6e-4)
    assert body["forecast"]["return_distribution_family"] == "student_t"
    assert body["forecast"]["probability_up"] is None
    assert body["evidence"]["certified_heads"] == _DistributionRuntime.certified_heads
    assert "terminal location and variance are certified" in body["evidence"]["quantile_model"]


def test_mixed_release_labels_only_the_learned_horizon(monkeypatch) -> None:
    class _MixedRuntime(_FakeRuntime):
        certified_heads = {
            "volatility": True,
            "return_distribution": True,
            "direction": False,
        }
        return_distribution_family = "student_t"
        return_distribution_degrees_of_freedom = 5.0

        def is_return_distribution_horizon(self, horizon: int) -> bool:
            return horizon == 3

        def is_certified_horizon(self, horizon: int) -> bool:
            return horizon in (1, 3)

    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        lambda ticker: _fake_snapshot(ticker),
    )
    _install_release(monkeypatch, runtime=_MixedRuntime(return_location=0.035))
    baseline_response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 1})
    assert baseline_response.status_code == 200
    baseline = baseline_response.json()
    assert baseline["evidence"]["certified_heads"]["return_distribution"] is False
    assert baseline["forecast"]["expected_cumulative_return"] is None
    assert baseline["forecast"]["price_quantiles"]["p50"][-1] == pytest.approx(
        baseline["current_price"]
    )

    learned_response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 3})
    assert learned_response.status_code == 200
    learned = learned_response.json()
    assert learned["evidence"]["certified_heads"]["return_distribution"] is True
    assert learned["forecast"]["expected_cumulative_return"] == pytest.approx(0.035)
    assert learned["forecast"]["price_quantiles"]["p50"][-1] > learned["current_price"]


def test_abstains_on_horizons_that_failed_the_guardrail(monkeypatch) -> None:
    _install_release(monkeypatch, runtime=_FakeRuntime())
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 3})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "abstain_no_certified_model"
    assert "guardrails" in detail["reason"]


def test_news_release_abstains_until_matching_live_provider_is_configured(monkeypatch) -> None:
    runtime = _FakeRuntime()
    runtime.news_status = "certified"
    _install_release(monkeypatch, runtime=runtime)
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 7})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "abstain_no_certified_model"
    assert "live point-in-time news vector" in detail["reason"]
    assert runtime.calls == 0
    readiness = volatility_v2.volatility_release_readiness()
    assert readiness["status"] == "news_input_unavailable"
    assert readiness["certified_horizons"] == []


class _NewsRuntime(_FakeRuntime):
    news_status = "certified"
    news_feature_names = ("News_Ticker_Intensity_1D", "News_Ticker_Missing_1D")

    def __init__(self, variance: float = 4e-4) -> None:
        super().__init__(variance)
        self.news_vectors: list[np.ndarray | None] = []

    def certified_horizon_list(self):
        return (1, 5, 7)

    def forecast(self, snapshot, *, news_features=None):  # noqa: ANN001
        self.news_vectors.append(news_features)
        return super().forecast(snapshot)


def _install_news_provider(monkeypatch, provider) -> None:
    monkeypatch.setattr(settings, "volatility_news_provider_enabled", True)
    monkeypatch.setattr("services.news_aggregator.get_news_provider", lambda: provider)


def test_news_release_serves_the_live_news_vector_when_provider_is_enabled(monkeypatch) -> None:
    from types import SimpleNamespace

    from services.news_aggregator import NewsFeatureVector

    runtime = _NewsRuntime()
    _install_release(monkeypatch, runtime=runtime)
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        lambda ticker: _fake_snapshot(ticker),
    )

    def provider(ticker, *, cutoff_at, feature_names):
        assert ticker == "MSFT"
        assert str(cutoff_at).startswith("2026-08-21 20:00:00")
        assert feature_names == _NewsRuntime.news_feature_names
        return NewsFeatureVector(
            values=np.asarray([1.5, 0.0], dtype=np.float32),
            feature_names=tuple(feature_names),
            cutoff_at="2026-08-21T20:00:00+00:00",
            eligible_article_count=3,
        )

    _install_news_provider(
        monkeypatch,
        SimpleNamespace(features_for=provider),
    )
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 7})
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert evidence["news_enabled"] is True
    assert evidence["news_input"] == {
        "provider_cutoff_utc": "2026-08-21T20:00:00+00:00",
        "eligible_article_count": 3,
        "news_feature_count": 2,
    }
    assert len(runtime.news_vectors) == 1
    assert np.array_equal(runtime.news_vectors[0], np.asarray([1.5, 0.0], dtype=np.float32))
    readiness = volatility_v2.volatility_release_readiness()
    assert readiness["status"] == "ready"
    assert readiness["news_provider_enabled"] is True


def test_news_release_abstains_when_the_live_provider_fails(monkeypatch) -> None:
    from services.news_aggregator import NewsProviderUnavailable

    def broken(ticker, *, cutoff_at, feature_names):
        raise NewsProviderUnavailable("live news ingestion failed: upstream down")

    runtime = _NewsRuntime()
    _install_release(monkeypatch, runtime=runtime)
    _install_news_provider(monkeypatch, SimpleNamespace(features_for=broken))
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        lambda ticker: _fake_snapshot(ticker),
    )
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 7})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "abstain_no_certified_model"
    assert "live point-in-time news vector unavailable" in detail["reason"]
    assert runtime.calls == 0


def test_news_release_abstains_without_a_declared_feature_schema(monkeypatch) -> None:
    class _SchemalessRuntime(_NewsRuntime):
        news_feature_names = ()

    runtime = _SchemalessRuntime()
    _install_release(monkeypatch, runtime=runtime)
    _install_news_provider(
        monkeypatch,
        SimpleNamespace(features_for=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        lambda ticker: _fake_snapshot(ticker),
    )
    response = CLIENT.get("/api/v2/forecast", params={"ticker": "MSFT", "horizon": 7})
    assert response.status_code == 503
    assert "does not declare its news feature schema" in response.json()["detail"]["reason"]


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


def test_release_identity_partitions_response_cache(monkeypatch) -> None:
    first_runtime = _FakeRuntime()
    _install_release(monkeypatch, runtime=first_runtime)

    def counting_snapshot(ticker: str):
        counting_snapshot.calls += 1
        return _fake_snapshot(ticker)

    counting_snapshot.calls = 0
    monkeypatch.setattr(
        "services.volatility_snapshot.build_volatility_inference_snapshot",
        counting_snapshot,
    )
    first = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    assert first.status_code == 200
    assert counting_snapshot.calls == 1

    second_runtime = _FakeRuntime()
    second_runtime.model_id = "global-volatility-tcn-v2"
    volatility_v2._reset_release_state()
    _install_release(monkeypatch, runtime=second_runtime)
    second = CLIENT.get("/api/v2/forecast", params={"ticker": "NMM", "horizon": 7})
    assert second.status_code == 200
    assert second.json()["evidence"]["model_id"] == "global-volatility-tcn-v2"
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
