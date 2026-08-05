import hashlib
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from api import app
from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier
from server_models.compatibility import CompatibilityReport
from server_models.contracts import (
    FORECAST_LENGTH,
    HISTORY_DISPLAY_WINDOW,
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
)
from server_models.response_models import PriceForecastResponse

CLIENT = TestClient(app)


def make_record(ticker="AAPL", forecast_type="price", snapshot_id="snap-1"):
    key = ServerArtifactKey.create(
        ticker=ticker, snapshot_id=snapshot_id, forecast_type=forecast_type
    )
    repro = ReproducibilityMetadata(
        feature_names=["f1"] * 28,
        scaler=RobustScalerParams(medians=[0.0] * 28, iqrs=[1.0] * 28),
        python_version="3.12",
        git_commit="unknown",
    )
    return ServerModelRecord(key=key, reproducibility=repro, sha256_digest="0" * 64)


ORIGIN_DATE = date(2026, 7, 31)
ORIGIN_CLOSE = 150.0


def _bundle(ticker="AAPL", version_id="unused", forecast_type="price", generated_at=None):
    return ServerForecastBundle(
        version_id=version_id,
        ticker=ticker,
        forecast_type=forecast_type,
        generated_at=generated_at or datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC),
        origin_close=ORIGIN_CLOSE,
        origin_date=ORIGIN_DATE,
        evidence={
            "metric_source": "server_purged_walk_forward",
            "family": "elastic_net",
            "pooled": {"relative_rmse": 0.9},
        },
        future_dates=[ORIGIN_DATE + timedelta(days=1 + i) for i in range(FORECAST_LENGTH)],
        predicted_prices=[100.0 * 1.001**i for i in range(FORECAST_LENGTH)],
        predicted_log_returns=[0.001] * FORECAST_LENGTH,
        historical_dates=[
            ORIGIN_DATE - timedelta(days=HISTORY_DISPLAY_WINDOW - 1 - i)
            for i in range(HISTORY_DISPLAY_WINDOW)
        ],
        historical_prices=[
            ORIGIN_CLOSE - (HISTORY_DISPLAY_WINDOW - 1 - i) * 0.25
            for i in range(HISTORY_DISPLAY_WINDOW)
        ],
    )


def make_bundle(record, *, signer=None):
    """Return (bundle, digest, payload, record-with-matched-digest)."""
    bundle = _bundle(
        ticker=record.key.ticker,
        version_id=record.key.version_id,
        forecast_type=record.key.forecast_type,
        generated_at=record.key.trained_at,
    )
    payload = bundle.model_dump_json().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    record = record.model_copy(update={"sha256_digest": digest})
    if signer is not None:
        record = record.model_copy(update={"signature": signer(payload)})
    return bundle, digest, payload, record


def _enable(
    monkeypatch,
    *,
    enabled=True,
    mode="hybrid",
    allowlist=("AAPL",),
    configured=True,
):
    import config

    settings = config.settings
    monkeypatch.setattr(settings, "server_forecast_serving_enabled", enabled)
    monkeypatch.setattr(settings, "training_mode", mode)
    monkeypatch.setattr(settings, "server_forecast_allowlist", list(allowlist))
    monkeypatch.setattr(settings, "server_forecast_public_key_path", None)
    monkeypatch.setattr(
        settings, "registry_database_url", "postgresql://fake" if configured else None
    )
    monkeypatch.setattr(settings, "s3_bucket", "fake-bucket" if configured else None)


def _stub_serve_deps(monkeypatch, *, registry=None, storage=None, verifier=None):
    monkeypatch.setattr(
        "server_models.api.get_registry",
        (lambda: registry) if registry is not None else (lambda: MagicMock()),
    )
    monkeypatch.setattr(
        "server_models.api.get_storage",
        (lambda: storage) if storage is not None else (lambda: MagicMock()),
    )
    if verifier is not None:
        monkeypatch.setattr("server_models.api.load_verifier", lambda: (verifier, None))
    else:
        monkeypatch.setattr("server_models.api.load_verifier", lambda: (MagicMock(), None))


class _BrokenRegistry:
    def get_promoted(self, *args, **kwargs):
        raise RuntimeError("postgres down")

    def close(self):
        pass


def _compat_ok(monkeypatch):
    monkeypatch.setattr(
        "server_models.api.check_record_compatibility",
        lambda r: CompatibilityReport(compatible=True, reason="ok"),
    )


def _happy(monkeypatch, record, signer=None):
    _bundle, _digest, payload, final_record = make_bundle(record, signer=signer)
    registry = MagicMock()
    registry.get_promoted.return_value = final_record
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(monkeypatch, registry=registry, storage=storage)
    _compat_ok(monkeypatch)
    return registry, final_record


@pytest.fixture(autouse=True)
def _clear_server_caches(monkeypatch):
    from server_models import api as server_api

    server_api._availability_cache.clear()
    server_api._bundle_cache.clear()
    _enable(monkeypatch)
    yield
    server_api._availability_cache.clear()
    server_api._bundle_cache.clear()


def test_openapi_includes_server_paths_unconditionally():
    response = CLIENT.get("/openapi.json")
    paths = response.json()["paths"]
    assert "/api/v1/server-forecasts/availability" in paths
    assert "/api/v1/server-forecasts/{ticker}" in paths


# ── availability readiness ───────────────────────────────────────────


def test_availability_reports_disabled(monkeypatch):
    _enable(monkeypatch, enabled=False)
    response = CLIENT.get("/api/v1/server-forecasts/availability")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["mode"] == "browser_only"
    assert data["configured"] is False
    assert data["reason"] == "disabled"
    assert data["tickers"] == []


def test_availability_unconfigured_when_infrastructure_missing(monkeypatch):
    _enable(monkeypatch, configured=False)
    response = CLIENT.get("/api/v1/server-forecasts/availability")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["configured"] is False
    assert data["reason"] == "unconfigured"
    assert data["tickers"] == []


def test_availability_integrity_failure_when_key_broken(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr("server_models.api.load_verifier", lambda: (None, "integrity_failure"))
    response = CLIENT.get("/api/v1/server-forecasts/availability")
    assert response.status_code == 200
    data = response.json()
    assert data["configured"] is False
    assert data["reason"] == "integrity_failure"


def test_availability_reports_fresh_when_promoted(monkeypatch):
    record = make_record()
    registry = MagicMock()
    registry.get_promoted.return_value = record
    _stub_serve_deps(monkeypatch, registry=registry)

    response = CLIENT.get("/api/v1/server-forecasts/availability")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["configured"] is True
    assert data["mode"] == "hybrid"
    assert len(data["tickers"]) == 1
    assert data["tickers"][0]["ticker"] == "AAPL"
    assert data["tickers"][0]["status"] == "fresh"
    assert data["tickers"][0]["version_id"] == record.key.version_id


def test_availability_uses_one_registry_and_closes_it(monkeypatch):
    record = make_record()
    registry = MagicMock()
    registry.get_promoted.return_value = record
    _stub_serve_deps(monkeypatch, registry=registry)

    CLIENT.get("/api/v1/server-forecasts/availability")
    assert registry.get_promoted.call_count == 1
    registry.close.assert_called_once()


# ── soft absences ────────────────────────────────────────────────────


def test_forecast_disabled_fallback(monkeypatch):
    _enable(monkeypatch, enabled=False)
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "disabled"
    assert response.json()["fallback"] == "browser_training"


def test_forecast_not_allowlisted_fallback(monkeypatch):
    _enable(monkeypatch, allowlist=("MSFT",))
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "not_allowlisted"


def test_forecast_unconfigured_fallback_in_hybrid(monkeypatch):
    _enable(monkeypatch, configured=False)
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "unconfigured"
    assert response.json()["fallback"] == "browser_training"


def test_forecast_unconfigured_is_503_in_server_pretrained_mode(monkeypatch):
    _enable(monkeypatch, configured=False, mode="server_pretrained")
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "unconfigured"
    assert detail["fallback"] is None


def test_forecast_integrity_failure_is_503_in_all_modes(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr("server_models.api.load_verifier", lambda: (None, "integrity_failure"))
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "integrity_failure"
    assert response.json()["detail"]["fallback"] is None


def test_forecast_direction_is_unsupported_and_browser_falls_back(monkeypatch):
    _enable(monkeypatch)
    _stub_serve_deps(monkeypatch)
    response = CLIENT.get("/api/v1/server-forecasts/AAPL?forecast_type=trend")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["reason"] == "unsupported_forecast_type"
    assert "predicted_prices" not in data


# ── expected absences: 200 in browser modes, 503 in server_pretrained ─


def test_forecast_missing_fallback_in_hybrid(monkeypatch):
    _enable(monkeypatch, mode="hybrid")
    registry = MagicMock()
    registry.get_promoted.return_value = None
    _stub_serve_deps(monkeypatch, registry=registry)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "missing"
    assert response.json()["fallback"] == "browser_training"


def test_forecast_stale_fallback_in_hybrid(monkeypatch):
    record = make_record()
    registry = MagicMock()
    registry.get_promoted.return_value = record
    _stub_serve_deps(monkeypatch, registry=registry)
    monkeypatch.setattr("server_models.api.is_fresh", lambda *a, **k: False)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "stale"


def test_forecast_incompatible_fallback_in_hybrid(monkeypatch):
    record = make_record()
    registry = MagicMock()
    registry.get_promoted.return_value = record
    _stub_serve_deps(monkeypatch, registry=registry)
    monkeypatch.setattr(
        "server_models.api.check_record_compatibility",
        lambda r: CompatibilityReport(compatible=False, reason="flagged"),
    )

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "incompatible"


def test_forecast_missing_is_503_in_server_pretrained_mode(monkeypatch):
    _enable(monkeypatch, mode="server_pretrained")
    registry = MagicMock()
    registry.get_promoted.return_value = None
    _stub_serve_deps(monkeypatch, registry=registry)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "missing"
    assert response.json()["detail"]["fallback"] is None


# ── infrastructure failures: always 503 ──────────────────────────────


def test_forecast_registry_unavailable_fail_closed(monkeypatch):
    _enable(monkeypatch, mode="hybrid")
    _stub_serve_deps(monkeypatch, registry=_BrokenRegistry())

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "registry_unavailable"
    assert detail["fallback"] == "browser_training"


def test_forecast_registry_unavailable_has_no_fallback_in_server_pretrained(monkeypatch):
    _enable(monkeypatch, mode="server_pretrained")
    _stub_serve_deps(monkeypatch, registry=_BrokenRegistry())

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "registry_unavailable"
    assert detail["fallback"] is None


def test_forecast_success_closes_registry(monkeypatch):
    registry = MagicMock()
    registry.get_promoted.return_value = make_record()
    storage = MagicMock()
    storage.get_bundle.return_value = b"{}"
    _stub_serve_deps(monkeypatch, registry=registry, storage=storage)
    _compat_ok(monkeypatch)
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503  # bundle does not parse; infra failure path
    registry.close.assert_called_once()


# ── successful serving ───────────────────────────────────────────────


def test_forecast_success_returns_canonical_bundle(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private_key)
    record = make_record()
    _bundle, _digest, payload, final_record = make_bundle(record, signer=signer)
    registry = MagicMock()
    registry.get_promoted.return_value = final_record
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(monkeypatch, registry=registry, storage=storage)
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["ticker"] == "AAPL"
    assert data["forecast_days"] == 7
    assert len(data["predicted_prices"]) == 7
    assert len(data["future_dates"]) == 7
    assert len(data["historical_prices"]) == HISTORY_DISPLAY_WINDOW
    assert len(data["historical_dates"]) == HISTORY_DISPLAY_WINDOW
    assert data["metadata"]["engine"]["role"] == "server_pretrained"
    assert data["metadata"]["engine"]["version_id"] == final_record.key.version_id
    assert data["metadata"]["authenticity"] == "ed25519_verified"
    assert data["metadata"]["origin"]["date"] == "2026-07-31"
    assert data["metadata"]["origin"]["close"] == 150.0
    assert data["metrics"]["metric_source"] == "server_purged_walk_forward"

    # The response genuinely validates against the shared forecast contract.
    validated = PriceForecastResponse.model_validate(data)
    assert validated.ticker == "AAPL"
    assert validated.metadata.execution.mode == "artifact_loaded"
    assert validated.metadata.execution.coalesced is False
    assert validated.metadata.artifact_state_before == "fresh"
    assert validated.metadata.artifact_action == "loaded"
    triage = validated.metadata.timings_seconds
    assert triage.artifact_load_validation is not None
    assert triage.total is not None and triage.total >= 0
    assert triage.queue_wait is None and triage.training is None and triage.inference is None
    assert response.headers["etag"] == final_record.key.version_id


def test_forecast_success_verifies_ed25519_when_key_configured(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private_key)
    record = make_record()
    registry, final_record = _happy(monkeypatch, record, signer=signer)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 200
    assert response.json()["metadata"]["authenticity"] == "ed25519_verified"


def test_forecast_tampered_bundle_fails_closed(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private_key)
    record = make_record()
    _bundle, _digest, payload, final_record = make_bundle(record, signer=signer)
    registry = MagicMock()
    registry.get_promoted.return_value = final_record
    storage = MagicMock()
    tampered = bytearray(payload)
    tampered[0] ^= 0xFF
    storage.get_bundle.return_value = bytes(tampered)
    _stub_serve_deps(
        monkeypatch,
        registry=registry,
        storage=storage,
        verifier=Ed25519ManifestVerifier(private_key.public_key()),
    )
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "signature_verification_failed"


def test_forecast_digest_mismatch_fails_closed(monkeypatch):
    record = make_record()
    _bundle, _digest, payload, _final_record = make_bundle(record)
    registry = MagicMock()
    registry.get_promoted.return_value = record.model_copy(update={"sha256_digest": "f" * 64})
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(monkeypatch, registry=registry, storage=storage)
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "signature_verification_failed"


def test_forecast_signed_bundle_with_foreign_ticker_fails_closed(monkeypatch):
    """A correctly signed bundle whose embedded identity differs from the
    promoted registry row must never be served for another ticker."""
    private_key = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private_key)
    record = make_record("AAPL")
    bundle = _bundle(
        ticker="MSFT",
        version_id=record.key.version_id,
        generated_at=record.key.trained_at,
    )
    payload = bundle.model_dump_json().encode("utf-8")
    foreign_record = record.model_copy(
        update={
            "sha256_digest": hashlib.sha256(payload).hexdigest(),
            "signature": signer(payload),
        }
    )
    registry = MagicMock()
    registry.get_promoted.return_value = foreign_record
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(
        monkeypatch,
        registry=registry,
        storage=storage,
        verifier=Ed25519ManifestVerifier(private_key.public_key()),
    )
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "identity_mismatch"


def test_forecast_signed_bundle_with_mismatched_version_fails_closed(monkeypatch):
    """The bundle's version_id must equal the promoted record's version_id."""
    private_key = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private_key)
    record = make_record("AAPL")
    bundle = _bundle(
        ticker="AAPL",
        version_id=f"{record.key.version_id}-bogus",
        generated_at=record.key.trained_at,
    )
    payload = bundle.model_dump_json().encode("utf-8")
    registry = MagicMock()
    registry.get_promoted.return_value = record.model_copy(
        update={
            "sha256_digest": hashlib.sha256(payload).hexdigest(),
            "signature": signer(payload),
        }
    )
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(
        monkeypatch,
        registry=registry,
        storage=storage,
        verifier=Ed25519ManifestVerifier(private_key.public_key()),
    )
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "identity_mismatch"
