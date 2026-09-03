from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.collect_live_forecasts import (
    ApiResult,
    _readiness_errors,
    collection_session,
    export_live_ledger,
    run_live_collection,
    run_preflight,
    write_manifest,
)

from services.live_collection import (
    LIVE_COLLECTION_ITEMS_V1,
    LIVE_EXPECTED_RECORD_COUNT,
    LIVE_MODEL_POLICY_V1,
)


class FakeClient:
    def __init__(
        self,
        *,
        session: str = "2026-09-02",
        fail_preview: tuple[str, int] | None = None,
        fail_live: tuple[str, int] | None = None,
        conflict_live: tuple[str, int] | None = None,
        market_cold_until_preview: bool = False,
        ledger_available: bool = True,
    ) -> None:
        self.session = session
        self.fail_preview = fail_preview
        self.fail_live = fail_live
        self.conflict_live = conflict_live
        self.market_cold_until_preview = market_cold_until_preview
        self.ledger_available = ledger_available
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        token: str | None = None,
        attempts: int = 2,
    ) -> ApiResult:
        del attempts
        params = params or {}
        self.calls.append((method, path, params))
        if path == "/ready":
            has_preview = any(
                call_path == "/api/v1/volatility/forecast" for _, call_path, _ in self.calls
            )
            market_available = not self.market_cold_until_preview or has_preview
            ready = self.ledger_available and market_available
            return ApiResult(
                200 if ready else 503,
                {
                    "status": "ready" if ready else "degraded",
                    "dependencies": {
                        "market_data": {
                            "status": "available" if market_available else "unavailable"
                        },
                        "forecast_ledger": {
                            "status": "available" if self.ledger_available else "unavailable",
                            "backend": "postgresql",
                            "durable": True,
                            "required": True,
                        },
                    },
                },
            )
        if path == "/api/v1/volatility/export-ledger":
            return ApiResult(
                200,
                {
                    "entries": [
                        {
                            "forecast_date": self.session,
                            "ticker": "MSFT",
                            "horizon": 5,
                            "model_name": "rolling_mean",
                            "record_source": "live",
                        }
                    ]
                },
            )

        ticker = str(params["ticker"])
        horizon = int(params["horizon"])
        item = (ticker, horizon)
        is_live = method == "POST"
        if not is_live and item == self.fail_preview:
            return ApiResult(503, {"error": "MARKET_DATA_UNAVAILABLE"})
        if is_live and item == self.conflict_live:
            return ApiResult(409, {"error": "FORECAST_LEDGER_CONFLICT"})
        if is_live and item == self.fail_live:
            return ApiResult(503, {"error": "FORECAST_LEDGER_UNAVAILABLE"})
        fingerprint = hashlib.sha256(f"{ticker}:{horizon}:{self.session}".encode()).hexdigest()
        return ApiResult(
            200,
            {
                "ticker": ticker,
                "horizon": horizon,
                "forecast": {"model": LIVE_MODEL_POLICY_V1[horizon]},
                "evidence": {
                    "data_as_of": self.session,
                    "code_commit": "3d1d0c0",
                    "forecast_fingerprint": fingerprint,
                    "ledger_write": "recorded_live" if is_live else "disabled_public_preview",
                },
            },
        )


def _preflight(client: FakeClient) -> dict[str, Any]:
    return run_preflight(
        client,
        expected_session="2026-09-02",
        run_timestamp="2026-09-02T22:30:00+00:00",
        interval_seconds=0,
    )


def test_dry_run_previews_all_60_without_posting() -> None:
    client = FakeClient()
    manifest = _preflight(client)
    assert manifest["batch_status"] == "dry_run_passed"
    assert manifest["succeeded_count"] == LIVE_EXPECTED_RECORD_COUNT
    assert len(manifest["items"]) == LIVE_EXPECTED_RECORD_COUNT
    assert all(method == "GET" for method, _path, _params in client.calls)


def test_preflight_safely_warms_cold_market_but_never_ignores_ledger_failure() -> None:
    cold_client = FakeClient(market_cold_until_preview=True)
    cold_manifest = _preflight(cold_client)
    assert cold_manifest["batch_status"] == "dry_run_passed"
    assert cold_manifest["batch_errors"] == []

    ledger_client = FakeClient(ledger_available=False)
    ledger_manifest = _preflight(ledger_client)
    assert ledger_manifest["batch_status"] == "aborted"
    assert ledger_manifest["abort_reason"] == "initial_readiness_failed"
    assert "ledger_unavailable" in ledger_manifest["batch_errors"]
    assert not any(
        path == "/api/v1/volatility/forecast" for _method, path, _params in ledger_client.calls
    )


def test_failed_or_stale_preflight_produces_zero_live_writes() -> None:
    failed_client = FakeClient(fail_preview=LIVE_COLLECTION_ITEMS_V1[3])
    failed = _preflight(failed_client)
    result = run_live_collection(failed_client, failed, token="secret", interval_seconds=0)
    assert failed["batch_status"] == "aborted"
    assert result["batch_status"] == "aborted"
    assert not any(method == "POST" for method, _path, _params in failed_client.calls)

    stale_client = FakeClient(session="2026-09-01")
    stale = _preflight(stale_client)
    assert stale["batch_status"] == "aborted"
    assert all("data_as_of_mismatch" in item["errors"] for item in stale["items"])
    assert not any(method == "POST" for method, _path, _params in stale_client.calls)


def test_live_partial_and_conflict_accounting_is_exact() -> None:
    failed_item = LIVE_COLLECTION_ITEMS_V1[10]
    client = FakeClient(fail_live=failed_item)
    manifest = run_live_collection(client, _preflight(client), token="secret", interval_seconds=0)
    assert manifest["batch_status"] == "partial"
    assert manifest["succeeded_count"] == 59
    assert manifest["failed_count"] == 1
    assert [item for item in manifest["items"] if item["status"] == "failed"] == [
        {
            "ticker": failed_item[0],
            "horizon": failed_item[1],
            "status": "failed",
            "http_status": 503,
            "forecast_fingerprint": None,
            "data_as_of": None,
            "model": None,
            "errors": ["http_503"],
        }
    ]

    conflict_item = LIVE_COLLECTION_ITEMS_V1[0]
    conflict_client = FakeClient(conflict_live=conflict_item)
    conflict = run_live_collection(
        conflict_client, _preflight(conflict_client), token="secret", interval_seconds=0
    )
    assert conflict["batch_status"] == "partial"
    assert conflict["items"][0]["errors"] == ["immutable_fingerprint_conflict"]


def test_collection_session_skips_weekends_and_accepts_early_close() -> None:
    weekend, weekend_reason = collection_session(
        datetime(2026, 9, 6, 21, 30, tzinfo=UTC), scheduled=True
    )
    assert weekend is None
    assert weekend_reason == "no_completed_nyse_session_on_london_date"

    early_close, reason = collection_session(
        datetime(2026, 11, 27, 22, 30, tzinfo=UTC), scheduled=True
    )
    assert early_close == "2026-11-27"
    assert reason is None


def test_manifest_and_live_only_export_checksums(tmp_path: Path) -> None:
    client = FakeClient()
    manifest_path = write_manifest(_preflight(client), tmp_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["expected_record_count"] == 60

    result = export_live_ledger(client, token="secret", output_dir=tmp_path)
    assert result == {"operation": "export", "status": "complete", "records": 1}
    payload = json.loads((tmp_path / "live-ledger.json").read_text(encoding="utf-8"))
    assert {entry["record_source"] for entry in payload["entries"]} == {"live"}
    checksum_lines = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        digest, filename = line.split("  ")
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest


def _healthy_payload(*, status: str, market: str) -> dict[str, Any]:
    return {
        "status": status,
        "dependencies": {
            "market_data": {"status": market},
            "forecast_ledger": {
                "status": "available",
                "backend": "postgresql",
                "durable": True,
                "required": True,
            },
        },
    }


def test_readiness_rejects_unexpected_http_status_even_with_healthy_body() -> None:
    healthy_ready = _healthy_payload(status="ready", market="available")
    assert _readiness_errors(healthy_ready, allow_market_cold=True, status_code=500) == [
        "unexpected_readiness_status_500"
    ]
    assert _readiness_errors(healthy_ready, allow_market_cold=False, status_code=500) == [
        "unexpected_readiness_status_500"
    ]
    assert _readiness_errors(healthy_ready, allow_market_cold=True, status_code=0) == [
        "unexpected_readiness_status_0"
    ]
    # Initial 200 must be ready; a degraded body on 200 cannot pass.
    degraded_body = _healthy_payload(status="degraded", market="available")
    assert "initial_readiness_not_ready" in _readiness_errors(
        degraded_body, allow_market_cold=True, status_code=200
    )
    # Valid states still pass.
    assert _readiness_errors(healthy_ready, allow_market_cold=True, status_code=200) == []
    assert (
        _readiness_errors(
            _healthy_payload(status="degraded", market="unavailable"),
            allow_market_cold=True,
            status_code=503,
        )
        == []
    )
    assert _readiness_errors(healthy_ready, allow_market_cold=False, status_code=200) == []


def test_preflight_aborts_on_malformed_500_readiness_without_preview_writes() -> None:
    class MalformedReadyClient(FakeClient):
        def request(
            self,
            method: str,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            token: str | None = None,
            attempts: int = 2,
        ) -> ApiResult:
            if path == "/ready":
                self.calls.append((method, path, params or {}))
                return ApiResult(500, _healthy_payload(status="ready", market="available"))
            return super().request(method, path, params=params, token=token, attempts=attempts)

    client = MalformedReadyClient()
    manifest = _preflight(client)
    assert manifest["batch_status"] == "aborted"
    assert manifest["abort_reason"] == "initial_readiness_failed"
    assert manifest["batch_errors"] == ["unexpected_readiness_status_500"]
    assert not any(path == "/api/v1/volatility/forecast" for _method, path, _params in client.calls)
