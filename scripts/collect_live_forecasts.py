"""Secure operational collector for the frozen forward-volatility experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calendars import latest_completed_trading_session  # noqa: E402
from services.live_collection import (  # noqa: E402
    LIVE_COLLECTION_ITEMS_V1,
    LIVE_EXPECTED_RECORD_COUNT,
    LIVE_HORIZONS_V1,
    LIVE_MODEL_POLICY_V1,
    LIVE_START_DATE,
    LIVE_UNIVERSE_V1,
    LIVE_UNIVERSE_VERSION,
)

FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
COLLECTOR_TOKEN_ENV = "FORECAST_COLLECTOR_TOKEN"


@dataclass(frozen=True)
class ApiResult:
    status_code: int
    payload: dict[str, Any]


class CollectorClient:
    """Small injectable HTTP adapter with bounded ambiguous-failure retries."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        token: str | None = None,
        attempts: int = 2,
    ) -> ApiResult:
        clean_token = token.strip().strip("\"'") if token else None
        headers = {"Authorization": f"Bearer {clean_token}"} if clean_token else {}
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.request(method, path, params=params, headers=headers)
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"error": "INVALID_JSON_RESPONSE"}
                if response.status_code < 500 or attempt + 1 >= attempts:
                    return ApiResult(response.status_code, payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
            time.sleep(1.0)
        return ApiResult(
            0,
            {
                "error": "AMBIGUOUS_TRANSPORT_FAILURE",
                "detail": type(last_error).__name__ if last_error else "server_error",
            },
        )


def collection_session(
    now: datetime,
    *,
    scheduled: bool,
) -> tuple[str | None, str | None]:
    """Resolve the eligible same-day completed NYSE session or a skip reason."""
    instant = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    london_now = instant.astimezone(ZoneInfo("Europe/London"))
    if scheduled and not (london_now.hour == 22 and london_now.minute >= 20):
        return None, "outside_22_30_europe_london_window"
    latest = latest_completed_trading_session(instant).date()
    if latest != london_now.date():
        return None, "no_completed_nyse_session_on_london_date"
    if latest < LIVE_START_DATE:
        return None, "before_live_start_date"
    return latest.isoformat(), None


def _base_manifest(run_timestamp: str, expected_session: str | None) -> dict[str, Any]:
    return {
        "run_timestamp": run_timestamp,
        "expected_nyse_session": expected_session,
        "universe_version": LIVE_UNIVERSE_VERSION,
        "live_start_date": LIVE_START_DATE.isoformat(),
        "tickers": list(LIVE_UNIVERSE_V1),
        "ticker_count": len(LIVE_UNIVERSE_V1),
        "horizons": list(LIVE_HORIZONS_V1),
        "expected_record_count": LIVE_EXPECTED_RECORD_COUNT,
        "succeeded_count": 0,
        "failed_count": 0,
        "items": [],
        "deployed_code_commit": None,
        "batch_status": "aborted",
    }


def _validate_preview(
    payload: dict[str, Any],
    ticker: str,
    horizon: int,
    expected_session: str,
    *,
    expected_ledger_write: str = "disabled_public_preview",
) -> list[str]:
    errors: list[str] = []
    forecast = payload.get("forecast") if isinstance(payload.get("forecast"), dict) else {}
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    if payload.get("ticker") != ticker:
        errors.append("ticker_mismatch")
    if payload.get("horizon") != horizon:
        errors.append("horizon_mismatch")
    if evidence.get("data_as_of") != expected_session:
        errors.append("data_as_of_mismatch")
    if forecast.get("model") != LIVE_MODEL_POLICY_V1[horizon]:
        errors.append("model_policy_mismatch")
    if evidence.get("ledger_write") != expected_ledger_write:
        errors.append("ledger_write_status_mismatch")
    fingerprint = str(evidence.get("forecast_fingerprint", ""))
    if not FINGERPRINT_RE.fullmatch(fingerprint):
        errors.append("invalid_forecast_fingerprint")
    if not str(evidence.get("code_commit", "")).strip():
        errors.append("missing_code_commit")
    return errors


def _readiness_errors(
    payload: dict[str, Any],
    *,
    allow_market_cold: bool,
    status_code: int | None = None,
) -> list[str]:
    errors: list[str] = []
    if status_code is not None:
        if allow_market_cold:
            if status_code not in (200, 503):
                return [f"unexpected_readiness_status_{status_code}"]
            if status_code == 200 and payload.get("status") != "ready":
                errors.append("initial_readiness_not_ready")
            if status_code == 503 and payload.get("status") != "degraded":
                errors.append("unexpected_readiness_status_503")
        elif status_code != 200:
            errors.append(f"unexpected_readiness_status_{status_code}")
    dependencies = (
        payload.get("dependencies") if isinstance(payload.get("dependencies"), dict) else {}
    )
    market = (
        dependencies.get("market_data") if isinstance(dependencies.get("market_data"), dict) else {}
    )
    ledger = (
        dependencies.get("forecast_ledger")
        if isinstance(dependencies.get("forecast_ledger"), dict)
        else {}
    )
    if ledger.get("status") != "available":
        errors.append("ledger_unavailable")
    if ledger.get("backend") != "postgresql":
        errors.append("ledger_not_postgresql")
    if ledger.get("durable") is not True:
        errors.append("ledger_not_durable")
    if ledger.get("required") is not True:
        errors.append("ledger_not_required")
    market_status = market.get("status")
    if market_status != "available" and not (allow_market_cold and market_status == "unavailable"):
        errors.append("market_data_unavailable")
    return errors


def run_preflight(
    client: CollectorClient,
    *,
    expected_session: str,
    run_timestamp: str,
    interval_seconds: float,
) -> dict[str, Any]:
    """Preview all 60 combinations and write nothing."""
    initial_ready = client.request("GET", "/ready", attempts=1)
    manifest = _base_manifest(run_timestamp, expected_session)
    initial_readiness_errors = _readiness_errors(
        initial_ready.payload, allow_market_cold=True, status_code=initial_ready.status_code
    )
    if initial_readiness_errors:
        manifest["abort_reason"] = "initial_readiness_failed"
        manifest["batch_errors"] = initial_readiness_errors
        return manifest

    commits: set[str] = set()
    for index, (ticker, horizon) in enumerate(LIVE_COLLECTION_ITEMS_V1):
        response = client.request(
            "GET",
            "/api/v1/volatility/forecast",
            params={"ticker": ticker, "horizon": horizon, "model": "auto"},
        )
        evidence = response.payload.get("evidence", {})
        errors = (
            _validate_preview(response.payload, ticker, horizon, expected_session)
            if response.status_code == 200
            else [f"http_{response.status_code}"]
        )
        fingerprint = evidence.get("forecast_fingerprint") if not errors else None
        commit = evidence.get("code_commit")
        if commit:
            commits.add(str(commit))
        manifest["items"].append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "status": "preview_passed" if not errors else "preview_failed",
                "http_status": response.status_code,
                "data_as_of": evidence.get("data_as_of"),
                "model": response.payload.get("forecast", {}).get("model"),
                "forecast_fingerprint": fingerprint,
                "errors": errors,
            }
        )
        status_label = "preview_passed" if not errors else f"preview_failed: {','.join(errors)}"
        print(
            f"[PREFLIGHT {index + 1}/{LIVE_EXPECTED_RECORD_COUNT}] {ticker} h={horizon} -> HTTP {response.status_code} ({status_label})",
            flush=True,
        )
        if interval_seconds and index + 1 < LIVE_EXPECTED_RECORD_COUNT:
            time.sleep(interval_seconds)

    final_ready = client.request("GET", "/ready", attempts=1)
    final_readiness_errors = _readiness_errors(
        final_ready.payload, allow_market_cold=False, status_code=final_ready.status_code
    )
    if final_ready.status_code != 200 or final_ready.payload.get("status") != "ready":
        final_readiness_errors.append("final_readiness_not_ready")

    failures = sum(bool(item["errors"]) for item in manifest["items"])
    manifest["succeeded_count"] = LIVE_EXPECTED_RECORD_COUNT - failures
    manifest["failed_count"] = failures
    manifest["batch_errors"] = list(dict.fromkeys(final_readiness_errors))
    if final_readiness_errors:
        manifest["abort_reason"] = "final_readiness_failed"
        failures += 1
    if len(commits) == 1:
        manifest["deployed_code_commit"] = next(iter(commits))
    elif commits:
        manifest["abort_reason"] = "mixed_deployed_commits"
        manifest["batch_errors"].append("mixed_deployed_commits")
        failures += 1
    if failures:
        manifest.setdefault("abort_reason", "preview_validation_failed")
        return manifest
    manifest["batch_status"] = "dry_run_passed"
    return manifest


def run_live_collection(
    client: CollectorClient,
    preflight: dict[str, Any],
    *,
    token: str,
    interval_seconds: float,
) -> dict[str, Any]:
    """Record the frozen batch only after a complete validated preflight."""
    manifest = json.loads(json.dumps(preflight))
    if preflight.get("batch_status") != "dry_run_passed":
        manifest["batch_status"] = "aborted"
        manifest["abort_reason"] = "preflight_not_complete"
        return manifest

    preview_by_item = {
        (item["ticker"], item["horizon"]): item for item in preflight.get("items", [])
    }
    live_items: list[dict[str, Any]] = []
    for index, (ticker, horizon) in enumerate(LIVE_COLLECTION_ITEMS_V1):
        response = client.request(
            "POST",
            "/api/v1/volatility/collect",
            params={"ticker": ticker, "horizon": horizon},
            token=token,
        )
        evidence = response.payload.get("evidence", {})
        returned_fingerprint = evidence.get("forecast_fingerprint")
        expected_fingerprint = preview_by_item[(ticker, horizon)].get("forecast_fingerprint")
        errors = (
            _validate_preview(
                response.payload,
                ticker,
                horizon,
                preflight["expected_nyse_session"],
                expected_ledger_write="recorded_live",
            )
            if response.status_code == 200
            else []
        )
        if response.status_code == 409:
            errors.append("immutable_fingerprint_conflict")
        elif response.status_code != 200:
            detail = response.payload.get("detail")
            errors.append(
                f"http_{response.status_code}_{detail}"
                if detail
                else f"http_{response.status_code}"
            )
        elif returned_fingerprint != expected_fingerprint:
            errors.append("preflight_fingerprint_changed")
        elif evidence.get("ledger_write") != "recorded_live":
            errors.append("ledger_write_not_confirmed")
        live_items.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "status": "recorded" if not errors else "failed",
                "http_status": response.status_code,
                "forecast_fingerprint": returned_fingerprint,
                "data_as_of": evidence.get("data_as_of"),
                "model": response.payload.get("forecast", {}).get("model"),
                "errors": errors,
            }
        )
        status_label = "recorded" if not errors else f"failed: {','.join(errors)}"
        print(
            f"[LIVE WRITE {index + 1}/{LIVE_EXPECTED_RECORD_COUNT}] {ticker} h={horizon} -> HTTP {response.status_code} ({status_label})",
            flush=True,
        )
        if interval_seconds and index + 1 < LIVE_EXPECTED_RECORD_COUNT:
            time.sleep(interval_seconds)

    succeeded = sum(item["status"] == "recorded" for item in live_items)
    manifest["items"] = live_items
    manifest["succeeded_count"] = succeeded
    manifest["failed_count"] = LIVE_EXPECTED_RECORD_COUNT - succeeded
    manifest["batch_status"] = "complete" if succeeded == LIVE_EXPECTED_RECORD_COUNT else "partial"
    return manifest


def run_settlement(client: CollectorClient, *, token: str) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for ticker in LIVE_UNIVERSE_V1:
        response = client.request(
            "POST",
            "/api/v1/volatility/score-ledger",
            params={"ticker": ticker},
            token=token,
        )
        items.append(
            {
                "ticker": ticker,
                "http_status": response.status_code,
                "scored_count": response.payload.get("scored_count"),
                "status": "succeeded" if response.status_code == 200 else "failed",
            }
        )
    return {
        "operation": "settlement",
        "run_timestamp": datetime.now(UTC).isoformat(),
        "universe_version": LIVE_UNIVERSE_VERSION,
        "items": items,
        "status": "complete" if all(item["status"] == "succeeded" for item in items) else "partial",
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = manifest.get("expected_nyse_session") or "no-session"
    status = manifest.get("batch_status", manifest.get("status", "unknown"))
    run_timestamp = str(manifest.get("run_timestamp", datetime.now(UTC).isoformat()))
    timestamp_slug = re.sub(r"[^0-9]", "", run_timestamp)[:14]
    output = output_dir / f"live-collection-{session}-{timestamp_slug}-{status}.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def export_live_ledger(client: CollectorClient, *, token: str, output_dir: Path) -> dict[str, Any]:
    response = client.request("GET", "/api/v1/volatility/export-ledger", token=token, attempts=2)
    if response.status_code != 200:
        return {"operation": "export", "status": "failed", "http_status": response.status_code}
    entries = response.payload.get("entries")
    if not isinstance(entries, list):
        return {"operation": "export", "status": "failed", "error": "invalid_entries"}
    entries = sorted(
        entries,
        key=lambda item: (
            item.get("forecast_date", ""),
            item.get("ticker", ""),
            int(item.get("horizon", 0)),
            item.get("model_name", ""),
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "live-ledger.json"
    csv_path = output_dir / "live-ledger.csv"
    sums_path = output_dir / "SHA256SUMS"
    json_payload = {"record_source": "live", "entries": entries}
    json_path.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for entry in entries for key in entry})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(entries)
    checksums = []
    for path in (json_path, csv_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.name}")
    sums_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return {"operation": "export", "status": "complete", "records": len(entries)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--mode", choices=("dry-run", "live", "settle", "export"), default="dry-run"
    )
    parser.add_argument("--scheduled", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=2.3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/live-collection"))
    parser.add_argument("--expected-session", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(UTC)
    run_timestamp = now.isoformat()
    expected_session = args.expected_session
    skip_reason = None
    if args.mode in {"dry-run", "live"} and expected_session is None:
        expected_session, skip_reason = collection_session(now, scheduled=args.scheduled)
    if args.mode in {"dry-run", "live"} and expected_session is None:
        manifest = _base_manifest(run_timestamp, None)
        manifest.update({"batch_status": "aborted", "abort_reason": skip_reason})
        output = write_manifest(manifest, args.output_dir)
        print(json.dumps({"manifest": str(output), "status": "aborted", "reason": skip_reason}))
        return 0 if args.scheduled else 1

    client = CollectorClient(args.base_url, timeout_seconds=args.timeout_seconds)
    try:
        if args.mode in {"dry-run", "live"}:
            preflight = run_preflight(
                client,
                expected_session=str(expected_session),
                run_timestamp=run_timestamp,
                interval_seconds=args.interval_seconds,
            )
            result = preflight
            if args.mode == "live":
                token = os.getenv(COLLECTOR_TOKEN_ENV, "")
                if not token:
                    result = dict(preflight)
                    result.update(
                        {"batch_status": "aborted", "abort_reason": "collector_token_missing"}
                    )
                else:
                    result = run_live_collection(
                        client,
                        preflight,
                        token=token,
                        interval_seconds=args.interval_seconds,
                    )
            output = write_manifest(result, args.output_dir)
            print(json.dumps({"manifest": str(output), "status": result["batch_status"]}))
            expected_status = "dry_run_passed" if args.mode == "dry-run" else "complete"
            return 0 if result["batch_status"] == expected_status else 1

        token = os.getenv(COLLECTOR_TOKEN_ENV, "")
        if not token:
            print(json.dumps({"status": "aborted", "reason": "collector_token_missing"}))
            return 1
        if args.mode == "settle":
            result = run_settlement(client, token=token)
            output = write_manifest(result, args.output_dir)
            print(json.dumps({"manifest": str(output), "status": result["status"]}))
            return 0 if result["status"] == "complete" else 1
        result = export_live_ledger(client, token=token, output_dir=args.output_dir)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "complete" else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
