"""Reusable Render/Vercel deployment smoke checker."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EXPECTED_FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "SMA_20", "EMA_20", "RSI_14",
    "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "ATR_14", "OBV",
    "SPY_Return_1D", "QQQ_Return_1D", "VIX_Return_1D", "TNX_Return_1D",
    "Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def safe_error(exc: BaseException) -> str:
    return str(exc).replace("\n", " ").strip()[:240] or type(exc).__name__


def get_json(base_url: str, path: str, *, timeout: float, origin: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    headers = {"Accept": "application/json"}
    if origin:
        headers["Origin"] = origin
    with urlopen(Request(f"{base_url}{path}", headers=headers), timeout=timeout) as response:
        return json.load(response), {key.lower(): value for key, value in response.headers.items()}


def add(results: list[CheckResult], name: str, passed: bool, detail: str) -> None:
    results.append(CheckResult(name=name, passed=passed, detail="" if passed else detail))


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    ticker = args.ticker.strip().upper()
    started = time.monotonic()
    results: list[CheckResult] = []
    observed_commit = None
    observed_environment = None
    try:
        first_health, health_headers = get_json(base_url, "/health", timeout=args.timeout, origin=args.cors_origin)
        add(results, "health_status", first_health.get("status") == "ok", "/health did not return status=ok")
        deployment = first_health.get("deployment") or {}
        observed_commit = deployment.get("commit")
        observed_environment = deployment.get("environment")
        if args.expected_commit:
            add(results, "expected_commit", observed_commit == args.expected_commit[:12].lower(), "deployment commit did not match expected commit")
        if args.expected_environment:
            add(results, "expected_environment", observed_environment == args.expected_environment, "deployment environment did not match expected environment")
        if args.cors_origin:
            add(results, "cors_origin", health_headers.get("access-control-allow-origin") == args.cors_origin, "CORS did not echo the expected origin")
        if args.restart_window > 0:
            time.sleep(args.restart_window)
            second_health, _ = get_json(base_url, "/health", timeout=args.timeout)
            add(results, "restart_detection", second_health.get("deployment") == first_health.get("deployment"), "health identity changed during smoke window")

        models, _ = get_json(base_url, "/models", timeout=args.timeout)
        add(results, "server_models_disabled", models.get("server_models", {}).get("status") == "disabled", "/models did not disable server-side model storage")
        browser = models.get("browser_training", {})
        add(results, "browser_training_contract", browser.get("status") == "available" and browser.get("storage") == "indexeddb", "/models did not advertise browser IndexedDB training")

        snapshot, _ = get_json(base_url, f"/api/v1/training-data?ticker={ticker}", timeout=args.timeout)
        rows = snapshot.get("features") or []
        add(results, "schema_version", snapshot.get("schema_version") == 3, "training-data schema version is not 3")
        add(results, "feature_order", snapshot.get("feature_names") == EXPECTED_FEATURES, "training-data feature ordering is incompatible")
        add(results, "snapshot_shape", snapshot.get("window_size") == 60 and snapshot.get("output_width") == 30 and len(rows) == len(snapshot.get("dates", [])), "training-data shape/date contract is incompatible")
        add(results, "finite_features", bool(rows) and all(math.isfinite(float(value)) for row in rows for value in row), "training-data returned no finite feature rows")

        price, _ = get_json(base_url, f"/api/v1/predict?ticker={ticker}&days=1", timeout=args.timeout)
        price_engine = price.get("metadata", {}).get("engine", {})
        price_mode = price.get("metadata", {}).get("execution", {}).get("mode")
        add(results, "price_baseline_label", bool(price_engine.get("baseline_fallback")) and price_engine.get("role") == "server_disabled_fallback" and price_mode == "baseline_fallback", "price forecast was not explicitly labelled baseline fallback")

        direction, _ = get_json(base_url, f"/api/v1/predict/direction?ticker={ticker}&days=1", timeout=args.timeout)
        direction_engine = direction.get("metadata", {}).get("engine", {})
        add(results, "direction_baseline_label", bool(direction_engine.get("baseline_fallback")) and direction_engine.get("role") == "server_disabled_fallback", "direction forecast was not explicitly labelled baseline fallback")
    except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
        add(results, "request", False, safe_error(exc))
    passed = all(item.passed for item in results)
    return {
        "status": "passed" if passed else "failed",
        "base_url": base_url,
        "ticker": ticker,
        "duration_seconds": round(time.monotonic() - started, 3),
        "deployment": {"commit": observed_commit, "environment": observed_environment},
        "checks": [item.__dict__ for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://stock-predictor-lstm.onrender.com")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-environment")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--restart-window", type=float, default=0)
    parser.add_argument("--cors-origin")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif result["status"] == "passed":
        print(f"deployment smoke test passed for {result['base_url']}")
    else:
        for check in result["checks"]:
            if not check["passed"]:
                print(f"deployment smoke test failed: {check['name']}: {check['detail']}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
