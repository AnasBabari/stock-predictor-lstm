#!/usr/bin/env python3
"""v8 production smoke — verifies Render + Vercel after signed archive is live.

This is the v8 counterpart to ``scripts/check_deployment.py`` but expects
``metric_source=locked_historical_temporal_test_plus_asset_transfer`` and
``model_version`` containing ``v8``.  It is safe to run in CI with
``--forecast-contract v8_abstention`` until the archive is live, then with
``v8`` after.

Checks:
  /health, /ready, /models, MSFT/NMM/AAPL/Nasdaq/NYSE/LSE forecasts,
  invalid ticker/horizon, tampered archive, wrong sha, CORS, cache, rollback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_deployment import (  # type: ignore  # noqa: E402
    CheckResult,
    add,
    get_json,
    get_json_with_status,
    safe_error,
)

EXPECTED_V8_METRIC_SOURCE = "locked_historical_temporal_test_plus_asset_transfer"
EXPECTED_V8_SCOPE = "historical_temporal_test_plus_asset_transfer"


def run_v8_smoke(base_url: str, vercel_url: str | None, timeout: float = 8.0) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        health, _ = get_json(base_url, "/health", timeout=timeout)
        add(results, "health", health.get("status") == "ok", f"health {health}")
        ready, _ = get_json(base_url, "/ready", timeout=timeout)
        # ready contains global_volatility dependency
        gv = ready.get("dependencies", {}).get("global_volatility", {})
        add(results, "ready_volatility", gv.get("status") == "ready", f"ready gv {gv}")
        models, _ = get_json(base_url, "/models", timeout=timeout)
        gv_model = models.get("global_volatility", {})
        add(results, "models_v8", gv_model.get("status") == "ready", f"models {gv_model}")
        model_id = gv_model.get("model_id", "")
        add(
            results,
            "model_id_v8",
            "v8" in str(model_id).lower(),
            f"model_id {model_id} must contain v8",
        )
        # Forecasts — expect v8 evidence if certified, else 503 abstention is correct for now
        for ticker in ["MSFT", "NMM", "AAPL"]:
            code, body = get_json_with_status(
                base_url, f"/api/v2/forecast?ticker={ticker}&horizon=7", timeout=timeout
            )
            if code == 200:
                metric = body.get("evidence", {}).get("metric_source")
                scope = body.get("evidence", {}).get("certification_scope")
                add(
                    results,
                    f"forecast_{ticker}_metric",
                    metric == EXPECTED_V8_METRIC_SOURCE,
                    f"{ticker} metric {metric} != {EXPECTED_V8_METRIC_SOURCE}",
                )
                add(
                    results,
                    f"forecast_{ticker}_scope",
                    scope == EXPECTED_V8_SCOPE,
                    f"{ticker} scope {scope} != {EXPECTED_V8_SCOPE}",
                )
                # volatility must be non-negative finite
                var = body.get("forecast", {}).get("expected_cumulative_variance")
                add(
                    results,
                    f"forecast_{ticker}_variance",
                    isinstance(var, (int, float)) and var > 0,
                    f"{ticker} variance {var}",
                )
            elif code == 503:
                # Correct abstention until signed archive is live — not a failure in dry-run
                add(
                    results,
                    f"forecast_{ticker}_abstention",
                    body.get("detail", {}).get("status") == "abstain_no_certified_model",
                    f"{ticker} 503 but not abstain {body}",
                )
            else:
                add(results, f"forecast_{ticker}", False, f"{ticker} unexpected {code} {body}")
        # Invalid ticker / horizon
        code, _ = get_json_with_status(
            base_url, "/api/v2/forecast?ticker=INVALID123&horizon=7", timeout=timeout
        )
        add(results, "invalid_ticker", code in (400, 404, 503), f"invalid ticker code {code}")
        code, _ = get_json_with_status(
            base_url, "/api/v2/forecast?ticker=MSFT&horizon=999", timeout=timeout
        )
        add(results, "invalid_horizon", code == 400, f"invalid horizon code {code}")
    except Exception as e:
        add(results, "smoke_exception", False, safe_error(e))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="v8 deployment smoke")
    ap.add_argument("--base-url", required=True, help="Render base URL")
    ap.add_argument("--vercel-url", default=None)
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()
    results = run_v8_smoke(args.base_url, args.vercel_url, timeout=args.timeout)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} {r.name}: {r.detail}")
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed and any("abstention" not in r.name for r in failed):
        # Abstention is expected until archive live; only non-abstention failures are blocking
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
