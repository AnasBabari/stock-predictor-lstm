"""Smoke-test the lightweight deployment and browser-training data contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EXPECTED_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_Upper",
    "BB_Lower",
    "ATR_14",
    "OBV",
    "SPY_Return_1D",
    "QQQ_Return_1D",
    "VIX_Return_1D",
    "TNX_Return_1D",
    "Month_Sin",
    "Month_Cos",
    "Day_Sin",
    "Day_Cos",
]


def get(base_url: str, path: str) -> dict:
    request = Request(f"{base_url}{path}", headers={"Accept": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="https://stock-predictor-lstm.onrender.com",
        help="deployed API origin (default: Render production)",
    )
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    ticker = args.ticker.strip().upper()

    try:
        health = get(base_url, "/health")
        if health.get("status") != "ok":
            raise RuntimeError("/health did not report status=ok")
        models = get(base_url, "/models")
        if models.get("server_models", {}).get("status") != "disabled":
            raise RuntimeError("/models does not disable server-side model storage")
        browser = models.get("browser_training", {})
        if (
            browser.get("status") != "available"
            or browser.get("storage") != "indexeddb"
        ):
            raise RuntimeError(
                "/models does not advertise browser training with IndexedDB"
            )

        snapshot = get(base_url, f"/api/v1/training-data?ticker={ticker}")
        if snapshot.get("schema_version") != 3:
            raise RuntimeError("training-data schema version is not 3")
        if snapshot.get("feature_names") != EXPECTED_FEATURES:
            raise RuntimeError("training-data feature ordering is incompatible")
        if snapshot.get("window_size") != 60 or snapshot.get("output_width") != 30:
            raise RuntimeError("training-data window/output contract is incompatible")
        rows = snapshot.get("features") or []
        if not rows or any(
            not math.isfinite(float(value)) for row in rows for value in row
        ):
            raise RuntimeError("training-data returned no finite feature rows")
        if len(rows) != len(snapshot.get("dates", [])):
            raise RuntimeError("training-data dates and rows are not aligned")

        price = get(base_url, f"/api/v1/predict?ticker={ticker}&days=1")
        price_engine = price.get("metadata", {}).get("engine", {})
        price_mode = price.get("metadata", {}).get("execution", {}).get("mode")
        if (
            not price_engine.get("baseline_fallback")
            or price_engine.get("role") != "server_disabled_fallback"
        ):
            raise RuntimeError(
                "compatibility price forecast is not explicitly labelled baseline"
            )
        if price_mode != "baseline_fallback":
            raise RuntimeError(
                f"compatibility price mode is {price_mode!r}, expected baseline_fallback"
            )

        direction = get(base_url, f"/api/v1/predict/direction?ticker={ticker}&days=1")
        direction_engine = direction.get("metadata", {}).get("engine", {})
        if direction_engine.get(
            "role"
        ) != "server_disabled_fallback" or not direction_engine.get(
            "baseline_fallback"
        ):
            raise RuntimeError(
                "compatibility direction forecast is not explicitly labelled baseline"
            )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "ticker": ticker,
                    "feature_rows": len(rows),
                    "server_models": models["server_models"],
                    "browser_training": browser,
                    "price_engine": price_engine,
                    "direction_engine": direction_engine,
                }
            )
        )
        return 0
    except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
        print(f"deployment smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
