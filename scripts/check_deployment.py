"""Smoke-test a deployed StockLSTM API and its learned/baseline contracts."""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("DEPLOYMENT_BASE_URL", "https://stock-predictor-lstm.onrender.com").rstrip("/")


def get(path: str) -> dict:
    request = Request(f"{BASE_URL}{path}", headers={"Accept": "application/json"})
    with urlopen(request, timeout=90) as response:  # noqa: S310 - operator-supplied HTTPS endpoint
        return json.load(response)


def main() -> int:
    try:
        models = get("/models")
        manifest = models.get("manifest", {})
        if not manifest:
            raise RuntimeError("/models returned an empty manifest.")
        price = get("/api/v1/predict?ticker=MSFT&days=1")
        mode = price.get("metadata", {}).get("execution", {}).get("mode")
        if mode != "artifact_loaded":
            raise RuntimeError(f"MSFT price forecast mode is {mode!r}, expected 'artifact_loaded'.")
        direction = get("/api/v1/predict/direction?ticker=MSFT&days=1")
        engine = direction.get("metadata", {}).get("engine", {})
        if engine.get("baseline_fallback") and engine.get("family") != "base_rate":
            raise RuntimeError("Direction fallback is not explicitly labelled base_rate.")
        print(json.dumps({"status": "passed", "models": len(manifest), "direction_engine": engine}))
        return 0
    except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
        print(f"deployment smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())