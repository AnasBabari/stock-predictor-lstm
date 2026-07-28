"""Verify that the public API paths documented in docs/API.md exist in OpenAPI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from api import (
    ARTIFACT_ACTIONS,
    ARTIFACT_STATES,
    EXECUTION_MODES,
    STATUS_STAGES,
    TIMING_FIELDS,
    app,
)

DOCUMENTED_PATHS = {
    "/health",
    "/ready",
    "/models",
    "/api/v1/search",
    "/api/v1/info",
    "/api/v1/predict",
    "/api/v1/predict/direction",
    "/api/v1/prediction-status/{request_id}",
    "/api/v1/diagnostics/{ticker}",
}


def main() -> None:
    docs = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    paths = app.openapi()["paths"]
    missing_from_schema = DOCUMENTED_PATHS - set(paths)
    if missing_from_schema:
        raise SystemExit(
            "Documented API paths absent from OpenAPI: "
            + ", ".join(sorted(missing_from_schema))
        )

    missing_from_docs = [path for path in DOCUMENTED_PATHS if path not in docs]
    if missing_from_docs:
        raise SystemExit(
            "OpenAPI paths absent from docs/API.md: "
            + ", ".join(sorted(missing_from_docs))
        )

    predict_parameters = {
        parameter["name"]
        for parameter in paths["/api/v1/predict"]["get"].get("parameters", [])
    }
    if not {"ticker", "days"}.issubset(predict_parameters):
        raise SystemExit(
            "Prediction OpenAPI schema no longer exposes ticker and days parameters."
        )

    header_parameters = {
        parameter["name"]
        for parameter in paths["/api/v1/predict"]["get"].get("parameters", [])
        if parameter.get("in") == "header"
    }
    if "X-Prediction-Request-ID" not in header_parameters:
        raise SystemExit(
            "Prediction OpenAPI schema no longer exposes the request ID header."
        )

    status_schema = paths["/api/v1/prediction-status/{request_id}"]["get"]["responses"][
        "200"
    ]["content"]["application/json"]["schema"]
    if "$ref" not in status_schema:
        raise SystemExit(
            "Prediction status OpenAPI response no longer has a typed schema."
        )

    for required in (
        "timings_seconds",
        "execution",
        "artifact_state_before",
        "artifact_action",
        *TIMING_FIELDS,
        *EXECUTION_MODES,
        *ARTIFACT_STATES,
        *ARTIFACT_ACTIONS,
        *STATUS_STAGES,
    ):
        if required not in docs:
            raise SystemExit(f"Telemetry API documentation is missing: {required}")

    print("API documentation paths match OpenAPI.")


if __name__ == "__main__":
    main()
