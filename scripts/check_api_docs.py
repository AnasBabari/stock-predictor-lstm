"""Verify that the public API paths documented in docs/API.md exist in OpenAPI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from api import app  # noqa: E402 - backend added to sys.path above
from server_models.response_models import (  # noqa: E402
    ARTIFACT_ACTIONS,
    ARTIFACT_STATES,
    EXECUTION_MODES,
    STATUS_STAGES,
    TIMING_FIELDS,
)

DOCUMENTED_PATHS = {
    "/health",
    "/ready",
    "/models",
    "/api/v1/search",
    "/api/v1/info",
    "/api/v1/training-data",
    "/api/v1/predict",
    "/api/v1/predict/direction",
    "/api/v1/prediction-status/{request_id}",
    "/api/v1/diagnostics/{ticker}",
    "/api/v1/model-performance/{ticker}",
    "/api/v1/server-forecasts/availability",
    "/api/v1/server-forecasts/{ticker}",
}


def resolve_schema(schema: dict, components: dict) -> dict:
    reference = schema.get("$ref")
    if not reference:
        return schema
    prefix = "#/components/schemas/"
    if not reference.startswith(prefix):
        raise SystemExit(f"Unexpected OpenAPI schema reference: {reference}")
    return components[reference.removeprefix(prefix)]


def main() -> None:
    docs = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    paths = app.openapi()["paths"]
    missing_from_schema = DOCUMENTED_PATHS - set(paths)
    if missing_from_schema:
        raise SystemExit(
            "Documented API paths absent from OpenAPI: " + ", ".join(sorted(missing_from_schema))
        )

    missing_from_docs = [path for path in DOCUMENTED_PATHS if path not in docs]
    if missing_from_docs:
        raise SystemExit(
            "OpenAPI paths absent from docs/API.md: " + ", ".join(sorted(missing_from_docs))
        )

    predict_parameters = {
        parameter["name"] for parameter in paths["/api/v1/predict"]["get"].get("parameters", [])
    }
    if not {"ticker", "days"}.issubset(predict_parameters):
        raise SystemExit("Prediction OpenAPI schema no longer exposes ticker and days parameters.")

    header_parameters = {
        parameter["name"]
        for parameter in paths["/api/v1/predict"]["get"].get("parameters", [])
        if parameter.get("in") == "header"
    }
    if "X-Prediction-Request-ID" not in header_parameters:
        raise SystemExit("Prediction OpenAPI schema no longer exposes the request ID header.")

    openapi = app.openapi()
    paths = openapi["paths"]
    components = openapi["components"]["schemas"]

    status_schema = paths["/api/v1/prediction-status/{request_id}"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    if "$ref" not in status_schema:
        raise SystemExit("Prediction status OpenAPI response no longer has a typed schema.")

    forecast_names = {
        "/api/v1/predict": "PriceForecastResponse",
        "/api/v1/predict/direction": "DirectionForecastResponse",
    }
    for path, expected_name in forecast_names.items():
        response_schema = paths[path]["get"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        if not response_schema.get("$ref", "").endswith(f"/{expected_name}"):
            raise SystemExit(f"{path} no longer exposes the {expected_name} schema.")
        response = resolve_schema(response_schema, components)
        metadata = response.get("properties", {}).get("metadata", {})
        if not metadata.get("$ref", "").endswith("/ForecastMetadata"):
            raise SystemExit(f"{path} no longer references ForecastMetadata.")

    metadata = components["ForecastMetadata"]
    metadata_properties = metadata["properties"]
    timings = resolve_schema(metadata_properties["timings_seconds"], components)
    timing_properties = timings["properties"]
    if set(timing_properties) != set(TIMING_FIELDS):
        raise SystemExit("Forecast timing fields no longer match the documented contract.")
    if not set(TIMING_FIELDS).issubset(timings.get("required", [])):
        raise SystemExit(
            "Forecast timing fields must remain required, with null for skipped stages."
        )
    total = timing_properties["total"]
    if total.get("type") != "number" or total.get("minimum") != 0:
        raise SystemExit("Forecast total timing must remain a non-negative number.")
    for name in TIMING_FIELDS:
        if name == "total":
            continue
        variants = timing_properties[name].get("anyOf", [])
        if {variant.get("type") for variant in variants} != {"number", "null"}:
            raise SystemExit(f"Forecast timing {name} must remain nullable and numeric.")

    execution = resolve_schema(metadata_properties["execution"], components)["properties"]
    if set(execution["mode"].get("enum", [])) != set(EXECUTION_MODES):
        raise SystemExit("Forecast execution modes no longer match the documented contract.")
    artifact_state = metadata_properties["artifact_state_before"].get("anyOf", [])
    documented_states = next(
        (variant.get("enum", []) for variant in artifact_state if "enum" in variant), []
    )
    if set(documented_states) != set(ARTIFACT_STATES):
        raise SystemExit("Forecast artifact states no longer match the documented contract.")
    if set(metadata_properties["artifact_action"].get("enum", [])) != set(ARTIFACT_ACTIONS):
        raise SystemExit("Forecast artifact actions no longer match the documented contract.")

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
        "caller-level semantics",
        "up to 10 minutes",
        "capacity pressure",
    ):
        if required not in docs:
            raise SystemExit(f"Telemetry API documentation is missing: {required}")

    print("API documentation paths match OpenAPI.")


if __name__ == "__main__":
    main()
