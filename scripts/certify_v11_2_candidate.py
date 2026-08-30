#!/usr/bin/env python3
"""Open and score the frozen V11.2 holdout exactly once.

The command is deliberately a certification boundary, not another research
runner.  It audits the development directory first, atomically consumes the
external holdout through :func:`unseal_v112_test_once`, loads only the frozen
route artifacts, and writes one immutable receipt.  No model is retrained and
no route is selected after the test payload is opened.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.volatility_forecasting.model import (  # noqa: E402
    BaselineResidualLSTM,
    BaselineResidualTCNConfig,
    RobustSequenceScaler,
)
from research.volatility_forecasting.v11_2_evaluation import (  # noqa: E402
    evaluate_horizon_gates,
    evaluate_m0_adequacy,
)
from research.volatility_forecasting.v11_2_protocol import (  # noqa: E402
    V112Protocol,
    canonical_json_digest,
    feature_schema_digest,
)
from research.volatility_forecasting.v11_2_sealed_store import (  # noqa: E402
    V112SealedTestPayload,
    load_v112_development,
    unseal_v112_test_once,
)
from research.volatility_forecasting.v11_2_trainer import (  # noqa: E402
    V112Forecast,
    make_forecast,
)
from scripts.pre_unseal_audit_v11_2 import (  # noqa: E402
    _required_file,
    _sha256,
    audit_pre_unseal,
)
from scripts.run_v11_2_numeric_development import (  # noqa: E402
    _fit_har,
    _persistence_variance,
)

_BASELINE_FAMILIES = {
    "ZERO_RETURN_CONST_VAR",
    "ZERO_RETURN_PERSISTENCE_VOL",
    "M0_HAR_BASELINE",
}


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    return payload


def _atomic_write(path: Path, data: bytes) -> None:
    """Write a certification receipt without replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
    except FileExistsError as exc:
        raise SystemExit(f"certification receipt already exists: {path}") from exc


def _require_digest(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _validate_payload(
    payload: V112SealedTestPayload,
    development: Any,
    protocol: V112Protocol,
    sealed_metadata: dict[str, Any],
) -> None:
    features = np.asarray(payload.features)
    returns = np.asarray(payload.returns)
    rv = np.asarray(payload.rv)
    rows = len(payload.dates)
    expected_shape = (protocol.window_size, len(protocol.feature_names))
    if features.ndim != 3 or features.shape[1:] != expected_shape:
        raise SystemExit("sealed test features do not match the V11.2 [rows, 60, 26] contract")
    if returns.shape != (rows, len(protocol.horizons)) or rv.shape != returns.shape:
        raise SystemExit("sealed test targets do not match the V11.2 four-horizon contract")
    if not rows or not all(np.isfinite(values).all() for values in (features, returns, rv)):
        raise SystemExit("sealed test payload contains no rows or non-finite values")
    for value in payload.dates:
        try:
            parsed = dt.date.fromisoformat(str(value))
        except ValueError as exc:
            raise SystemExit("sealed test dates must be ISO calendar dates") from exc
        if parsed.isoformat() != str(value):
            raise SystemExit("sealed test dates must use canonical ISO form")
    if len(set(payload.dates)) < protocol.bootstrap_block_sessions:
        raise SystemExit("sealed test has fewer sessions than the bootstrap block length")
    if development.train_features.ndim != 3 or development.train_features.shape[-1] != len(
        protocol.feature_names
    ):
        raise SystemExit("development feature schema does not match V11.2")
    expected_sessions = sealed_metadata.get("test_sessions")
    if (
        not isinstance(expected_sessions, list)
        or len(expected_sessions) != 2
        or min(payload.dates) != expected_sessions[0]
        or max(payload.dates) != expected_sessions[1]
    ):
        raise SystemExit("sealed test dates do not match sealed metadata")
    if len(set(payload.dates)) != int(sealed_metadata.get("test_unique_sessions", -1)):
        raise SystemExit("sealed test session count does not match sealed metadata")
    if rows != int(sealed_metadata.get("test_stock_origin_observations", -1)):
        raise SystemExit("sealed test row count does not match sealed metadata")


def _load_scaler(
    results_root: Path, protocol: V112Protocol
) -> tuple[RobustSequenceScaler, dict[str, Any]]:
    path = _required_file(results_root, "numeric_scaler.json", "numeric scaler")
    payload = _json_object(path, "numeric scaler")
    scaler = RobustSequenceScaler.from_dict(payload)
    if len(scaler.median) != len(protocol.feature_names) or len(scaler.iqr) != len(
        protocol.feature_names
    ):
        raise SystemExit("numeric scaler does not match the V11.2 feature count")
    if not np.isfinite(scaler.median).all() or not np.isfinite(scaler.iqr).all():
        raise SystemExit("numeric scaler contains non-finite statistics")
    if (scaler.iqr <= 0).any():
        raise SystemExit("numeric scaler contains non-positive IQR values")
    return scaler, payload


def _load_m1_forecast(
    path: Path,
    *,
    x_eval: np.ndarray,
    base_variance: np.ndarray,
    returns: np.ndarray,
    rv: np.ndarray,
    horizon: int,
    protocol: V112Protocol,
) -> V112Forecast:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise SystemExit(f"M1 artifact cannot be loaded safely: {path.name}") from exc
    if not isinstance(state, dict):
        raise SystemExit("M1 artifact is not a state dictionary")
    config = BaselineResidualTCNConfig(
        feature_count=len(protocol.feature_names),
        horizon_count=1,
        encoder_family="lstm",
        window_size=protocol.window_size,
        channels=32,
        lstm_hidden=32,
        lstm_layers=1,
        dropout=0.15,
        patch_length=2,
        patch_stride=1,
    )
    model = BaselineResidualLSTM(config)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise SystemExit("M1 artifact does not match the frozen V11.2 architecture") from exc
    model.eval()
    with torch.no_grad():
        variance, location, _direction, _residual = model(
            torch.tensor(np.asarray(x_eval, dtype=np.float32)),
            torch.tensor(np.asarray(base_variance, dtype=np.float32).reshape(-1, 1)),
        )
    try:
        return make_forecast(
            "M1_NUMERIC_RESIDUAL",
            horizon,
            location.numpy().reshape(-1),
            variance.numpy().reshape(-1),
            returns,
            rv,
        )
    finally:
        del model, state, variance, location
        gc.collect()


def _load_sklearn_forecast(
    path: Path,
    family: str,
    *,
    x_eval: np.ndarray,
    base_variance: np.ndarray,
    returns: np.ndarray,
    rv: np.ndarray,
    horizon: int,
    scaler_payload: dict[str, Any],
) -> V112Forecast:
    try:
        artifact = pickle.loads(path.read_bytes())
    except (
        OSError,
        pickle.PickleError,
        EOFError,
        AttributeError,
        ValueError,
        ImportError,
        ModuleNotFoundError,
    ) as exc:
        raise SystemExit(f"{family} artifact cannot be loaded: {path.name}") from exc
    if not isinstance(artifact, dict) or artifact.get("horizon") != horizon:
        raise SystemExit(f"{family} artifact metadata does not match horizon {horizon}")
    if artifact.get("scaler") != scaler_payload:
        raise SystemExit(f"{family} artifact embeds a different scaler")
    model = artifact.get("model")
    expected_type = Ridge if family == "RIDGE_LOCATION_HAR_SCALE" else HistGradientBoostingRegressor
    if not isinstance(model, expected_type):
        raise SystemExit(f"{family} artifact model type is incompatible")
    values = np.asarray(x_eval, dtype=np.float64).reshape(len(x_eval), -1)
    location = np.asarray(model.predict(values), dtype=np.float64).reshape(-1)
    if not np.isfinite(location).all():
        raise SystemExit(f"{family} artifact produced non-finite locations")
    return make_forecast(family, horizon, location, base_variance, returns, rv)


def _route_forecast(
    route: dict[str, Any],
    *,
    results_root: Path,
    protocol: V112Protocol,
    scaler_payload: dict[str, Any],
    test_x: np.ndarray,
    test_returns: np.ndarray,
    test_rv: np.ndarray,
    test_har: np.ndarray,
    test_persistence: np.ndarray,
    test_constant: np.ndarray,
) -> V112Forecast:
    horizon = int(route["horizon"])
    family = str(route.get("family", ""))
    column = list(protocol.horizons).index(horizon)
    artifact = _required_file(results_root, route.get("artifact_path"), f"route {horizon} artifact")
    if _sha256(artifact) != route.get("model_digest"):
        raise SystemExit(f"route {horizon} artifact changed after the pre-unseal audit")
    if family == "M0_HAR_BASELINE":
        return make_forecast(
            family,
            horizon,
            np.zeros(len(test_returns)),
            test_har[:, column],
            test_returns[:, column],
            test_rv[:, column],
        )
    if family == "ZERO_RETURN_CONST_VAR":
        return make_forecast(
            family,
            horizon,
            np.zeros(len(test_returns)),
            test_constant[:, column],
            test_returns[:, column],
            test_rv[:, column],
        )
    if family == "ZERO_RETURN_PERSISTENCE_VOL":
        return make_forecast(
            family,
            horizon,
            np.zeros(len(test_returns)),
            test_persistence[:, column],
            test_returns[:, column],
            test_rv[:, column],
        )
    if family == "M1_NUMERIC_RESIDUAL":
        return _load_m1_forecast(
            artifact,
            x_eval=test_x,
            base_variance=test_har[:, column],
            returns=test_returns[:, column],
            rv=test_rv[:, column],
            horizon=horizon,
            protocol=protocol,
        )
    if family in {"RIDGE_LOCATION_HAR_SCALE", "HISTGB_LOCATION_HAR_SCALE"}:
        return _load_sklearn_forecast(
            artifact,
            family,
            x_eval=test_x,
            base_variance=test_har[:, column],
            returns=test_returns[:, column],
            rv=test_rv[:, column],
            horizon=horizon,
            scaler_payload=scaler_payload,
        )
    raise SystemExit(f"route {horizon} family is outside the V11.2 contract")


def _metric_payload(forecast: V112Forecast) -> dict[str, float]:
    metrics = forecast.metrics()
    return {key: float(value) for key, value in metrics.items()}


def certify(
    *,
    dataset_dir: Path,
    results_dir: Path,
    key_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Audit, consume, and score the one-shot sealed test."""
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise SystemExit("certification output directory must be absent or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    protocol = V112Protocol()
    audit = audit_pre_unseal(dataset_dir, results_dir)
    bundle_digest = _require_digest(audit["routing_bundle_sha256"], "routing bundle digest")
    bundle = _json_object(results_dir / "v11_2_routing_bundle.json", "routing bundle")
    routes = bundle.get("routes")
    if not isinstance(routes, list) or len(routes) != len(protocol.horizons):
        raise SystemExit("routing bundle does not contain exactly four routes")
    sealed_metadata = _json_object(
        dataset_dir / "sealed" / "sealed_metadata.json", "sealed metadata"
    )
    development = load_v112_development(dataset_dir)
    if development.protocol_id != protocol.protocol_id:
        raise SystemExit("development dataset protocol does not match V11.2")
    scaler, scaler_payload = _load_scaler(results_dir.resolve(), protocol)

    payload = unseal_v112_test_once(
        output_dir=dataset_dir,
        key_path=key_path,
        candidate_digest=bundle_digest,
        repository_root=repository_root,
    )
    _validate_payload(payload, development, protocol, sealed_metadata)

    test_features = np.asarray(payload.features, dtype=np.float32)
    test_returns = np.asarray(payload.returns, dtype=np.float32)
    test_rv = np.asarray(payload.rv, dtype=np.float32)
    _train_har, test_har = _fit_har(
        development.train_features,
        development.train_rv,
        test_features,
    )
    test_x = scaler.transform(test_features)
    test_persistence = _persistence_variance(test_features, protocol.horizons)
    test_constant = np.tile(np.mean(development.train_rv, axis=0), (len(test_features), 1))

    har_forecasts: dict[int, V112Forecast] = {}
    constant_forecasts: dict[int, V112Forecast] = {}
    persistence_forecasts: dict[int, V112Forecast] = {}
    for column, horizon in enumerate(protocol.horizons):
        har_forecasts[horizon] = make_forecast(
            "M0_HAR_BASELINE",
            horizon,
            np.zeros(len(test_features)),
            test_har[:, column],
            test_returns[:, column],
            test_rv[:, column],
        )
        constant_forecasts[horizon] = make_forecast(
            "ZERO_RETURN_CONST_VAR",
            horizon,
            np.zeros(len(test_features)),
            test_constant[:, column],
            test_returns[:, column],
            test_rv[:, column],
        )
        persistence_forecasts[horizon] = make_forecast(
            "ZERO_RETURN_PERSISTENCE_VOL",
            horizon,
            np.zeros(len(test_features)),
            test_persistence[:, column],
            test_returns[:, column],
            test_rv[:, column],
        )

    har_losses = {horizon: forecast.crps for horizon, forecast in har_forecasts.items()}
    constant_losses = {horizon: forecast.crps for horizon, forecast in constant_forecasts.items()}
    persistence_losses = {
        horizon: forecast.crps for horizon, forecast in persistence_forecasts.items()
    }
    m0_adequacy = evaluate_m0_adequacy(
        dates=list(payload.dates),
        horizons=protocol.horizons,
        har_losses_by_horizon=har_losses,
        constant_losses_by_horizon=constant_losses,
        persistence_losses_by_horizon=persistence_losses,
        har_crps_by_horizon={h: float(np.mean(f.crps)) for h, f in har_forecasts.items()},
        constant_crps_by_horizon={h: float(np.mean(f.crps)) for h, f in constant_forecasts.items()},
        persistence_crps_by_horizon={
            h: float(np.mean(f.crps)) for h, f in persistence_forecasts.items()
        },
        block_sessions=protocol.bootstrap_block_sessions,
        n_replicates=protocol.bootstrap_replicates,
        seed=protocol.bootstrap_seed,
    )

    route_results: list[dict[str, Any]] = []
    all_learned_passed = True
    for route in sorted(routes, key=lambda item: int(item["horizon"])):
        horizon = int(route["horizon"])
        family = str(route.get("family", ""))
        forecast = _route_forecast(
            route,
            results_root=results_dir.resolve(),
            protocol=protocol,
            scaler_payload=scaler_payload,
            test_x=test_x,
            test_returns=test_returns,
            test_rv=test_rv,
            test_har=test_har,
            test_persistence=test_persistence,
            test_constant=test_constant,
        )
        comparator = har_forecasts[horizon]
        gate_payload: dict[str, Any]
        if family in _BASELINE_FAMILIES:
            gate_payload = {
                "passed": True,
                "decision": "frozen_baseline_route",
                "reason": "route is an explicitly non-learned V11.2 baseline",
            }
        else:
            gates = evaluate_horizon_gates(
                dates=list(payload.dates),
                horizons=[horizon],
                candidate=family,
                comparator="M0_HAR_BASELINE",
                candidate_losses_by_horizon={horizon: forecast.crps},
                comparator_losses_by_horizon={horizon: comparator.crps},
                candidate_crps_by_horizon={horizon: float(np.mean(forecast.crps))},
                comparator_crps_by_horizon={horizon: float(np.mean(comparator.crps))},
                qlike_candidate_by_horizon={horizon: float(np.mean(forecast.qlike))},
                qlike_comparator_by_horizon={horizon: float(np.mean(comparator.qlike))},
                coverage_candidate_by_horizon={horizon: forecast.coverage_80},
                coverage_comparator_by_horizon={horizon: comparator.coverage_80},
                block_sessions=protocol.bootstrap_block_sessions,
                n_replicates=protocol.bootstrap_replicates,
                seed=protocol.bootstrap_seed,
            )
            gate_payload = gates[0].to_dict()
            all_learned_passed = all_learned_passed and bool(gates[0].passed)
        route_results.append(
            {
                "horizon": horizon,
                "family": family,
                "learned_promotion": bool(route.get("learned_promotion")),
                "metrics": _metric_payload(forecast),
                "return_mae": float(
                    np.mean(
                        np.abs(
                            forecast.location
                            - test_returns[:, list(protocol.horizons).index(horizon)]
                        )
                    )
                ),
                "gate": gate_payload,
            }
        )

    status = "passed" if all_learned_passed else "failed"
    report_body: dict[str, Any] = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.digest(),
        "feature_schema_sha256": feature_schema_digest(protocol),
        "candidate_digest": bundle_digest,
        "metric_source": "sealed_holdout_once",
        "sealed_test_status": "OPENED_ONCE",
        "status": status,
        "test_rows": len(payload.dates),
        "test_unique_sessions": len(set(payload.dates)),
        "test_sessions": [min(payload.dates), max(payload.dates)],
        "routes": route_results,
        "m0_adequacy": {
            name: [gate.to_dict() for gate in gates] for name, gates in m0_adequacy.items()
        },
        "pre_unseal_audit": audit,
    }
    report_digest = canonical_json_digest(report_body)
    report = {**report_body, "report_sha256": report_digest}
    _atomic_write(
        output_dir / "v11_2_holdout_certification.json",
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(output_dir / "v11_2_holdout_certification.sha256", report_digest.encode("ascii"))
    receipt = {
        "protocol_id": protocol.protocol_id,
        "candidate_digest": bundle_digest,
        "report_sha256": report_digest,
        "sealed_test_status": "OPENED_ONCE",
        "status": status,
    }
    _atomic_write(
        output_dir / "v11_2_certification_receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    del payload, development, test_features, test_returns, test_rv
    gc.collect()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--key-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--open-sealed-holdout",
        action="store_true",
        help="Required acknowledgement: this irreversibly consumes the sealed test",
    )
    args = parser.parse_args()
    if not args.open_sealed_holdout:
        parser.error("--open-sealed-holdout is required")
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must not exist or must be empty")
    report = certify(
        dataset_dir=args.dataset_dir.resolve(),
        results_dir=args.results_dir.resolve(),
        key_path=args.key_path.resolve(),
        output_dir=output,
        repository_root=args.repository_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_digest": report["candidate_digest"],
                "report": str(output / "v11_2_holdout_certification.json"),
                "sealed_test_status": report["sealed_test_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
