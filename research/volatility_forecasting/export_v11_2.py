"""Assemble a signed production release from a passed V11.2 certification.

V11.2 evaluates four horizons (1, 3, 5, and 7 sessions), while the deployed
volatility API has a six-horizon tensor contract.  This adapter composes the
verified canonical seed-42 per-horizon routes into one six-output ONNX graph.
The two horizons that V11.2 did not evaluate (14 and 30) are explicit
baseline passthroughs and are *not* marked certified in release metadata.

The adapter is deliberately stricter than the research runner: only the
numeric residual LSTM and explicit baseline routes are exportable.  Ridge and
HistGB routes remain valid research evidence, but cannot be silently converted
to a different production implementation.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import BaselineResidualLSTM, RobustSequenceScaler
from .v11_2_model import (
    V11_2_RESIDUAL_ARCHITECTURE_VERSION,
    build_v11_2_residual_model,
    v11_2_residual_architecture_manifest,
)
from .v11_2_protocol import (
    V11_2_HORIZONS,
    V11_2_MODEL_VERSION,
    V11_2_PROTOCOL_ID,
    V11_2_PROTOCOL_VERSION,
    V112Protocol,
    canonical_json_digest,
    feature_schema_digest,
    protocol_manifest,
)

V11_2_RUNTIME_HORIZONS: tuple[int, ...] = (1, 3, 5, 7, 14, 30)
V11_2_LEARNED_FAMILY = "M1_NUMERIC_RESIDUAL"
V11_2_BASELINE_FAMILIES = frozenset(
    {
        "ZERO_RETURN_CONST_VAR",
        "ZERO_RETURN_PERSISTENCE_VOL",
        "M0_HAR_BASELINE",
    }
)
V11_2_EXPORTABLE_FAMILIES = V11_2_BASELINE_FAMILIES | {V11_2_LEARNED_FAMILY}
_OUTPUT_NAMES = (
    "forecast_variance",
    "return_location",
    "direction_probabilities",
    "return_variance",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or malformed") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _safe_relative(root: Path, value: object, label: str) -> Path:
    """Resolve a results artifact without permitting path traversal."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} is not a safe portable path")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the results directory") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is missing")
    return resolved


class V112ProductionGraph(nn.Module):
    """Compose the four V11.2 single-horizon models into six outputs.

    The graph contains no Python-side preprocessing.  The train-only robust
    scaler is embedded as buffers, and baseline variance is supplied by the
    causal serving snapshot.  Horizons 14 and 30 intentionally pass that
    baseline through and remain uncertified in metadata.
    """

    def __init__(
        self,
        *,
        models: Mapping[int, BaselineResidualLSTM],
        scaler: RobustSequenceScaler,
    ) -> None:
        super().__init__()
        unknown = set(models) - set(V11_2_HORIZONS)
        if unknown:
            raise ValueError(f"V11.2 graph received unsupported horizons: {sorted(unknown)}")
        if len(scaler.median) != 26 or len(scaler.iqr) != 26:
            raise ValueError("V11.2 scaler must contain exactly 26 features")
        if not np.isfinite(scaler.median).all() or not np.isfinite(scaler.iqr).all():
            raise ValueError("V11.2 scaler contains non-finite values")
        if (scaler.iqr <= 0).any():
            raise ValueError("V11.2 scaler contains non-positive IQR values")
        if not np.isfinite(scaler.clip) or scaler.clip <= 0:
            raise ValueError("V11.2 scaler clip must be finite and positive")
        self.models = nn.ModuleDict(
            {str(horizon): model.eval() for horizon, model in models.items()}
        )
        self.register_buffer(
            "market_median",
            torch.from_numpy(np.asarray(scaler.median, dtype=np.float32)),
        )
        self.register_buffer(
            "market_iqr",
            torch.from_numpy(np.asarray(scaler.iqr, dtype=np.float32)),
        )
        self.market_clip = float(scaler.clip)

    def forward(
        self,
        features: torch.Tensor,
        baseline_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            features.ndim != 3
            or tuple(features.shape[1:]) != (60, 26)
            or baseline_variance.ndim != 2
            or baseline_variance.shape[1] != len(V11_2_RUNTIME_HORIZONS)
        ):
            raise ValueError("V11.2 graph expects [batch, 60, 26] and [batch, 6] inputs")
        scaled = torch.clamp(
            (features - self.market_median) / self.market_iqr,
            min=-self.market_clip,
            max=self.market_clip,
        )
        variance_parts: list[torch.Tensor] = []
        location_parts: list[torch.Tensor] = []
        probability_parts: list[torch.Tensor] = []
        for index, horizon in enumerate(V11_2_RUNTIME_HORIZONS):
            baseline = baseline_variance[:, index : index + 1]
            try:
                model = self.models[str(horizon)]
            except KeyError:
                model = None
            if model is None:
                variance = baseline
                location = torch.zeros_like(baseline)
                logits = torch.zeros(
                    (features.shape[0], 1, 3),
                    dtype=features.dtype,
                    device=features.device,
                )
            else:
                variance, location, logits, _residual = model(scaled, baseline)
            variance_parts.append(variance)
            location_parts.append(location)
            probability_parts.append(torch.softmax(logits, dim=-1))
        variance_output = torch.cat(variance_parts, dim=1)
        location_output = torch.cat(location_parts, dim=1)
        probability_output = torch.cat(probability_parts, dim=1)
        # V11.2 does not certify a separate return-variance calibration head.
        # Reusing the validated volatility output is explicit and shape-safe.
        return variance_output, location_output, probability_output, variance_output


def _load_routing_bundle(results_dir: Path, protocol: V112Protocol) -> tuple[dict[str, Any], str]:
    path = results_dir / "v11_2_routing_bundle.json"
    digest_path = results_dir / "v11_2_routing_bundle.sha256"
    bundle = _json_object(path, "V11.2 routing bundle")
    try:
        declared = digest_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError("V11.2 routing bundle digest is missing") from error
    _require_digest(declared, "routing bundle digest")
    body = {key: value for key, value in bundle.items() if key != "master_freeze_sha256"}
    if canonical_json_digest(body) != declared or bundle.get("master_freeze_sha256") != declared:
        raise ValueError("V11.2 routing bundle digest does not match its canonical contents")
    if bundle.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise ValueError("V11.2 routing bundle does not prove a frozen certification input")
    if canonical_json_digest(bundle.get("protocol")) != canonical_json_digest(
        protocol_manifest(protocol)
    ):
        raise ValueError("V11.2 routing bundle protocol is incompatible")
    for label in (
        "universe_sha256",
        "panel_sha256",
        "schema_sha256",
        "split_sha256",
        "development_evidence_sha256",
        "sealed_ciphertext_sha256",
    ):
        _require_digest(bundle.get(label), f"routing bundle {label}")
    if bundle.get("schema_sha256") != feature_schema_digest(protocol):
        raise ValueError("V11.2 routing bundle schema is incompatible")
    seed_evidence = bundle.get("seed_evidence_sha256")
    if (
        not isinstance(seed_evidence, list)
        or len(seed_evidence) != len(V11_2_HORIZONS) * len(protocol.seeds)
        or len(set(seed_evidence)) != len(seed_evidence)
    ):
        raise ValueError("V11.2 routing bundle seed evidence is incomplete")
    for digest in seed_evidence:
        _require_digest(digest, "routing bundle seed evidence")
    routes = bundle.get("routes")
    if not isinstance(routes, list) or len(routes) != len(V11_2_HORIZONS):
        raise ValueError("V11.2 routing bundle must contain exactly four routes")
    route_horizons = [route.get("horizon") for route in routes if isinstance(route, dict)]
    if sorted(route_horizons) != list(V11_2_HORIZONS) or len(set(route_horizons)) != len(
        V11_2_HORIZONS
    ):
        raise ValueError("V11.2 routing bundle horizons are malformed")
    return bundle, declared


def _load_certification_report(
    certification_dir: Path,
    *,
    routing_digest: str,
    protocol: V112Protocol,
) -> tuple[dict[str, Any], str]:
    path = certification_dir / "v11_2_holdout_certification.json"
    report = _json_object(path, "V11.2 holdout certification")
    report_body = {key: value for key, value in report.items() if key != "report_sha256"}
    report_digest = _require_digest(report.get("report_sha256"), "certification report digest")
    if canonical_json_digest(report_body) != report_digest:
        raise ValueError("V11.2 certification report digest does not match its contents")
    if (
        report.get("protocol_id") != protocol.protocol_id
        or report.get("protocol_sha256") != protocol.digest()
        or report.get("feature_schema_sha256") != feature_schema_digest(protocol)
        or report.get("candidate_digest") != routing_digest
        or report.get("metric_source") != "sealed_holdout_once"
        or report.get("status") != "passed"
        or report.get("sealed_test_status") != "OPENED_ONCE"
        or report.get("m0_adequacy_passed") is not True
    ):
        raise ValueError("V11.2 certification report does not authorize a release")
    adequacy = report.get("m0_adequacy")
    expected_comparators = {
        "har_vs_constant": "ZERO_RETURN_CONST_VAR",
        "har_vs_persistence": "ZERO_RETURN_PERSISTENCE_VOL",
    }
    if not isinstance(adequacy, dict) or set(adequacy) != set(expected_comparators):
        raise ValueError("V11.2 M0 adequacy evidence is incomplete")
    for comparison, comparator in expected_comparators.items():
        gates = adequacy[comparison]
        if (
            not isinstance(gates, list)
            or len(gates) != len(V11_2_HORIZONS)
            or sorted(
                gate.get("horizon")
                for gate in gates
                if isinstance(gate, dict)
                and isinstance(gate.get("horizon"), int)
                and not isinstance(gate.get("horizon"), bool)
            )
            != list(V11_2_HORIZONS)
            or any(
                not isinstance(gate, dict)
                or gate.get("candidate") != "M0_HAR_BASELINE"
                or gate.get("comparator") != comparator
                or gate.get("passed") is not True
                for gate in gates
            )
        ):
            raise ValueError("V11.2 M0 adequacy evidence is incomplete")
    route_results = report.get("routes")
    if not isinstance(route_results, list) or len(route_results) != len(V11_2_HORIZONS):
        raise ValueError("V11.2 certification report route evidence is incomplete")
    return report, _sha256_file(path)


def _load_scaler(results_dir: Path, routes: list[dict[str, Any]]) -> RobustSequenceScaler:
    path = results_dir / "numeric_scaler.json"
    scaler = RobustSequenceScaler.from_dict(_json_object(path, "V11.2 numeric scaler"))
    actual_digest = _sha256_file(path)
    if any(route.get("scaler_digest") != actual_digest for route in routes):
        raise ValueError("V11.2 route scaler digest does not match numeric_scaler.json")
    if len(scaler.median) != 26 or len(scaler.iqr) != 26:
        raise ValueError("V11.2 scaler does not match deployable_v5")
    if not np.isfinite(scaler.median).all() or not np.isfinite(scaler.iqr).all():
        raise ValueError("V11.2 scaler contains non-finite values")
    if (scaler.iqr <= 0).any():
        raise ValueError("V11.2 scaler contains non-positive IQR values")
    return scaler


def _load_m1_model(
    results_dir: Path, route: dict[str, Any], protocol: V112Protocol
) -> tuple[int, BaselineResidualLSTM] | None:
    horizon = route.get("horizon")
    family = route.get("family")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in V11_2_HORIZONS:
        raise ValueError("V11.2 route horizon is malformed")
    if family in V11_2_BASELINE_FAMILIES:
        return None
    if family != V11_2_LEARNED_FAMILY:
        raise ValueError(f"V11.2 route family {family!r} cannot be exported safely")
    artifact = _safe_relative(results_dir, route.get("artifact_path"), f"horizon {horizon} model")
    if _sha256_file(artifact) != route.get("model_digest"):
        raise ValueError(f"V11.2 horizon {horizon} model digest does not match its route")
    model = build_v11_2_residual_model(
        feature_count=len(protocol.feature_names),
        window_size=protocol.window_size,
    )
    try:
        state = torch.load(artifact, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"V11.2 horizon {horizon} model is incompatible") from error
    model.eval()
    return horizon, model


def _verify_route_evidence(
    results_dir: Path,
    bundle_routes: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    report_routes = {
        int(row["horizon"]): row
        for row in report["routes"]
        if isinstance(row, dict) and isinstance(row.get("horizon"), int)
    }
    if set(report_routes) != set(V11_2_HORIZONS):
        raise ValueError("V11.2 certification route horizons do not match the protocol")
    for route in bundle_routes:
        horizon = int(route["horizon"])
        family = route.get("family")
        if family not in V11_2_EXPORTABLE_FAMILIES:
            raise ValueError(f"V11.2 route family {family!r} is not exportable")
        if bool(route.get("learned_promotion")) != (family == V11_2_LEARNED_FAMILY):
            raise ValueError(f"V11.2 route {horizon} has inconsistent promotion metadata")
        selection = _safe_relative(
            results_dir,
            f"selection_horizon_{horizon}.json",
            f"horizon {horizon} selection record",
        )
        if _sha256_file(selection) != route.get("selection_record_digest"):
            raise ValueError(f"V11.2 horizon {horizon} selection record changed")
        artifact = _safe_relative(
            results_dir,
            route.get("artifact_path"),
            f"horizon {horizon} route artifact",
        )
        if family in V11_2_BASELINE_FAMILIES and _sha256_file(artifact) != route.get(
            "model_digest"
        ):
            raise ValueError(f"V11.2 horizon {horizon} baseline artifact changed")
        evidence = report_routes[horizon]
        if evidence.get("family") != family or bool(evidence.get("learned_promotion")) != bool(
            route.get("learned_promotion")
        ):
            raise ValueError(f"V11.2 horizon {horizon} certification route identity differs")
        gate = evidence.get("gate")
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise ValueError(f"V11.2 horizon {horizon} does not have a passing certification gate")


def _verify_onnx_parity(
    graph: V112ProductionGraph,
    onnx_path: Path,
    *,
    rows: int,
    seed: int,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> dict[str, float]:
    if rows < 2:
        raise ValueError("parity rows must be at least two")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required before a V11.2 release can be signed"
        ) from error
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 60, 26)).astype(np.float32)
    baseline = rng.lognormal(mean=-7.0, sigma=0.4, size=(rows, 6)).astype(np.float32)
    with torch.no_grad():
        expected = [
            value.cpu().numpy()
            for value in graph(torch.from_numpy(features), torch.from_numpy(baseline))
        ]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"features": features, "baseline_variance": baseline})
    if tuple(output.name for output in session.get_outputs()) != _OUTPUT_NAMES:
        raise ValueError("V11.2 ONNX output names do not match the serving contract")
    errors: dict[str, float] = {}
    for name, expected_values, actual_values in zip(_OUTPUT_NAMES, expected, actual, strict=True):
        actual_array = np.asarray(actual_values)
        if actual_array.shape != expected_values.shape or not np.isfinite(actual_array).all():
            raise ValueError(f"V11.2 ONNX parity output is invalid: {name}")
        np.testing.assert_allclose(
            actual_array,
            expected_values,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            err_msg=f"V11.2 ONNX parity failed for {name}",
        )
        errors[name] = float(np.max(np.abs(actual_array - expected_values)))
    return errors


def assemble_v11_2_release(
    *,
    results_dir: Path,
    certification_dir: Path,
    output_dir: Path,
    private_key_path: Path,
    public_key_path: Path | None = None,
    opset_version: int = 18,
    parity_rows: int = 7,
) -> dict[str, Any]:
    """Verify a passed V11.2 candidate and create one signed runtime bundle."""
    if opset_version < 17:
        raise ValueError("V11.2 production export requires ONNX opset 17 or newer")
    if output_dir.exists():
        raise ValueError("V11.2 release output must not already exist")
    results = results_dir.resolve()
    certification = certification_dir.resolve()
    protocol = V112Protocol()
    bundle, routing_digest = _load_routing_bundle(results, protocol)
    report, report_file_digest = _load_certification_report(
        certification,
        routing_digest=routing_digest,
        protocol=protocol,
    )
    routes = [row for row in bundle["routes"] if isinstance(row, dict)]
    _verify_route_evidence(results, routes, report)
    scaler = _load_scaler(results, routes)
    models: dict[int, BaselineResidualLSTM] = {}
    route_families: dict[str, str] = {}
    for route in routes:
        loaded = _load_m1_model(results, route, protocol)
        horizon = int(route["horizon"])
        route_families[str(horizon)] = str(route["family"])
        if loaded is not None:
            models[loaded[0]] = loaded[1]
    graph = V112ProductionGraph(models=models, scaler=scaler).eval()
    with tempfile.TemporaryDirectory(prefix="v11-2-release-") as temporary:
        onnx_path = Path(temporary) / "v11_2_numeric_ensemble.onnx"
        sample_features = torch.zeros(1, 60, 26, dtype=torch.float32)
        sample_baseline = torch.ones(1, 6, dtype=torch.float32)
        torch.onnx.export(
            graph,
            (sample_features, sample_baseline),
            onnx_path,
            input_names=["features", "baseline_variance"],
            output_names=list(_OUTPUT_NAMES),
            dynamic_axes={
                "features": {0: "batch"},
                "baseline_variance": {0: "batch"},
                **{name: {0: "batch"} for name in _OUTPUT_NAMES},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
        if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
            raise ValueError("V11.2 ONNX export did not produce an artifact")
        parity_errors = _verify_onnx_parity(
            graph,
            onnx_path,
            rows=parity_rows,
            seed=20260830,
        )
        onnx_bytes = onnx_path.read_bytes()

    certified_horizons = list(V11_2_HORIZONS)
    baseline_horizons = [
        horizon
        for horizon in V11_2_HORIZONS
        if route_families[str(horizon)] in V11_2_BASELINE_FAMILIES
    ]
    learned_horizons = [
        horizon
        for horizon in V11_2_HORIZONS
        if route_families[str(horizon)] == V11_2_LEARNED_FAMILY
    ]
    report_metrics = {
        str(int(row["horizon"])): {
            "family": row.get("family"),
            "metrics": row.get("metrics"),
            "gate": row.get("gate"),
        }
        for row in report["routes"]
        if isinstance(row, dict)
    }
    model_id = f"global-volatility-v11.2-numeric-pit64:{routing_digest[:16]}"
    metadata: dict[str, Any] = {
        "runtime_schema": "volatility-runtime-v1",
        "model_id": model_id,
        "model_version": V11_2_MODEL_VERSION,
        "architecture_version": V11_2_RESIDUAL_ARCHITECTURE_VERSION,
        "architecture": v11_2_residual_architecture_manifest(
            feature_count=len(protocol.feature_names),
            window_size=protocol.window_size,
        ),
        "protocol_version": V11_2_PROTOCOL_VERSION,
        "protocol_id": V11_2_PROTOCOL_ID,
        "artifact_role": "locked_v11_2_certification_release",
        "release_eligible": True,
        "feature_schema_version": protocol.feature_schema_version,
        "feature_schema_sha256": feature_schema_digest(protocol),
        "window_size": protocol.window_size,
        "horizons": list(V11_2_RUNTIME_HORIZONS),
        "feature_names": list(protocol.feature_names),
        "news_feature_count": 0,
        "news_feature_names": [],
        "members": [
            {"seed": protocol.canonical_seed, "file": "members/v11_2_numeric_ensemble.onnx"}
        ],
        "certified_horizons": certified_horizons,
        "uncertified_horizons": [14, 30],
        "learned_horizons": learned_horizons,
        "baseline_horizons": baseline_horizons,
        "route_families": route_families,
        "metric_source": "sealed_holdout_once",
        "certification_scope": "sealed_holdout_once",
        "certification_metrics": report_metrics,
        "certification_report_sha256": report_file_digest,
        "certification_report_body_sha256": report["report_sha256"],
        "routing_bundle_sha256": routing_digest,
        "panel_sha256": bundle.get("panel_sha256"),
        "universe_sha256": bundle.get("universe_sha256"),
        "split_sha256": bundle.get("split_sha256"),
        "test_stock_origin_observations": report.get("test_stock_origin_observations"),
        "test_unique_sessions": report.get("test_unique_sessions"),
        "test_sessions": report.get("test_sessions"),
        "news_status": "not_certified",
        "ensemble_semantics": (
            "canonical seed-42 per-horizon V11.2 routes; 14/30 are baseline "
            "passthroughs and remain uncertified"
        ),
        "parity_errors": parity_errors,
    }
    try:
        from release.bundle import build_release, verify_release
    except ImportError:  # pragma: no cover - repository-root invocation
        from backend.release.bundle import build_release, verify_release  # type: ignore[no-redef]

    build_release(
        output_dir,
        {"members/v11_2_numeric_ensemble.onnx": onnx_bytes},
        metadata,
        private_key_path=Path(private_key_path),
    )
    if public_key_path is not None:
        verify_release(output_dir, public_key_path=Path(public_key_path))
    return {
        "model_id": model_id,
        "bundle": str(output_dir.resolve()),
        "routing_bundle_sha256": routing_digest,
        "certification_report_sha256": report_file_digest,
        "certified_horizons": certified_horizons,
        "learned_horizons": learned_horizons,
        "baseline_horizons": baseline_horizons,
        "max_parity_error": max(parity_errors.values()),
        "parity_errors": parity_errors,
        "signed_release_metadata": metadata,
    }
