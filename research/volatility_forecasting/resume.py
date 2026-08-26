"""Fail-closed resume validation for pre-registered prospective runs.

A prospective run directory is append-only evidence. Resuming may skip a
profile/seed evaluation only when its existing record is provably the exact
artifact the current configuration would regenerate: identical protocol,
panel-derived fold plan and OOF identity, objective, seed, and a training
trace consistent with the frozen full-run early-stopping contract. Anything
unexpected fails the whole resume instead of being silently retrained or
silently accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import VolatilityForecastProtocol
from .folds import VolatilityFoldPlan, build_inner_training_split
from .model import TorchTrainingConfig, VolatilityLossWeights

REPORT_FILENAME = "prospective-development-report.json"
RECORD_MANIFEST_FORMAT = "prospective-seed-record-v2"
_RECORD_NAME = re.compile(r"^(?P<profile>[a-z0-9_]+)-seed-(?P<seed>\d+)\.json$")

_REQUIRED_RECORD_KEYS = frozenset(
    {
        "folds",
        "oof_identity",
        "oof_rows",
        "pooled_metrics",
        "promotion",
        "protocol_version",
        "seed",
    }
)
_ALLOWED_RECORD_KEYS = _REQUIRED_RECORD_KEYS | {"run_manifest"}

_PROMOTION_BOOL_KEYS = (
    "promoted",
    "volatility_promoted",
    "return_distribution_promoted",
    "return_location_promoted",
    "direction_promoted",
    "holm_significant",
)
_PROMOTION_FINITE_KEYS = (
    "relative_qlike",
    "relative_qlike_upper_95",
    "relative_gaussian_crps",
    "relative_variance_only_gaussian_crps",
    "relative_return_mae",
    "relative_return_rmse",
    "dm_statistic",
    "dm_p_value",
    "coverage_80",
    "worst_fold_relative_qlike",
)
_FOLD_MATCH_KEYS = (
    "fold",
    "fit_rows",
    "early_stopping_rows",
    "fit_end",
    "early_stopping_start",
    "early_stopping_end",
    "validation_start",
    "validation_end",
    "rows",
)


class ProspectiveResumeError(RuntimeError):
    """Existing run-directory evidence failed fail-closed resume validation."""


def record_filename(profile: str, seed: int) -> str:
    return f"{profile}-seed-{seed}.json"


def protocol_fingerprint(protocol: VolatilityForecastProtocol) -> str:
    """Content-address the full protocol, including the ordered feature schema."""
    payload = json.dumps(asdict(protocol), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_manifest(
    *,
    profile_name: str,
    loss_weights: VolatilityLossWeights,
    seed: int,
    quick: bool,
    panel_checksum: str,
    panel_id: object,
    device: str,
    resamples: int,
    training: TorchTrainingConfig,
    architecture_manifest: Mapping[str, object],
    protocol: VolatilityForecastProtocol,
    context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind one seed record to the exact run configuration that produced it."""
    return {
        "format": RECORD_MANIFEST_FORMAT,
        "profile": profile_name,
        "loss_weights": asdict(loss_weights),
        "seed": int(seed),
        "quick": bool(quick),
        "panel_checksum": panel_checksum,
        "panel_id": panel_id,
        "device": device,
        "resamples": int(resamples),
        "training": asdict(training),
        "architecture": dict(architecture_manifest),
        "protocol_sha256": protocol_fingerprint(protocol),
        "context": dict(context or {}),
    }


def scan_run_directory(
    run_dir: Path,
    profiles: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, int], Path]:
    """Inventory an existing run directory, rejecting anything unexpected.

    Only the exact record filenames for the requested profile/seed grid are
    permitted. The final report, temp files, stray files, directories, and
    records for unrequested profiles or seeds all fail the resume closed.
    """
    expected = {record_filename(p, s): (p, s) for p in profiles for s in seeds}
    found: dict[tuple[str, int], Path] = {}
    seen_normalized: set[str] = set()
    for entry in sorted(run_dir.iterdir(), key=lambda item: item.name):
        name = entry.name
        if name == REPORT_FILENAME:
            raise ProspectiveResumeError(
                "run directory already contains the final report; a completed run "
                "cannot be resumed or rewritten"
            )
        if not entry.is_file() or entry.is_symlink():
            raise ProspectiveResumeError(f"run directory contains an unexpected entry: {name}")
        normalized = name.casefold()
        if normalized in seen_normalized:
            raise ProspectiveResumeError(f"run directory contains a duplicate record: {name}")
        seen_normalized.add(normalized)
        if name not in expected:
            raise ProspectiveResumeError(
                f"run directory contains an unknown file: {name}; inspect it manually "
                "before resuming"
            )
        found[expected[name]] = entry
    return found


def expected_oof_identity(
    tickers: np.ndarray,
    origin_dates: np.ndarray,
    fold_plan: VolatilityFoldPlan,
) -> dict[str, object]:
    """Recompute the deterministic OOF identity the evaluation must report.

    This mirrors ``evaluate_tcn_development`` exactly: pooled validation rows
    ordered by origin date, then ticker. It requires no model training, so a
    resumed run can prove an old record used the same panel-derived examples
    and fold plan before trusting its metrics.
    """
    indices = np.concatenate([fold.validation_indices for fold in fold_plan.folds])
    order = np.lexsort((tickers[indices], origin_dates[indices]))
    ordered = np.asarray(indices[order], dtype=np.int64)
    if len(ordered) == 0:
        raise ProspectiveResumeError("fold plan contains no validation rows")
    return {
        "rows": int(len(ordered)),
        "first_index": int(ordered.min()),
        "last_index": int(ordered.max()),
        "sha256": hashlib.sha256(ordered.tobytes()).hexdigest(),
    }


def expected_fold_summaries(
    examples,
    fold_plan: VolatilityFoldPlan,
    protocol: VolatilityForecastProtocol,
) -> tuple[dict[str, object], ...]:
    """Recompute the per-fold boundary evidence a valid record must contain."""
    summaries: list[dict[str, object]] = []
    for fold in fold_plan.folds:
        inner = build_inner_training_split(examples, fold.train_indices, protocol)
        summaries.append(
            {
                "fold": int(fold.fold),
                "fit_rows": int(len(inner.fit_indices)),
                "early_stopping_rows": int(len(inner.early_stopping_indices)),
                "fit_end": str(inner.fit_end),
                "early_stopping_start": str(inner.early_stopping_start),
                "early_stopping_end": str(inner.early_stopping_end),
                "validation_start": str(fold.validation_start),
                "validation_end": str(fold.validation_end),
                "rows": int(len(fold.validation_indices)),
            }
        )
    return tuple(summaries)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProspectiveResumeError(message)


def _exact_int(value: object, expected: int, message: str) -> None:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        message,
    )


def _finite_number(value: object, message: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        message,
    )
    return float(value)


def _validate_fold(
    fold_record: object,
    summary: Mapping[str, object],
    training: TorchTrainingConfig,
    label: str,
) -> None:
    _require(isinstance(fold_record, Mapping), f"{label}: fold record must be an object")
    assert isinstance(fold_record, Mapping)
    for key in _FOLD_MATCH_KEYS:
        _require(
            fold_record.get(key) == summary[key],
            f"{label}: fold field {key!r} does not match the recomputed fold plan "
            f"({fold_record.get(key)!r} != {summary[key]!r})",
        )
    best_epoch = fold_record.get("best_epoch")
    _require(
        isinstance(best_epoch, int) and not isinstance(best_epoch, bool) and best_epoch >= 1,
        f"{label}: best_epoch must be a positive integer",
    )
    assert isinstance(best_epoch, int)
    history = fold_record.get("training_history")
    _require(
        isinstance(history, list) and len(history) >= 1,
        f"{label}: training_history must be a non-empty list",
    )
    assert isinstance(history, list)
    for position, entry in enumerate(history, start=1):
        _require(
            isinstance(entry, Mapping) and float(entry.get("epoch", -1)) == float(position),
            f"{label}: training_history epochs must be contiguous from one",
        )
    # Early stopping stops training exactly ``patience`` epochs after the last
    # improvement, or at the epoch cap. A record therefore matches the frozen
    # full-run contract only when its trace length equals
    # min(best_epoch + patience, maximum_epochs). Quick screens (three epochs,
    # patience two) can never satisfy the full-run trace equation.
    expected_length = min(best_epoch + training.patience, training.maximum_epochs)
    _require(
        len(history) == expected_length,
        f"{label}: training trace length {len(history)} is inconsistent with the "
        f"frozen contract (best_epoch={best_epoch}, patience={training.patience}, "
        f"maximum_epochs={training.maximum_epochs}); quick or non-default records "
        "cannot satisfy a full resume",
    )
    duration = _finite_number(
        fold_record.get("duration_seconds"),
        f"{label}: duration_seconds must be finite",
    )
    _require(duration > 0, f"{label}: duration_seconds must be positive")
    parameter_count = fold_record.get("parameter_count")
    _require(
        isinstance(parameter_count, int)
        and not isinstance(parameter_count, bool)
        and parameter_count > 0,
        f"{label}: parameter_count must be a positive integer",
    )


def _validate_manifest(
    manifest: object,
    *,
    label: str,
    profile_name: str,
    loss_weights: VolatilityLossWeights,
    seed: int,
    panel_checksum: str,
    device: str,
    resamples: int,
    training: TorchTrainingConfig,
    protocol: VolatilityForecastProtocol,
    architecture_manifest: Mapping[str, object] | None,
) -> None:
    _require(isinstance(manifest, Mapping), f"{label}: run_manifest must be an object")
    assert isinstance(manifest, Mapping)
    checks: list[tuple[str, object]] = [
        ("format", RECORD_MANIFEST_FORMAT),
        ("profile", profile_name),
        ("loss_weights", asdict(loss_weights)),
        ("quick", False),
        ("panel_checksum", panel_checksum),
        ("device", device),
        ("training", asdict(training)),
        ("protocol_sha256", protocol_fingerprint(protocol)),
    ]
    if architecture_manifest is not None:
        checks.append(("architecture", dict(architecture_manifest)))
    for key, expected in checks:
        _require(
            manifest.get(key) == expected,
            f"{label}: run_manifest field {key!r} does not match the requested run "
            f"({manifest.get(key)!r} != {expected!r})",
        )
    _exact_int(manifest.get("seed"), seed, f"{label}: run_manifest seed mismatch")
    _exact_int(
        manifest.get("resamples"),
        resamples,
        f"{label}: run_manifest resamples mismatch",
    )


def validate_seed_record(
    record: object,
    *,
    profile_name: str,
    loss_weights: VolatilityLossWeights,
    seed: int,
    protocol: VolatilityForecastProtocol,
    training: TorchTrainingConfig,
    oof_identity: Mapping[str, object],
    fold_summaries: Sequence[Mapping[str, object]],
    panel_checksum: str,
    device: str,
    resamples: int,
    architecture_manifest: Mapping[str, object] | None = None,
    accept_missing_manifest: bool = False,
) -> dict[str, object]:
    """Prove one existing seed record matches the requested full run exactly.

    Raises :class:`ProspectiveResumeError` on the first mismatch. Returns the
    record so callers can embed it in the final report unchanged.
    """
    label = record_filename(profile_name, seed)
    _require(isinstance(record, Mapping), f"{label}: record must be a JSON object")
    assert isinstance(record, Mapping)
    keys = set(record)
    _require(
        keys >= _REQUIRED_RECORD_KEYS,
        f"{label}: record is missing required fields "
        f"{sorted(_REQUIRED_RECORD_KEYS - keys)}; a partial or truncated record "
        "cannot be resumed",
    )
    _require(
        keys <= _ALLOWED_RECORD_KEYS,
        f"{label}: record contains unknown fields {sorted(keys - _ALLOWED_RECORD_KEYS)}",
    )
    _require(
        record.get("protocol_version") == protocol.protocol_version,
        f"{label}: protocol_version {record.get('protocol_version')!r} does not match "
        f"{protocol.protocol_version!r}",
    )
    _exact_int(record.get("seed"), seed, f"{label}: record seed does not match the filename")

    identity = record.get("oof_identity")
    _require(isinstance(identity, Mapping), f"{label}: oof_identity must be an object")
    assert isinstance(identity, Mapping)
    _exact_int(
        record.get("oof_rows"),
        int(oof_identity["rows"]),  # type: ignore[arg-type]
        f"{label}: oof_rows does not match the recomputed fold plan",
    )
    for key in ("first_index", "last_index"):
        _exact_int(
            identity.get(key),
            int(oof_identity[key]),  # type: ignore[arg-type]
            f"{label}: oof_identity {key} does not match the recomputed fold plan",
        )
    _require(
        identity.get("sha256") == oof_identity["sha256"],
        f"{label}: oof_identity checksum does not match the panel-derived fold plan; "
        "the record was produced from different data, folds, or feature schema",
    )

    folds = record.get("folds")
    _require(
        isinstance(folds, list) and len(folds) == len(fold_summaries),
        f"{label}: record must contain exactly {len(fold_summaries)} folds",
    )
    assert isinstance(folds, list)
    for fold_record, summary in zip(folds, fold_summaries, strict=True):
        _validate_fold(fold_record, summary, training, label)

    promotion = record.get("promotion")
    _require(
        isinstance(promotion, list) and len(promotion) == len(protocol.horizons),
        f"{label}: promotion must cover every preregistered horizon {list(protocol.horizons)}",
    )
    assert isinstance(promotion, list)
    pooled = record.get("pooled_metrics")
    _require(
        isinstance(pooled, list) and len(pooled) == len(protocol.horizons),
        f"{label}: pooled_metrics must cover every preregistered horizon",
    )
    assert isinstance(pooled, list)
    for column, horizon in enumerate(protocol.horizons):
        decision = promotion[column]
        _require(
            isinstance(decision, Mapping),
            f"{label}: promotion rows must be objects",
        )
        assert isinstance(decision, Mapping)
        _exact_int(
            decision.get("horizon"),
            horizon,
            f"{label}: promotion horizon order does not match the protocol",
        )
        for key in _PROMOTION_BOOL_KEYS:
            _require(
                isinstance(decision.get(key), bool),
                f"{label}: promotion field {key!r} must be a boolean at horizon {horizon}",
            )
        for key in _PROMOTION_FINITE_KEYS:
            _finite_number(
                decision.get(key),
                f"{label}: promotion field {key!r} must be finite at horizon {horizon}",
            )
        for key in ("reasons", "return_distribution_reasons"):
            reasons = decision.get(key)
            _require(
                isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons),
                f"{label}: promotion field {key!r} must be a list of strings",
            )
        metrics_row = pooled[column]
        _require(
            isinstance(metrics_row, Mapping),
            f"{label}: pooled_metrics rows must be objects",
        )
        assert isinstance(metrics_row, Mapping)
        _exact_int(
            metrics_row.get("horizon"),
            horizon,
            f"{label}: pooled_metrics horizon order does not match the protocol",
        )
        pooled_qlike = _finite_number(
            metrics_row.get("relative_qlike"),
            f"{label}: pooled relative_qlike must be finite at horizon {horizon}",
        )
        promotion_qlike = float(decision["relative_qlike"])
        _require(
            math.isclose(pooled_qlike, promotion_qlike, rel_tol=1e-9, abs_tol=0.0),
            f"{label}: pooled and promotion relative QLIKE disagree at horizon {horizon}",
        )

    if "run_manifest" in record:
        _validate_manifest(
            record["run_manifest"],
            label=label,
            profile_name=profile_name,
            loss_weights=loss_weights,
            seed=seed,
            panel_checksum=panel_checksum,
            device=device,
            resamples=resamples,
            training=training,
            protocol=protocol,
            architecture_manifest=architecture_manifest,
        )
    else:
        _require(
            accept_missing_manifest,
            f"{label}: record predates embedded run manifests, so its panel checksum "
            "and objective identity cannot be verified from the file alone; rerun in "
            "a new directory or explicitly pass --accept-legacy-records after "
            "auditing the checkpoint",
        )
    return dict(record)


def atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    default: Callable[[object], object] | None = None,
) -> None:
    """Serialize then atomically replace, so a crash never leaves half a file."""
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
