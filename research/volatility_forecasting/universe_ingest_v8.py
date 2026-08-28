"""Strict ingestion of operator-supplied v8 security-master snapshots.

The project intentionally does not scrape a current constituent list and call
it point-in-time data.  This module converts an immutable CSV plus explicit
source attestations into :class:`UniverseMember` values.  It performs no
network access and never infers missing provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .universe_v8 import UniverseMember

V8_UNIVERSE_SOURCE_SCHEMA = 1

REQUIRED_COLUMNS = (
    "security_id",
    "ticker",
    "company_name",
    "primary_exchange_mic",
    "currency",
    "timezone",
    "sector",
    "security_type",
    "source",
    "source_snapshot_id",
)

OPTIONAL_COLUMNS = (
    "isin",
    "figi",
    "cik",
    "industry",
    "membership_start",
    "membership_end",
    "index_memberships_json",
    "required_history_sessions",
    "point_in_time_liquidity_ok",
)


def sha256_file(path: Path) -> str:
    """Return a labelled SHA-256 digest for exact source bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _optional_text(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


def _parse_bool(value: str | None, *, field_name: str, row_number: int) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"row {row_number}: {field_name} must be true or false")


def _parse_memberships(value: str | None, *, row_number: int) -> tuple[dict[str, str], ...]:
    if not (value or "").strip():
        return ()
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"row {row_number}: index_memberships_json is invalid JSON") from error
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"row {row_number}: index_memberships_json must be an array of objects")
    return tuple(payload)


def load_universe_members_csv(path: Path) -> list[UniverseMember]:
    """Parse the frozen v8 security-master CSV without filling missing facts."""

    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"unable to read universe members CSV: {path}") from error
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("universe members CSV has no header")
        duplicate_columns = sorted(
            {name for name in reader.fieldnames if reader.fieldnames.count(name) > 1}
        )
        if duplicate_columns:
            raise ValueError(
                "universe members CSV has duplicate columns: " + ", ".join(duplicate_columns)
            )
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError("universe members CSV missing columns: " + ", ".join(missing))
        unknown = sorted(set(reader.fieldnames) - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS))
        if unknown:
            raise ValueError("universe members CSV has unknown columns: " + ", ".join(unknown))

        members: list[UniverseMember] = []
        for row_number, row in enumerate(reader, start=2):
            required_history = (row.get("required_history_sessions") or "756").strip()
            try:
                required_history_sessions = int(required_history)
            except ValueError as error:
                raise ValueError(
                    f"row {row_number}: required_history_sessions must be an integer"
                ) from error
            if required_history_sessions < 60:
                raise ValueError(
                    f"row {row_number}: required_history_sessions must be at least 60"
                )
            liquidity_value = row.get("point_in_time_liquidity_ok")
            liquidity_ok = (
                True
                if liquidity_value is None or not liquidity_value.strip()
                else _parse_bool(
                    liquidity_value,
                    field_name="point_in_time_liquidity_ok",
                    row_number=row_number,
                )
            )
            members.append(
                UniverseMember(
                    security_id=(row.get("security_id") or "").strip(),
                    ticker=(row.get("ticker") or "").strip().upper(),
                    company_name=(row.get("company_name") or "").strip(),
                    isin=_optional_text(row.get("isin")),
                    figi=_optional_text(row.get("figi")),
                    cik=_optional_text(row.get("cik")),
                    primary_exchange_mic=(row.get("primary_exchange_mic") or "")
                    .strip()
                    .upper(),
                    index_memberships=_parse_memberships(
                        row.get("index_memberships_json"), row_number=row_number
                    ),
                    currency=(row.get("currency") or "").strip(),
                    timezone=(row.get("timezone") or "").strip(),
                    sector=_optional_text(row.get("sector")),
                    industry=_optional_text(row.get("industry")),
                    security_type=(row.get("security_type") or "").strip().upper(),
                    membership_start=_optional_text(row.get("membership_start")),
                    membership_end=_optional_text(row.get("membership_end")),
                    source=(row.get("source") or "").strip(),
                    source_snapshot_id=(row.get("source_snapshot_id") or "").strip(),
                    required_history_sessions=required_history_sessions,
                    point_in_time_liquidity_ok=liquidity_ok,
                )
            )
    if not members:
        raise ValueError("universe members CSV contains no data rows")
    return members


def load_source_attestations(path: Path) -> dict[str, dict[str, Any]]:
    """Load the source-attestation envelope used by certifiable manifests."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("source attestation file is missing or invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != V8_UNIVERSE_SOURCE_SCHEMA:
        raise ValueError(
            f"source attestation schema_version must be {V8_UNIVERSE_SOURCE_SCHEMA}"
        )
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source attestation file must contain a non-empty sources object")
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in sources.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, dict):
            raise ValueError("source attestation entries must be named JSON objects")
        evidence_files = value.get("evidence_files")
        if not isinstance(evidence_files, list) or not evidence_files:
            raise ValueError(f"source attestation {name!r} requires evidence_files")
        if any(not isinstance(item, str) or not item.strip() for item in evidence_files):
            raise ValueError(f"source attestation {name!r} has invalid evidence_files")
        normalized[name.strip()] = dict(value)
    return normalized


def validate_attestation_evidence_files(
    attestations: dict[str, dict[str, Any]], source_checksums: dict[str, str]
) -> None:
    """Require every attested evidence file to be among the hashed inputs."""

    for source_name, attestation in attestations.items():
        missing = sorted(set(attestation["evidence_files"]) - set(source_checksums))
        if missing:
            raise ValueError(
                f"source attestation {source_name!r} references unhashed evidence files: "
                + ", ".join(missing)
            )
