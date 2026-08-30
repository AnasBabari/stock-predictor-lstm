"""Chronological, session-grouped 70/15/15 split for V11.2.

Unlike the legacy V11.1 digest, the V11.2 digest commits every observation's
partition assignment.  This prevents a count/date collision from silently
reusing a different split.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .v11_2_protocol import (
    V11_2_EMBARGO_SESSIONS,
    V11_2_HORIZONS,
    V11_2_MAX_HORIZON,
    V11_2_PURGE_SESSIONS,
    V11_2_TRAIN_RATIO,
    V11_2_VALIDATION_RATIO,
)


@dataclass(frozen=True)
class V112Fold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_sessions: tuple[str, str]
    validation_sessions: tuple[str, str]
    purged_sessions: int
    embargo_sessions: int


@dataclass(frozen=True)
class V112Split:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_sessions: tuple[str, str]
    validation_sessions: tuple[str, str]
    test_sessions: tuple[str, str]
    train_session_count: int
    validation_session_count: int
    test_session_count: int
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_train_sessions: int
    purged_validation_sessions: int
    embargo_train_validation_sessions: int
    embargo_validation_test_sessions: int
    assignment_sha256: str
    split_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "stocklstm-volatility-v11.2-numeric-pit64",
            "nominal_split": "70/15/15",
            "train_sessions": self.train_sessions,
            "validation_sessions": self.validation_sessions,
            "test_sessions": self.test_sessions,
            "train_session_count": self.train_session_count,
            "validation_session_count": self.validation_session_count,
            "test_session_count": self.test_session_count,
            "train_stock_origin_observations": self.train_rows,
            "validation_stock_origin_observations": self.validation_rows,
            "test_stock_origin_observations": self.test_rows,
            "purged_train_sessions": self.purged_train_sessions,
            "purged_validation_sessions": self.purged_validation_sessions,
            "embargo_train_validation_sessions": self.embargo_train_validation_sessions,
            "embargo_validation_test_sessions": self.embargo_validation_test_sessions,
            "assignment_sha256": self.assignment_sha256,
            "split_sha256": self.split_sha256,
        }


def _canonical_assignments(
    dates: list[str], security_ids: list[str], assignments: list[str]
) -> bytes:
    if len(dates) != len(security_ids) or len(dates) != len(assignments):
        raise ValueError("dates, security IDs, and assignments must have equal length")
    lines = [f"{security_ids[i]}|{dates[i]}|{assignments[i]}" for i in range(len(dates))]
    return "\n".join(lines).encode("utf-8")


def _session_span(sessions: list[str], start: int, end: int) -> tuple[str, str]:
    if start < 0 or end <= start or end > len(sessions):
        raise ValueError("partition has no usable sessions")
    return sessions[start], sessions[end - 1]


def create_v112_split(
    dates: Iterable[str],
    security_ids: Iterable[str],
    *,
    train_ratio: float = V11_2_TRAIN_RATIO,
    validation_ratio: float = V11_2_VALIDATION_RATIO,
    purge_sessions: int = V11_2_PURGE_SESSIONS,
    embargo_sessions: int = V11_2_EMBARGO_SESSIONS,
) -> V112Split:
    """Build a session-grouped chronological split while preserving row order."""
    date_list = [str(value) for value in dates]
    security_list = [str(value) for value in security_ids]
    if not date_list or len(date_list) != len(security_list):
        raise ValueError("non-empty dates and security IDs of equal length are required")
    for value in date_list:
        try:
            parsed = dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"split dates must be ISO calendar dates: {value!r}") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"split dates must use canonical ISO form: {value!r}")
    if any(not value.strip() for value in security_list):
        raise ValueError("security IDs must be non-empty")
    if len(set(zip(security_list, date_list, strict=True))) != len(date_list):
        raise ValueError("panel contains duplicate security/session observations")
    sessions = sorted(set(date_list))
    if len(sessions) < 200:
        raise ValueError("V11.2 needs at least 200 unique market sessions")
    if not (0.0 < train_ratio < 1.0 and 0.0 < validation_ratio < 1.0):
        raise ValueError("split ratios must be in (0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train and validation ratios leave no test partition")
    if purge_sessions < V11_2_MAX_HORIZON or embargo_sessions < V11_2_MAX_HORIZON:
        raise ValueError("purge and embargo must cover all target horizons")

    raw_train_end = int(len(sessions) * train_ratio)
    raw_validation_end = int(len(sessions) * (train_ratio + validation_ratio))
    train_end = raw_train_end - purge_sessions
    validation_start = raw_train_end + embargo_sessions
    validation_end = raw_validation_end - purge_sessions
    test_start = raw_validation_end + embargo_sessions
    if train_end <= 0 or validation_start >= validation_end or test_start >= len(sessions):
        raise ValueError("split leaves an empty partition after purge and embargo")

    train_set = set(sessions[:train_end])
    validation_set = set(sessions[validation_start:validation_end])
    test_set = set(sessions[test_start:])
    overlap = (train_set & validation_set) | (train_set & test_set) | (validation_set & test_set)
    if overlap:
        raise ValueError("split partitions overlap")

    assignments: list[str] = []
    for date in date_list:
        if date in train_set:
            assignments.append("TRAIN")
        elif date in validation_set:
            assignments.append("VALIDATION")
        elif date in test_set:
            assignments.append("SEALED_TEST")
        else:
            assignments.append("PURGED_OR_EMBARGO")

    assignment_bytes = _canonical_assignments(date_list, security_list, assignments)
    assignment_sha = hashlib.sha256(assignment_bytes).hexdigest()
    split_sha = hashlib.sha256(
        json.dumps(
            {
                "assignment_sha256": assignment_sha,
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "purge_sessions": purge_sessions,
                "embargo_sessions": embargo_sessions,
                "horizons": V11_2_HORIZONS,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    train_idx = np.asarray(
        [i for i, part in enumerate(assignments) if part == "TRAIN"], dtype=np.int64
    )
    validation_idx = np.asarray(
        [i for i, part in enumerate(assignments) if part == "VALIDATION"], dtype=np.int64
    )
    test_idx = np.asarray(
        [i for i, part in enumerate(assignments) if part == "SEALED_TEST"], dtype=np.int64
    )
    return V112Split(
        train_indices=train_idx,
        validation_indices=validation_idx,
        test_indices=test_idx,
        train_sessions=_session_span(sessions, 0, train_end),
        validation_sessions=_session_span(sessions, validation_start, validation_end),
        test_sessions=_session_span(sessions, test_start, len(sessions)),
        train_session_count=len(train_set),
        validation_session_count=len(validation_set),
        test_session_count=len(test_set),
        train_rows=len(train_idx),
        validation_rows=len(validation_idx),
        test_rows=len(test_idx),
        purged_train_sessions=purge_sessions,
        purged_validation_sessions=purge_sessions,
        embargo_train_validation_sessions=embargo_sessions,
        embargo_validation_test_sessions=embargo_sessions,
        assignment_sha256=assignment_sha,
        split_sha256=split_sha,
    )


def create_v112_expanding_folds(
    dates: Iterable[str],
    *,
    n_folds: int = 5,
    min_train_sessions: int = 300,
    purge_sessions: int = V11_2_PURGE_SESSIONS,
    embargo_sessions: int = V11_2_EMBARGO_SESSIONS,
) -> list[V112Fold]:
    """Create expanding folds over development sessions only."""
    date_list = [str(value) for value in dates]
    sessions = sorted(set(date_list))
    if n_folds < 2 or len(sessions) < min_train_sessions + n_folds * 10:
        raise ValueError("insufficient development sessions for requested folds")
    available = len(sessions) - min_train_sessions
    step = available // n_folds
    folds: list[V112Fold] = []
    for fold in range(n_folds):
        raw_train_end = min_train_sessions + fold * step
        raw_validation_end = (
            min_train_sessions + (fold + 1) * step if fold < n_folds - 1 else len(sessions)
        )
        train_end = raw_train_end - purge_sessions
        validation_start = raw_train_end + embargo_sessions
        if train_end <= 0 or validation_start >= raw_validation_end:
            continue
        train_set = set(sessions[:train_end])
        validation_set = set(sessions[validation_start:raw_validation_end])
        train_idx = np.asarray(
            [i for i, date in enumerate(date_list) if date in train_set], dtype=np.int64
        )
        validation_idx = np.asarray(
            [i for i, date in enumerate(date_list) if date in validation_set], dtype=np.int64
        )
        if not len(train_idx) or not len(validation_idx):
            continue
        folds.append(
            V112Fold(
                fold=fold,
                train_indices=train_idx,
                validation_indices=validation_idx,
                train_sessions=_session_span(sessions, 0, train_end),
                validation_sessions=_session_span(sessions, validation_start, raw_validation_end),
                purged_sessions=purge_sessions,
                embargo_sessions=embargo_sessions,
            )
        )
    if len(folds) != n_folds:
        raise ValueError(f"expected {n_folds} expanding folds, built {len(folds)}")
    return folds


def save_split_manifest(split: V112Split, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
