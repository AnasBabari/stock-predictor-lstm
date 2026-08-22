"""Calendar-aligned global panel folds (slice 7).

Every ticker shares the SAME session grid and the SAME time boundaries: a
row from 2025 for one stock can never train a model evaluated on another
stock in 2024. Expanding training windows advance in calendar time with a
horizon purge plus an embargo gap; an optional asset-transfer holdout
reserves entire tickers that never appear in training.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PanelFold:
    fold: int
    train_end: int  # exclusive; rows [0, train_end) are training
    validation_start: int  # inclusive, >= train_end + horizon + embargo
    validation_end: int  # exclusive

    @property
    def purge_gap(self) -> int:
        return self.validation_start - self.train_end


def common_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Intersection of every ticker's sessions, sorted, unique."""
    if not frames:
        raise ValueError("Panel requires at least one frame.")
    calendars = [pd.DatetimeIndex(f.index).unique() for f in frames.values()]
    shared = calendars[0]
    for cal in calendars[1:]:
        shared = shared.intersection(cal)
    return shared.sort_values()


def calendar_folds(
    n_sessions: int,
    *,
    folds: int,
    horizon: int,
    embargo: int = 0,
    min_train_sessions: int = 250,
    validation_sessions: int | None = None,
) -> list[PanelFold]:
    """Expanding-window folds over the SHARED session count.

    validation_start = train_end + horizon + embargo guarantees that no
    training target window overlaps any evaluation row: a label at origin o
    consumes rows o+1..o+h, so origins < train_end have targets ≤ train_end+h−1
    and the first evaluation origin is ≥ train_end+h+embargo.

    Validation blocks are contiguous equal slices of the remaining tail.
    """
    if folds < 1:
        raise ValueError("folds must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if n_sessions < min_train_sessions + horizon + embargo + 1:
        raise ValueError(
            f"panel of {n_sessions} sessions too small for "
            f"min_train={min_train_sessions} + horizon={horizon} + embargo={embargo}"
        )

    usable_start = min_train_sessions
    tail = n_sessions - usable_start - horizon - embargo
    val_len = validation_sessions or max(1, tail // folds)
    out: list[PanelFold] = []
    for k in range(1, folds + 1):
        train_end = usable_start + (k - 1) * val_len
        validation_start = train_end + horizon + embargo
        validation_end = min(validation_start + val_len, n_sessions)
        if validation_start >= validation_end:
            break
        out.append(PanelFold(k, train_end, validation_start, validation_end))
    if not out:
        raise ValueError("No feasible folds for the requested configuration.")
    return out


def asset_transfer_split(
    tickers: list[str],
    *,
    holdout_fraction: float,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Reserve whole tickers that never appear in training.

    Deterministic under `seed`; validates fraction in (0, 1) and keeps at
    least one ticker on each side.
    """
    import random

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    unique = sorted(dict.fromkeys(tickers))
    if len(unique) < 2:
        raise ValueError("asset transfer requires at least two tickers")
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    held_out_count = max(1, int(round(len(shuffled) * holdout_fraction)))
    held_out_count = min(held_out_count, len(shuffled) - 1)
    held_out = sorted(shuffled[:held_out_count])
    train = sorted(set(unique) - set(held_out))
    return train, held_out


def assert_no_time_leakage(
    fold: PanelFold,
    *,
    horizon: int,
    embargo: int,
) -> None:
    """Raise unless the fold's purge arithmetic provably isolates evaluation.

    Training labels at origin o consume rows o+1..o+h. The last training
    origin is train_end−1 → its targets reach train_end+h−1. The first
    evaluation origin must therefore be ≥ train_end+h+embargo.
    """
    required = fold.train_end + horizon + embargo
    if fold.validation_start < required:
        raise AssertionError(
            f"fold {fold.fold}: leakage — validation_start {fold.validation_start} "
            f"< required {required} (train_end {fold.train_end} + horizon "
            f"{horizon} + embargo {embargo})"
        )
    if not (fold.validation_end - fold.validation_start >= 0):
        raise AssertionError(f"fold {fold.fold}: empty validation block")
