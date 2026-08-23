"""Calendar-aligned global panel folds (slice 7).

Every ticker shares the SAME session grid and the SAME time boundaries: a
row from 2025 for one stock can never train a model evaluated on another
stock in 2024. Expanding training windows advance in calendar time with a
horizon purge plus an embargo gap; an optional asset-transfer holdout
reserves entire tickers that never appear in training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class AssetCoverage:
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_master_sessions: int
    available_sessions: int
    coverage_fraction: float
    missing_session_count: int
    max_stale_run: int
    is_admissible: bool
    exclusion_reasons: list[str]


def master_session_calendar(
    frames: dict[str, pd.DataFrame],
    *,
    union: bool = True,
) -> pd.DatetimeIndex:
    """Master exchange session grid across the panel.

    When union=True (default), takes the union of all trading dates across tickers,
    avoiding truncation by recently listed IPOs or sparse assets.
    """
    if not frames:
        raise ValueError("Panel requires at least one frame.")
    calendars = [pd.DatetimeIndex(f.index).unique() for f in frames.values() if len(f) > 0]
    if not calendars:
        raise ValueError("All supplied frames are empty.")
    if union:
        shared = calendars[0]
        for cal in calendars[1:]:
            shared = shared.union(cal)
    else:
        shared = calendars[0]
        for cal in calendars[1:]:
            shared = shared.intersection(cal)
    return shared.sort_values().unique()


def common_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Intersection of every ticker's sessions, sorted, unique."""
    return master_session_calendar(frames, union=False)


def analyze_asset_coverage(
    ticker: str,
    frame: pd.DataFrame,
    master_cal: pd.DatetimeIndex,
    *,
    min_coverage_fraction: float = 0.50,
    min_sessions: int = 60,
    max_stale_run_limit: int = 10,
) -> tuple[pd.Series, AssetCoverage]:
    """Reindex frame against master_cal and assess per-asset coverage without forward-filling."""
    if len(frame) == 0:
        empty_mask = pd.Series(False, index=master_cal, name=ticker)
        cov = AssetCoverage(
            ticker=ticker,
            start_date=pd.NaT,
            end_date=pd.NaT,
            total_master_sessions=len(master_cal),
            available_sessions=0,
            coverage_fraction=0.0,
            missing_session_count=len(master_cal),
            max_stale_run=len(master_cal),
            is_admissible=False,
            exclusion_reasons=["empty_frame"],
        )
        return empty_mask, cov

    reindexed = frame.reindex(master_cal)  # Do NOT forward fill!
    close = reindexed["Close"] if "Close" in reindexed.columns else reindexed.iloc[:, 0]
    close_vals = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(close_vals) & (close_vals > 0)

    # Calculate stale price runs (consecutive identical closes)
    diff = np.diff(close_vals, prepend=np.nan)
    is_stale = (diff == 0) & valid_mask
    max_stale = 0
    current_stale = 0
    for st in is_stale:
        if st:
            current_stale += 1
            if current_stale > max_stale:
                max_stale = current_stale
        else:
            current_stale = 0

    valid_idx = master_cal[valid_mask]
    start_date = valid_idx[0] if len(valid_idx) > 0 else pd.NaT
    end_date = valid_idx[-1] if len(valid_idx) > 0 else pd.NaT
    avail_count = int(valid_mask.sum())
    cov_frac = float(avail_count / max(1, len(master_cal)))
    missing_count = len(master_cal) - avail_count

    reasons: list[str] = []
    if avail_count < min_sessions:
        reasons.append(f"available sessions {avail_count} < min {min_sessions}")
    if cov_frac < min_coverage_fraction:
        reasons.append(f"coverage fraction {cov_frac:.3f} < min {min_coverage_fraction}")
    if max_stale > max_stale_run_limit:
        reasons.append(f"max stale price run {max_stale} > limit {max_stale_run_limit}")

    coverage = AssetCoverage(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        total_master_sessions=len(master_cal),
        available_sessions=avail_count,
        coverage_fraction=cov_frac,
        missing_session_count=missing_count,
        max_stale_run=max_stale,
        is_admissible=(len(reasons) == 0),
        exclusion_reasons=reasons,
    )
    return pd.Series(valid_mask, index=master_cal, name=ticker), coverage


def cross_sectional_ranks_causal(
    values_by_ticker: dict[str, pd.Series],
    master_cal: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute cross-sectional percentile ranks per date using ONLY assets available on that date.

    Never includes future asset availability in a historical cross-sectional calculation.
    """
    df = pd.DataFrame(
        {t: s.reindex(master_cal) for t, s in values_by_ticker.items()}, index=master_cal
    )
    # rank pct=True computes rank across axis=1 (columns) ignoring NaNs
    return df.rank(axis=1, pct=True, method="average")


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


def reserve_temporal_holdout(
    master_cal: pd.DatetimeIndex,
    holdout_sessions: int = 252,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split master calendar into development sessions and locked temporal certification holdout.

    The temporal holdout occupies the most recent `holdout_sessions` dates.
    Returns (dev_calendar, holdout_calendar).
    """
    if holdout_sessions <= 0:
        return master_cal, pd.DatetimeIndex([])
    if len(master_cal) <= holdout_sessions:
        raise ValueError(
            f"Calendar length ({len(master_cal)}) must exceed holdout sessions ({holdout_sessions})"
        )
    dev_cal = master_cal[:-holdout_sessions]
    holdout_cal = master_cal[-holdout_sessions:]
    return dev_cal, holdout_cal
