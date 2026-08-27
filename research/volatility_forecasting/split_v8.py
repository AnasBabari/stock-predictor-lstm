"""Chronological 70/15/15 split with purge, embargo, and asset-transfer holdouts for v8.

This is the v8 counterpart to ``folds.py`` but for a sealed historical
test set rather than a future-prospective holdout. No v7 code is modified.

Split algorithm (frozen, see docs/VOLATILITY_V8_PREREGISTRATION.md):
1. Build all valid forecast origins (window 60 + max horizon 30 complete).
2. Sort by canonical origin timestamp.
3. Boundaries at 70% and 85% of sorted origins.
4. Purge rows whose target windows cross a boundary — per-row target-end.
5. Apply embargo 30 sessions after each boundary — per-asset calendar.
6. Scalers fit on train only (enforced by caller).
7. Validation for early stopping/HPO only.
8. Test sealed until candidate frozen.
9. One-shot ``v8-holdout-opened.json`` marker before evaluation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import VolatilityForecastProtocol
from .data import VolatilityPanelExamples

V8_REQUIRED_EXCHANGE_MICS = ("XLON", "XNAS", "XNYS")


@dataclass(frozen=True)
class V8SplitManifest:
    split_version: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    chronological: bool
    purge_horizon_sessions: int
    embargo_sessions: int
    train_origin_start: str
    train_origin_end: str
    validation_origin_start: str
    validation_origin_end: str
    test_origin_start: str
    test_origin_end: str
    train_rows: int
    validation_rows: int
    temporal_test_rows: int
    asset_transfer_test_rows: int
    pooled_test_rows: int
    holdout_assets: tuple[str, ...]
    train_assets: tuple[str, ...]
    row_assignment_sha256: str
    temporal_test_assignment_sha256: str
    asset_transfer_assignment_sha256: str
    asset_holdout_sha256: str
    asset_identity_sha256: str
    train_assets_per_exchange: dict[str, int]
    holdout_assets_per_exchange: dict[str, int]
    train_rows_per_exchange: dict[str, int]
    validation_rows_per_exchange: dict[str, int]
    temporal_test_rows_per_exchange: dict[str, int]
    asset_transfer_rows_per_exchange: dict[str, int]
    universe_manifest_sha256: str | None = None
    panel_checksum: str | None = None
    news_snapshot_checksum: str | None = None
    coverage_certifiable: bool = True


@dataclass(frozen=True)
class V8SplitIndices:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    temporal_test_indices: np.ndarray  # trainable assets in test window
    asset_transfer_test_indices: np.ndarray  # holdout assets in test window
    pooled_test_indices: np.ndarray  # union of above
    holdout_tickers: tuple[str, ...]
    train_tickers: tuple[str, ...]
    manifest: V8SplitManifest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _per_ticker_unique_dates(
    examples: VolatilityPanelExamples,
) -> dict[str, np.ndarray]:
    """Map ticker -> sorted unique origin_dates for that ticker (for per-asset purge)."""
    mapping: dict[str, np.ndarray] = {}
    for t in np.unique(examples.tickers):
        mask = examples.tickers == t
        dates = np.unique(examples.origin_dates[mask])
        dates.sort()
        mapping[str(t).upper()] = dates
    return mapping


def _target_end_date_for_origin(
    origin: np.datetime64,
    unique_dates_for_ticker: np.ndarray,
    purge_sessions: int,
) -> np.datetime64 | None:
    """Return the session that is ``purge_sessions`` trading sessions after ``origin`` for this ticker.

    If the ticker has insufficient future sessions to determine the target end, return None
    (origin is not target-complete and should have been filtered upstream).
    """
    pos = np.searchsorted(unique_dates_for_ticker, origin)
    if pos >= len(unique_dates_for_ticker) or unique_dates_for_ticker[pos] != origin:
        return None
    target_pos = pos + purge_sessions
    if target_pos >= len(unique_dates_for_ticker):
        return None
    # The target window ends at the session ``purge_sessions`` after origin;
    # for strict purge we compare origin's target-end to next partition's first origin.
    # Use the ticker's own calendar.
    return unique_dates_for_ticker[target_pos]


def _check_leakage(
    examples: VolatilityPanelExamples,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    temporal_test_idx: np.ndarray,
    asset_transfer_idx: np.ndarray,
    pooled_test_idx: np.ndarray,
    *,
    embargo_sessions: int,
    purge_sessions: int,
) -> None:
    """Fail closed if any temporal leakage is detectable — strict per-row target-end."""
    if not len(train_idx) or not len(val_idx) or not len(pooled_test_idx):
        raise ValueError("split leaves an empty partition (train/val/pooled_test)")
    if not len(temporal_test_idx) or not len(asset_transfer_idx):
        raise ValueError("split leaves empty temporal or asset-transfer test")

    # Basic ordering: max train < min val < min test (pooled)
    train_dates = examples.origin_dates[train_idx]
    val_dates = examples.origin_dates[val_idx]
    pooled_dates = examples.origin_dates[pooled_test_idx]
    if np.max(train_dates) >= np.min(val_dates):
        raise ValueError(f"train overlaps validation: {np.max(train_dates)} >= {np.min(val_dates)}")
    if np.max(val_dates) >= np.min(pooled_dates):
        raise ValueError(f"validation overlaps test: {np.max(val_dates)} >= {np.min(pooled_dates)}")

    # Global embargo via unique_dates positions
    unique = np.unique(examples.origin_dates)
    unique.sort()
    train_max = np.max(train_dates)
    val_min = np.min(val_dates)
    val_max = np.max(val_dates)
    test_min = np.min(pooled_dates)
    train_pos = int(np.searchsorted(unique, train_max))
    val_min_pos = int(np.searchsorted(unique, val_min))
    val_max_pos = int(np.searchsorted(unique, val_max))
    test_min_pos = int(np.searchsorted(unique, test_min))
    if val_min_pos - train_pos - 1 < embargo_sessions:
        raise ValueError(
            f"embargo violated between train and validation: {val_min_pos - train_pos - 1} < {embargo_sessions}"
        )
    if test_min_pos - val_max_pos - 1 < embargo_sessions:
        raise ValueError(
            f"embargo violated between validation and test: {test_min_pos - val_max_pos - 1} < {embargo_sessions}"
        )
    if purge_sessions < max(examples.horizons):
        raise ValueError(f"purge {purge_sessions} must be >= max horizon {max(examples.horizons)}")

    # Strict per-row target-end purge — per-asset calendar, tested for each horizon
    per_ticker_dates = _per_ticker_unique_dates(examples)
    max_horizon = max(examples.horizons)
    # Build lookup for each partition's first origin
    # For each train row, its target window must end *before* val_min
    # For each val row, its target window must end *before* test_min
    # Iterate per ticker to handle different holiday calendars (NYSE/NASDAQ/XLON)
    for partition_name, idx, next_start in [
        ("train", train_idx, val_min),
        ("validation", val_idx, test_min),
    ]:
        # Use set of tickers present in this partition for per-ticker check
        for ticker in np.unique(examples.tickers[idx]):
            t_dates = per_ticker_dates[str(ticker).upper()]
            row_positions = np.where(examples.tickers == ticker)[0]
            # Intersect with partition indices
            partition_rows_for_ticker = np.intersect1d(idx, row_positions, assume_unique=False)
            for row in partition_rows_for_ticker:
                origin = examples.origin_dates[row]
                target_end = _target_end_date_for_origin(origin, t_dates, purge_sessions)
                if target_end is None:
                    # Origin is not target-complete for this ticker's calendar — should not happen
                    # as examples are already target-complete, so this indicates inconsistent calendar
                    raise ValueError(
                        f"{partition_name} row {row} ticker {ticker} origin {origin} has no target-end "
                        f"for purge {purge_sessions} on its calendar"
                    )
                if target_end >= next_start:
                    raise ValueError(
                        f"purge violation: {partition_name} ticker {ticker} origin {origin} "
                        f"target_end {target_end} (purge {purge_sessions}) >= next partition start {next_start} "
                        f"(max_horizon={max_horizon})"
                    )
        # Also test horizons independently: ensure even smallest horizon (1) does not cross if purge is misconfigured
        for horizon in sorted(examples.horizons):
            # For each horizon, purge should be >= horizon; we already enforce max, but explicitly test
            if purge_sessions < horizon:
                raise ValueError(f"purge {purge_sessions} < horizon {horizon}")


def _hash_partition_rows(
    examples: VolatilityPanelExamples,
    indices: np.ndarray,
    partition: str,
    panel_checksum: str | None,
    universe_manifest_sha256: str | None,
    asset_exchange_map: dict[str, str] | None,
    asset_security_id_map: dict[str, str] | None,
) -> str:
    """Strong per-row assignment hash.

    Includes per row: stable security ID (ticker as proxy until security_id present),
    ticker, origin timestamp, target-end timestamp, partition, exchange (via ticker->MIC if known),
    and snapshot identities.  This binds index positions to underlying row identities.
    """
    h = hashlib.sha256()
    # Include partition and global snapshot identities in hash prefix
    h.update(partition.encode())
    h.update(b"|")
    h.update((panel_checksum or "no-panel-checksum").encode())
    h.update(b"|")
    h.update((universe_manifest_sha256 or "no-universe-checksum").encode())
    h.update(b"|")
    # Sort rows by (ticker, origin) for determinism
    order = np.lexsort((examples.origin_dates[indices], examples.tickers[indices]))
    sorted_idx = indices[order]
    # Per-ticker calendars for target-end
    per_ticker_dates = _per_ticker_unique_dates(examples)
    # Use max horizon for target-end in hash (or 30 as purge default)
    purge_for_hash = max(examples.horizons)
    for row in sorted_idx:
        ticker = str(examples.tickers[row]).upper()
        origin = str(examples.origin_dates[row])
        t_dates = per_ticker_dates.get(ticker)
        target_end = "unknown"
        if t_dates is not None:
            te = _target_end_date_for_origin(examples.origin_dates[row], t_dates, purge_for_hash)
            if te is not None:
                target_end = str(te)
        exchange = (asset_exchange_map or {}).get(ticker, "UNKNOWN_MIC")
        security_id = (asset_security_id_map or {}).get(ticker, "UNKNOWN_SECURITY_ID")
        h.update(
            f"{security_id}|{ticker}|{origin}|{target_end}|{partition}|{exchange}|"
            f"{panel_checksum or ''}".encode()
        )
        h.update(b"\n")
    return h.hexdigest()


def _stable_asset_order(tickers: np.ndarray, seed: int) -> list[str]:
    return sorted(
        (str(ticker).upper() for ticker in tickers),
        key=lambda ticker: (hashlib.sha256(f"{seed}:{ticker}".encode()).hexdigest(), ticker),
    )


def _select_asset_holdouts_stratified(
    tickers: np.ndarray,
    *,
    fraction: float,
    seed: int,
    required: tuple[str, ...],
    exchange_map: dict[str, str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic holdout with every represented venue included."""

    normalized = np.asarray(sorted({str(ticker).upper() for ticker in tickers}))
    required_set = set(required)
    if not required_set.issubset(set(normalized)):
        raise ValueError("required asset holdouts are absent from the panel")
    target = max(len(required_set), int(round(len(normalized) * fraction)))
    holdouts = set(required_set)
    if exchange_map is not None:
        represented = sorted({exchange_map[ticker] for ticker in normalized})
        target = max(target, len(represented))
        for mic in represented:
            if any(exchange_map[ticker] == mic for ticker in holdouts):
                continue
            candidates = np.asarray(
                [ticker for ticker in normalized if exchange_map[ticker] == mic]
            )
            holdouts.add(_stable_asset_order(candidates, seed)[0])
    for ticker in _stable_asset_order(normalized, seed):
        if len(holdouts) >= target:
            break
        holdouts.add(ticker)
    train = np.asarray(sorted(set(normalized) - holdouts))
    selected = np.asarray(sorted(holdouts))
    if not len(train) or not len(selected):
        raise ValueError("asset-transfer split leaves an empty asset population")
    return train, selected


def _counts_by_exchange(
    tickers: np.ndarray,
    exchange_map: dict[str, str],
) -> dict[str, int]:
    counts = {mic: 0 for mic in V8_REQUIRED_EXCHANGE_MICS}
    for ticker in tickers:
        mic = exchange_map[str(ticker).upper()]
        counts[mic] = counts.get(mic, 0) + 1
    return dict(sorted(counts.items()))


def build_v8_chronological_split(
    examples: VolatilityPanelExamples,
    protocol: VolatilityForecastProtocol | None = None,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    embargo_sessions: int = 30,
    purge_horizon_sessions: int | None = None,
    asset_split_seed: int = 42,
    required_asset_holdouts: tuple[str, ...] | None = None,
    universe_manifest_sha256: str | None = None,
    universe_coverage_certifiable: bool = False,
    panel_checksum: str | None = None,
    news_snapshot_checksum: str | None = None,
    asset_exchange_map: dict[str, str] | None = None,
    asset_security_id_map: dict[str, str] | None = None,
) -> V8SplitIndices:
    """Build the frozen v8 chronological split (pure, no I/O, no training).

    ``required_asset_holdouts`` must be supplied explicitly from the
    preregistered universe/split manifest — no silent default.  The list is
    bound into the protocol digest and split checksum; changing it after
    candidate training invalidates the candidate.
    """
    # Explicit holdout list required (no implicit NMM/MSFT default)
    if required_asset_holdouts is None:
        raise ValueError(
            "required_asset_holdouts must be supplied explicitly from the preregistered universe/split manifest "
            "(e.g. ('NMM','MSFT', ...)). Silent defaults are not allowed for a frozen methodology."
        )
    if not isinstance(required_asset_holdouts, (tuple, list)) or not required_asset_holdouts:
        raise ValueError("required_asset_holdouts must be a non-empty tuple/list of tickers")
    # Normalize holdouts
    required_asset_holdouts = tuple(
        sorted({str(t).upper().strip() for t in required_asset_holdouts})
    )
    if any(not t for t in required_asset_holdouts):
        raise ValueError("holdout list contains empty ticker")

    # Tolerance for float sum
    import math as _math

    if not _math.isclose(
        train_fraction + validation_fraction + test_fraction, 1.0, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("split fractions must sum to 1.0")
    contract = protocol or VolatilityForecastProtocol()
    purge = purge_horizon_sessions if purge_horizon_sessions is not None else max(contract.horizons)
    if embargo_sessions < max(contract.horizons):
        raise ValueError("embargo must be >= max horizon")
    if purge < max(contract.horizons):
        raise ValueError("purge must be >= max horizon")

    tickers = np.array(sorted({str(t).upper() for t in examples.tickers}))
    if len(tickers) < 3:
        raise ValueError("v8 split requires at least 3 tickers")

    normalized_exchange_map = (
        {str(ticker).upper(): str(mic).upper() for ticker, mic in asset_exchange_map.items()}
        if asset_exchange_map is not None
        else None
    )
    normalized_security_map = (
        {
            str(ticker).upper(): str(security_id)
            for ticker, security_id in asset_security_id_map.items()
        }
        if asset_security_id_map is not None
        else None
    )
    if (normalized_exchange_map is None) != (normalized_security_map is None):
        raise ValueError("exchange and security identity maps must be supplied together")
    if normalized_exchange_map is not None and (
        set(normalized_exchange_map) != set(tickers)
        or set(normalized_security_map or {}) != set(tickers)
    ):
        raise ValueError("split identity maps must exactly cover panel assets")
    if universe_coverage_certifiable:
        if normalized_exchange_map is None or normalized_security_map is None:
            raise ValueError("certifiable split requires exchange and security identity maps")
        unknown_mics = sorted(
            set(normalized_exchange_map.values()) - set(V8_REQUIRED_EXCHANGE_MICS)
        )
        if unknown_mics:
            raise ValueError("certifiable split contains unknown exchange MICs")

    # Asset-transfer holdouts declared before any model choice (point-in-time)
    train_tickers, holdout_tickers = _select_asset_holdouts_stratified(
        tickers,
        fraction=contract.asset_holdout_fraction,
        seed=asset_split_seed,
        required=required_asset_holdouts,
        exchange_map=normalized_exchange_map,
    )
    # Validate that the resolved holdouts exactly match the preregistered list plus deterministic remainder
    # The required list must be subset of holdout_tickers
    if not set(required_asset_holdouts).issubset(set(holdout_tickers)):
        raise ValueError(
            f"required holdouts {required_asset_holdouts} not subset of resolved holdouts {holdout_tickers}"
        )

    # Chronological boundaries by canonical origin date (UTC, post-close)
    unique_dates = np.unique(examples.origin_dates)
    unique_dates.sort()
    n = len(unique_dates)
    if n < contract.minimum_train_sessions + 2 * embargo_sessions + 10:
        raise ValueError(f"need more origin sessions for v8 split, got {n}")

    train_end_pos = int(round(n * train_fraction))
    val_end_pos = int(round(n * (train_fraction + validation_fraction)))

    # Enforce purge+embargo: advance val/test starts by embargo_sessions unique dates
    val_start_pos = train_end_pos + embargo_sessions
    test_start_pos = val_end_pos + embargo_sessions
    if test_start_pos >= n:
        raise ValueError(
            f"v8 split embargo leaves no test sessions: n={n} test_start_pos={test_start_pos}"
        )
    train_end = unique_dates[train_end_pos - 1]
    validation_start = unique_dates[val_start_pos]
    validation_end = unique_dates[val_end_pos - 1]
    test_start = unique_dates[test_start_pos]

    # Masks by origin date
    train_mask = examples.origin_dates <= train_end
    validation_mask = (examples.origin_dates >= validation_start) & (
        examples.origin_dates <= validation_end
    )
    test_mask = examples.origin_dates >= test_start

    # Asset holdout masks
    train_ticker_mask = np.isin(examples.tickers, train_tickers)
    holdout_ticker_mask = np.isin(examples.tickers, holdout_tickers)

    train_indices = np.flatnonzero(train_mask & train_ticker_mask)
    validation_indices = np.flatnonzero(validation_mask & train_ticker_mask)
    # Separate test identities: temporal (trainable assets) vs asset-transfer (holdout assets)
    temporal_test_indices = np.flatnonzero(test_mask & train_ticker_mask)
    asset_transfer_test_indices = np.flatnonzero(test_mask & holdout_ticker_mask)
    pooled_test_indices = np.flatnonzero(test_mask)

    # Leakage checks — strict per-row target-end, per-asset calendar
    _check_leakage(
        examples,
        train_indices,
        validation_indices,
        temporal_test_indices,
        asset_transfer_test_indices,
        pooled_test_indices,
        embargo_sessions=embargo_sessions,
        purge_sessions=purge,
    )

    # Coverage checks — enforce minimums for certifiable four-market model
    if not len(train_indices) or not len(validation_indices) or not len(pooled_test_indices):
        raise ValueError("v8 split produced an empty partition after masking")
    if not len(temporal_test_indices):
        raise ValueError("temporal test partition empty – check holdout coverage or embargo")
    if not len(asset_transfer_test_indices):
        raise ValueError("asset-transfer test partition empty – check holdout coverage")
    # Require panel/universe checksums for certifiable runs (not for dry-run diagnostics)
    # We allow None for non-certifiable diagnostics but record non-certifiable state
    coverage_certifiable = bool(
        panel_checksum is not None
        and universe_manifest_sha256 is not None
        and universe_coverage_certifiable
        and normalized_exchange_map is not None
        and normalized_security_map is not None
    )
    if normalized_exchange_map is None:
        train_assets_per_exchange: dict[str, int] = {}
        holdout_assets_per_exchange: dict[str, int] = {}
        train_rows_per_exchange: dict[str, int] = {}
        validation_rows_per_exchange: dict[str, int] = {}
        temporal_rows_per_exchange: dict[str, int] = {}
        transfer_rows_per_exchange: dict[str, int] = {}
    else:
        train_assets_per_exchange = _counts_by_exchange(train_tickers, normalized_exchange_map)
        holdout_assets_per_exchange = _counts_by_exchange(holdout_tickers, normalized_exchange_map)
        train_rows_per_exchange = _counts_by_exchange(
            examples.tickers[train_indices], normalized_exchange_map
        )
        validation_rows_per_exchange = _counts_by_exchange(
            examples.tickers[validation_indices], normalized_exchange_map
        )
        temporal_rows_per_exchange = _counts_by_exchange(
            examples.tickers[temporal_test_indices], normalized_exchange_map
        )
        transfer_rows_per_exchange = _counts_by_exchange(
            examples.tickers[asset_transfer_test_indices], normalized_exchange_map
        )
        if coverage_certifiable:
            tables = {
                "train assets": train_assets_per_exchange,
                "holdout assets": holdout_assets_per_exchange,
                "train rows": train_rows_per_exchange,
                "validation rows": validation_rows_per_exchange,
                "temporal test rows": temporal_rows_per_exchange,
                "asset-transfer rows": transfer_rows_per_exchange,
            }
            for label, counts in tables.items():
                missing_mics = [mic for mic in V8_REQUIRED_EXCHANGE_MICS if counts.get(mic, 0) < 1]
                if missing_mics:
                    raise ValueError(
                        f"certifiable split {label} missing exchange coverage: {missing_mics}"
                    )

    # Strong assignment hashes — per-row identity with target-end and partition
    row_assignment_sha256 = _hash_partition_rows(
        examples,
        np.concatenate((train_indices, validation_indices, pooled_test_indices)),
        "all",
        panel_checksum,
        universe_manifest_sha256,
        normalized_exchange_map,
        normalized_security_map,
    )
    temporal_sha = _hash_partition_rows(
        examples,
        temporal_test_indices,
        "temporal_test",
        panel_checksum,
        universe_manifest_sha256,
        normalized_exchange_map,
        normalized_security_map,
    )
    asset_transfer_sha = _hash_partition_rows(
        examples,
        asset_transfer_test_indices,
        "asset_transfer_test",
        panel_checksum,
        universe_manifest_sha256,
        normalized_exchange_map,
        normalized_security_map,
    )

    ah = hashlib.sha256()
    ah.update(",".join(sorted(holdout_tickers)).encode())
    ah.update(b"|")
    ah.update(",".join(sorted(train_tickers)).encode())
    asset_holdout_sha256 = ah.hexdigest()
    identity_hash = hashlib.sha256()
    for ticker in sorted(tickers):
        identity_hash.update(
            f"{ticker}|{(normalized_security_map or {}).get(ticker, 'UNKNOWN_SECURITY_ID')}|"
            f"{(normalized_exchange_map or {}).get(ticker, 'UNKNOWN_MIC')}\n".encode()
        )
    asset_identity_sha256 = identity_hash.hexdigest()

    manifest = V8SplitManifest(
        split_version="v8-chronological-70-15-15-purged",
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        chronological=True,
        purge_horizon_sessions=purge,
        embargo_sessions=embargo_sessions,
        train_origin_start=str(np.min(examples.origin_dates[train_indices])),
        train_origin_end=str(np.max(examples.origin_dates[train_indices])),
        validation_origin_start=str(np.min(examples.origin_dates[validation_indices])),
        validation_origin_end=str(np.max(examples.origin_dates[validation_indices])),
        test_origin_start=str(np.min(examples.origin_dates[pooled_test_indices])),
        test_origin_end=str(np.max(examples.origin_dates[pooled_test_indices])),
        train_rows=int(len(train_indices)),
        validation_rows=int(len(validation_indices)),
        temporal_test_rows=int(len(temporal_test_indices)),
        asset_transfer_test_rows=int(len(asset_transfer_test_indices)),
        pooled_test_rows=int(len(pooled_test_indices)),
        holdout_assets=tuple(sorted(holdout_tickers)),
        train_assets=tuple(sorted(train_tickers)),
        row_assignment_sha256=row_assignment_sha256,
        temporal_test_assignment_sha256=temporal_sha,
        asset_transfer_assignment_sha256=asset_transfer_sha,
        asset_holdout_sha256=asset_holdout_sha256,
        asset_identity_sha256=asset_identity_sha256,
        train_assets_per_exchange=train_assets_per_exchange,
        holdout_assets_per_exchange=holdout_assets_per_exchange,
        train_rows_per_exchange=train_rows_per_exchange,
        validation_rows_per_exchange=validation_rows_per_exchange,
        temporal_test_rows_per_exchange=temporal_rows_per_exchange,
        asset_transfer_rows_per_exchange=transfer_rows_per_exchange,
        universe_manifest_sha256=universe_manifest_sha256,
        panel_checksum=panel_checksum,
        news_snapshot_checksum=news_snapshot_checksum,
        coverage_certifiable=coverage_certifiable,
    )

    return V8SplitIndices(
        train_indices=train_indices,
        validation_indices=validation_indices,
        temporal_test_indices=temporal_test_indices,
        asset_transfer_test_indices=asset_transfer_test_indices,
        pooled_test_indices=pooled_test_indices,
        holdout_tickers=tuple(sorted(holdout_tickers)),
        train_tickers=tuple(sorted(train_tickers)),
        manifest=manifest,
    )


def save_v8_split_manifest(out_dir: Path, indices: V8SplitIndices) -> Path:
    """Atomically write ``split-v8-manifest.json`` (temp+fsync+replace)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "split-v8-manifest.json"
    if target.exists():
        raise FileExistsError(f"v8 split manifest already exists at {target} – immutable")
    payload = {
        **indices.manifest.__dict__,
        "holdout_assets": list(indices.manifest.holdout_assets),
        "train_assets": list(indices.manifest.train_assets),
    }
    # Atomic write via temp file in same directory
    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix=".split-v8-", dir=str(out_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(target)
        # Fsync directory (best effort, Unix only)
        try:
            if hasattr(os, "O_DIRECTORY"):
                dir_fd = os.open(str(out_dir), os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
    written = json.loads(target.read_text(encoding="utf-8"))
    if written.get("row_assignment_sha256") != indices.manifest.row_assignment_sha256:
        raise RuntimeError("v8 split manifest row_assignment checksum mismatch after write")
    if (
        written.get("temporal_test_assignment_sha256")
        != indices.manifest.temporal_test_assignment_sha256
    ):
        raise RuntimeError("v8 split manifest temporal sha mismatch after write")
    if (
        written.get("asset_transfer_assignment_sha256")
        != indices.manifest.asset_transfer_assignment_sha256
    ):
        raise RuntimeError("v8 split manifest asset-transfer sha mismatch after write")
    return target
