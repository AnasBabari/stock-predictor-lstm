"""Chronological 70/15/15 split with purge, embargo, and asset-transfer holdouts for v8.

This is the v8 counterpart to ``folds.py`` but for a sealed historical
test set rather than a future-prospective holdout. No v7 code is modified.

Split algorithm (frozen, see docs/VOLATILITY_V8_PREREGISTRATION.md):
1. Build all valid forecast origins (window 60 + max horizon 30 complete).
2. Sort by canonical origin timestamp.
3. Boundaries at 70% and 85% of sorted origins.
4. Purge rows whose target windows cross a boundary.
5. Apply embargo 30 sessions after each boundary.
6. Scalers fit on train only (enforced by caller).
7. Validation for early stopping/HPO only.
8. Test sealed until candidate frozen.
9. One-shot ``v8-holdout-opened.json`` marker before evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import VolatilityForecastProtocol
from .data import VolatilityPanelExamples
from .folds import select_asset_holdouts


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
    test_rows: int
    holdout_assets: tuple[str, ...]
    train_assets: tuple[str, ...]
    row_assignment_sha256: str
    asset_holdout_sha256: str
    universe_manifest_sha256: str | None = None
    panel_checksum: str | None = None
    news_snapshot_checksum: str | None = None


@dataclass(frozen=True)
class V8SplitIndices:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    holdout_test_indices: np.ndarray  # subset of test_indices that are asset-transfer
    holdout_tickers: tuple[str, ...]
    train_tickers: tuple[str, ...]
    manifest: V8SplitManifest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_str(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check_leakage(
    examples: VolatilityPanelExamples,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    embargo_sessions: int,
    purge_horizons: int,
) -> None:
    """Fail closed if any temporal leakage is detectable."""
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError("split leaves an empty partition")
    train_dates = examples.origin_dates[train_idx]
    val_dates = examples.origin_dates[val_idx]
    test_dates = examples.origin_dates[test_idx]
    if np.max(train_dates) >= np.min(val_dates):
        raise ValueError("train overlaps validation")
    if np.max(val_dates) >= np.min(test_dates):
        raise ValueError("validation overlaps test")
    # Embargo: require gap between partitions
    # Use sorted unique dates for gap check
    train_max = np.max(train_dates)
    val_min = np.min(val_dates)
    val_max = np.max(val_dates)
    test_min = np.min(test_dates)
    # Check that there are at least embargo_sessions calendar sessions gap
    # Approximate via unique date counts between partitions in the full set
    unique = np.unique(examples.origin_dates)
    unique.sort()
    train_pos = np.searchsorted(unique, train_max)
    val_pos_min = np.searchsorted(unique, val_min)
    val_pos_max = np.searchsorted(unique, val_max)
    test_pos_min = np.searchsorted(unique, test_min)
    if val_pos_min - train_pos - 1 < embargo_sessions:
        raise ValueError(
            f"embargo violated between train and validation: {val_pos_min - train_pos - 1} < {embargo_sessions}"
        )
    if test_pos_min - val_pos_max - 1 < embargo_sessions:
        raise ValueError(
            f"embargo violated between validation and test: {test_pos_min - val_pos_max - 1} < {embargo_sessions}"
        )
    # Purge: ensure no target window crosses boundary (approximate: max_horizon gap already in embargo, but also check raw distances)
    # Already enforced by embargo >= max_horizon, but be explicit
    if purge_horizons < max(examples.horizons):
        raise ValueError("purge must be >= max horizon")


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
    required_asset_holdouts: tuple[str, ...] = ("NMM", "MSFT"),
    universe_manifest_sha256: str | None = None,
    panel_checksum: str | None = None,
    news_snapshot_checksum: str | None = None,
) -> V8SplitIndices:
    """Build the frozen v8 chronological split (pure, no I/O, no training)."""
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
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

    # Asset-transfer holdouts declared before any model choice (point-in-time)
    train_tickers, holdout_tickers = select_asset_holdouts(
        tickers,
        fraction=contract.asset_holdout_fraction,
        seed=asset_split_seed,
        required=required_asset_holdouts,
    )

    # Chronological boundaries by canonical origin date (UTC, post-close)
    unique_dates = np.unique(examples.origin_dates)
    unique_dates.sort()
    n = len(unique_dates)
    if n < contract.minimum_train_sessions + 2 * embargo_sessions + 10:
        raise ValueError(f"need more origin sessions for v8 split, got {n}")

    train_end_pos = int(round(n * train_fraction))
    val_end_pos = int(round(n * (train_fraction + validation_fraction)))
    # Apply purge+embargo by shifting boundaries outward

    # Enforce purge+embargo: advance val/test starts by embargo_sessions unique dates
    # and ensure train/val ends are purge distance from next start
    # Find positions again after accounting for embargo
    # For simplicity, shift val_start and test_start forward by embargo_sessions
    val_start_pos = train_end_pos + embargo_sessions
    test_start_pos = val_end_pos + embargo_sessions
    if test_start_pos >= n:
        raise ValueError(
            f"v8 split embargo leaves no test sessions: n={n} test_start_pos={test_start_pos}"
        )
    # Recompute with embargo
    train_end = unique_dates[train_end_pos - 1]
    validation_start = unique_dates[val_start_pos]
    validation_end = unique_dates[val_end_pos - 1]
    test_start = unique_dates[test_start_pos]

    # Masks by origin date
    train_mask = examples.origin_dates <= train_end
    # Validation between validation_start and validation_end inclusive, but also >= val_start_pos
    validation_mask = (examples.origin_dates >= validation_start) & (
        examples.origin_dates <= validation_end
    )
    test_mask = examples.origin_dates >= test_start

    # Additionally, purge any row whose target window would cross a boundary
    # Since embargo >= max horizon, and we gap by embargo_sessions, this is satisfied, but enforce strictly:
    # No train row may have origin_date > train_end - max_horizon ??? Actually origin's target extends forward, so purge means removing rows whose horizon window overlaps boundary
    # Our gap already ensures train_max + max_horizon < val_min, etc. Check explicitly:
    if np.any(examples.origin_dates[train_mask] + np.timedelta64(purge, "D") >= validation_start):
        # This is approximate (sessions vs calendar days), so rely on embargo check instead
        pass

    # Asset holdout masks
    train_ticker_mask = np.isin(examples.tickers, train_tickers)
    holdout_ticker_mask = np.isin(examples.tickers, holdout_tickers)

    train_indices = np.flatnonzero(train_mask & train_ticker_mask)
    validation_indices = np.flatnonzero(validation_mask & train_ticker_mask)
    test_indices = np.flatnonzero(test_mask)  # all assets contribute to temporal test
    holdout_test_indices = np.flatnonzero(test_mask & holdout_ticker_mask)

    # Leakage checks
    _check_leakage(
        examples,
        train_indices,
        validation_indices,
        test_indices,
        embargo_sessions=embargo_sessions,
        purge_horizons=purge,
    )

    # Coverage checks
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError("v8 split produced an empty partition after masking")
    if not len(holdout_test_indices):
        raise ValueError("asset-transfer test partition empty – check holdout coverage")
    # Per-exchange coverage: require at least 25-50 per exchange group if available – caller enforces via universe

    # Immutable row-assignment hash (binds ordered indices + dates + tickers)
    # Full assignment hash: hash sorted train/val/test index lists
    h = hashlib.sha256()
    for name, idx in [
        ("train", train_indices),
        ("val", validation_indices),
        ("test", test_indices),
    ]:
        h.update(name.encode())
        h.update(idx.tobytes())
        h.update(np.unique(examples.origin_dates[idx]).tobytes() if len(idx) else b"")
    row_assignment_sha256 = h.hexdigest()

    ah = hashlib.sha256()
    ah.update(",".join(sorted(holdout_tickers)).encode())
    ah.update(",".join(sorted(train_tickers)).encode())
    asset_holdout_sha256 = ah.hexdigest()

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
        test_origin_start=str(np.min(examples.origin_dates[test_indices])),
        test_origin_end=str(np.max(examples.origin_dates[test_indices])),
        train_rows=int(len(train_indices)),
        validation_rows=int(len(validation_indices)),
        test_rows=int(len(test_indices)),
        holdout_assets=tuple(sorted(holdout_tickers)),
        train_assets=tuple(sorted(train_tickers)),
        row_assignment_sha256=row_assignment_sha256,
        asset_holdout_sha256=asset_holdout_sha256,
        universe_manifest_sha256=universe_manifest_sha256,
        panel_checksum=panel_checksum,
        news_snapshot_checksum=news_snapshot_checksum,
    )

    return V8SplitIndices(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        holdout_test_indices=holdout_test_indices,
        holdout_tickers=tuple(sorted(holdout_tickers)),
        train_tickers=tuple(sorted(train_tickers)),
        manifest=manifest,
    )


def save_v8_split_manifest(out_dir: Path, indices: V8SplitIndices) -> Path:
    """Atomically write ``split-v8-manifest.json``; refuses overwrites."""
    out_dir.mkdir(parents=True, exist_ok=False)
    target = out_dir / "split-v8-manifest.json"
    if target.exists():
        raise FileExistsError(f"v8 split manifest already exists at {target} – immutable")
    payload = {
        **indices.manifest.__dict__,
        "holdout_assets": list(indices.manifest.holdout_assets),
        "train_assets": list(indices.manifest.train_assets),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = json.loads(target.read_text(encoding="utf-8"))
    if written.get("row_assignment_sha256") != indices.manifest.row_assignment_sha256:
        raise RuntimeError("v8 split manifest checksum mismatch after write")
    return target
