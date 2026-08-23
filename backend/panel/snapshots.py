"""Immutable panel snapshot provenance (overhaul slice 4).

A panel snapshot is an immutable, content-addressed bundle of daily OHLCV
history for many tickers plus a manifest recording provider, retrieval time,
timezone, adjustment mode, per-ticker checksums, and a pooled checksum.

Immutability contract: files under a snapshot directory are write-once. The
builder refuses to overwrite an existing snapshot id; corrections produce a
new snapshot with a new content address.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PANEL_SCHEMA_VERSION = 1
PROVIDER = "yfinance"
TIMEZONE = "America/New_York"
ADJUST_MODE = "auto_adjust=true"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CANONICAL_COLUMNS = ["Date"] + OHLCV_COLUMNS
MAX_SINGLE_DAY_MOVE = 0.20


class PanelValidationError(ValueError):
    """Structural panel defect: duplicates, non-monotonic dates, bad values."""


class LicenseNotAcknowledged(RuntimeError):
    """The provider-license gate was not acknowledged for this run."""


def require_license_acknowledged(env: Mapping[str, str] | None = None) -> None:
    """Refuse any provider-backed panel acquisition without explicit consent.

    The current provider's terms cover local research; they do not silently
    become the legal basis for training or redistribution at scale. Callers
    must set PANEL_LICENSE_ACKNOWLEDGED=true after reviewing them.
    """
    env = dict(os.environ if env is None else env)
    if env.get("PANEL_LICENSE_ACKNOWLEDGED", "").strip().lower() != "true":
        raise LicenseNotAcknowledged(
            "PANEL_LICENSE_ACKNOWLEDGED=true is required before downloading "
            "panel data. Review the provider terms for offline model training, "
            "public application use, derived weight distribution, caching, and "
            "redistribution restrictions first."
        )


def canonical_csv(frame: pd.DataFrame) -> str:
    """Deterministic serialization used for content addressing."""
    ordered = frame[OHLCV_COLUMNS].sort_index()
    out = ordered.copy()
    out.index.name = "Date"
    out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m-%d")
    return out.to_csv(index=True, float_format="%.6f")


def checksum_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TickerProvenance:
    ticker: str
    rows: int
    start: str
    end: str
    checksum: str
    suspicious_adjustment_rows: list[str]


def validate_ohlcv(ticker: str, frame: pd.DataFrame) -> TickerProvenance:
    """Structural validation + provenance for one ticker's OHLCV history."""
    if frame is None or frame.empty:
        raise PanelValidationError(f"{ticker}: empty frame")
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise PanelValidationError(f"{ticker}: missing columns {missing}")
    if frame.index.has_duplicates:
        raise PanelValidationError(f"{ticker}: duplicate session dates")
    if not frame.index.is_monotonic_increasing:
        raise PanelValidationError(f"{ticker}: dates are not strictly chronological")

    values = frame[OHLCV_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PanelValidationError(f"{ticker}: non-finite OHLCV values")
    if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise PanelValidationError(f"{ticker}: non-positive prices or negative volume")

    close = frame["Close"].to_numpy(dtype=float)
    log_ret = np.abs(np.diff(np.log(close)))
    suspicious = [
        pd.DatetimeIndex(frame.index)[i + 1].strftime("%Y-%m-%d")
        for i in np.nonzero(log_ret > MAX_SINGLE_DAY_MOVE)[0]
    ]

    text = canonical_csv(frame)
    index = pd.DatetimeIndex(frame.index)
    return TickerProvenance(
        ticker=ticker,
        rows=int(len(frame)),
        start=index[0].strftime("%Y-%m-%d"),
        end=index[-1].strftime("%Y-%m-%d"),
        checksum=checksum_text(text),
        suspicious_adjustment_rows=suspicious,
    )


def build_snapshot(
    frames: Mapping[str, pd.DataFrame],
    *,
    license_acknowledged: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every frame and assemble the manifest dict (no I/O)."""
    if not license_acknowledged:
        raise LicenseNotAcknowledged(
            "build_snapshot requires license_acknowledged=True after the "
            "provider terms have been reviewed."
        )
    if not frames:
        raise PanelValidationError("Panel requires at least one ticker.")

    tickers: dict[str, Any] = {}
    hasher = hashlib.sha256()
    for ticker in sorted(frames):
        provenance = validate_ohlcv(ticker, frames[ticker])
        tickers[ticker] = {
            "rows": provenance.rows,
            "start": provenance.start,
            "end": provenance.end,
            "checksum": provenance.checksum,
            "suspicious_adjustment_rows": provenance.suspicious_adjustment_rows,
        }
        hasher.update(provenance.ticker.encode())
        hasher.update(b"\0")
        hasher.update(provenance.checksum.encode())

    pooled = hasher.hexdigest()
    manifest = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "panel_id": f"panel-{pooled[:16]}",
        "pooled_checksum": f"sha256:{pooled}",
        "provider": PROVIDER,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "timezone": TIMEZONE,
        "adjust_mode": ADJUST_MODE,
        "license": {
            "acknowledged": True,
            "note": (
                "Local research use acknowledged. Redistribution of derived "
                "weights requires a separate review of provider terms."
            ),
        },
        "tickers": tickers,
        "ticker_count": len(tickers),
    }
    if extra_metadata:
        manifest["extra"] = dict(extra_metadata)
    return manifest


def write_snapshot(
    root: Path,
    frames: Mapping[str, pd.DataFrame],
    *,
    license_acknowledged: bool,
) -> Path:
    """Materialize an immutable snapshot directory; refuses overwrites."""
    manifest = build_snapshot(frames, license_acknowledged=license_acknowledged)
    out_dir = root / manifest["panel_id"]
    marker = out_dir / "manifest.json"
    if marker.exists():
        raise FileExistsError(
            f"Snapshot {manifest['panel_id']} already exists at {out_dir}; "
            "snapshots are immutable — build a new one instead."
        )
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ticker in sorted(frames):
        text = canonical_csv(frames[ticker])
        digest = checksum_text(text)
        target = raw_dir / f"{ticker}.csv"
        if target.exists():
            raise FileExistsError(f"{target} already exists.")
        # Write bytes: newline translation would break the content address.
        target.write_bytes(text.encode("utf-8"))
        written = checksum_text(target.read_bytes().decode("utf-8"))
        if written != digest:
            raise RuntimeError(f"checksum mismatch writing {target}")
    manifest_path = out_dir / "manifest.json.tmp"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.replace(marker)
    return out_dir


def load_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Load a snapshot back, verifying every checksum (fail closed)."""
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for ticker, meta in manifest["tickers"].items():
        path = snapshot_dir / "raw" / f"{ticker}.csv"
        # Read bytes and decode without newline translation so the checksum
        # matches the exact written content regardless of platform.
        text = path.read_bytes().decode("utf-8")
        if checksum_text(text) != meta["checksum"]:
            raise PanelValidationError(f"{ticker}: checksum mismatch — snapshot file was modified.")
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame.index.name = None
        validate_ohlcv(ticker, frame)
        frames[ticker] = frame
    return manifest, frames


def fetch_panel_universe(
    tickers: list[str],
    *,
    years: int = 8,
    max_tickers: int | None = None,
    allow_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV for the universe (provider-backed entry point).

    Requires PANEL_LICENSE_ACKNOWLEDGED=true in the environment. This is the
    only function in this module that touches the network.
    """
    require_license_acknowledged()
    import logging

    import yfinance as yf  # type: ignore[import-untyped]

    universe = list(dict.fromkeys(t.upper() for t in tickers))
    if max_tickers is not None:
        universe = universe[:max_tickers]
    batch = yf.download(
        universe,
        period=f"{years}y",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    for ticker in universe:
        block = batch[ticker] if isinstance(batch.columns, pd.MultiIndex) else batch
        block = block.dropna(subset=["Close"])
        if not block.empty and len(block) >= 60:
            frames[ticker] = block
    missing = sorted(set(universe) - set(frames))
    if missing:
        if not allow_missing:
            raise PanelValidationError(f"No data returned for: {', '.join(missing)}")
        logging.getLogger("panel.snapshots").warning(
            f"Omitted {len(missing)} missing/delisted tickers from panel: {', '.join(missing)}"
        )
    if not frames:
        raise PanelValidationError("No data returned for any requested ticker.")
    return frames


def load_panel_from_directory(panel_dir: Path) -> dict[str, pd.DataFrame]:
    """Load universe panel from a snapshot directory or directory of CSV/Parquet files.

    Raises FileNotFoundError if directory does not exist.
    Raises PanelValidationError if no valid ticker data is found.
    """
    if not panel_dir.exists() or not panel_dir.is_dir():
        raise FileNotFoundError(f"Panel directory does not exist: {panel_dir}")

    manifest_file = panel_dir / "manifest.json"
    if manifest_file.exists():
        _, s_frames = load_snapshot(panel_dir)
        return s_frames

    child_snapshots = [
        d for d in panel_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()
    ]
    if child_snapshots:
        _, c_frames = load_snapshot(child_snapshots[0])
        return c_frames

    raw_dir = panel_dir / "raw" if (panel_dir / "raw").is_dir() else panel_dir
    csv_files = list(raw_dir.glob("*.csv"))
    parquet_files = list(raw_dir.glob("*.parquet"))

    if not csv_files and not parquet_files:
        raise PanelValidationError(f"No CSV or Parquet ticker files found in {panel_dir}")

    frames: dict[str, pd.DataFrame] = {}
    for p in sorted(csv_files):
        ticker = p.stem.upper()
        frame = pd.read_csv(p, index_col=0, parse_dates=True)
        frame.index.name = None
        validate_ohlcv(ticker, frame)
        frames[ticker] = frame

    for p in sorted(parquet_files):
        ticker = p.stem.upper()
        if ticker in frames:
            continue
        frame = pd.read_parquet(p)
        if not isinstance(frame.index, pd.DatetimeIndex) and "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame = frame.set_index("Date")
        frame.index.name = None
        validate_ohlcv(ticker, frame)
        frames[ticker] = frame

    if not frames:
        raise PanelValidationError(f"No valid ticker data parsed from {panel_dir}")
    return frames
