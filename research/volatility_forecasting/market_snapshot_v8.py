"""Immutable v8 market snapshots bound to an attested security universe.

Unlike the legacy panel format, a certifiable v8 snapshot preserves both raw
and adjusted OHLC values plus corporate actions. The model-facing OHLC
columns are split-adjusted for compatibility with the existing feature
pipeline; ``Raw*`` columns preserve provider values needed for audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.panel.snapshots import (
    LicenseNotAcknowledged,
    PanelValidationError,
    load_snapshot,
    require_license_acknowledged,
    validate_ohlcv,
)

from .universe_v8 import verify_universe_manifest
from .v8_protocol import V8_PROTOCOL_VERSION_NEWS, V8_PROTOCOL_VERSION_NUMERIC

V8_MARKET_SNAPSHOT_SCHEMA = 2
V8_MARKET_EXTRA_KEY = "v8_market"
V8_MARKET_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RawOpen",
    "RawHigh",
    "RawLow",
    "RawClose",
    "AdjustedClose",
    "Dividends",
    "SplitFactor",
)
_VALID_PROTOCOLS = {V8_PROTOCOL_VERSION_NEWS, V8_PROTOCOL_VERSION_NUMERIC}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_v8_market_csv(frame: pd.DataFrame) -> str:
    """Serialize the complete v8 market-data contract deterministically."""

    missing = sorted(set(V8_MARKET_COLUMNS) - set(frame.columns))
    if missing:
        raise PanelValidationError("v8 market frame missing columns: " + ", ".join(missing))
    ordered = frame[list(V8_MARKET_COLUMNS)].sort_index().copy()
    ordered.index = pd.DatetimeIndex(ordered.index).strftime("%Y-%m-%d")
    ordered.index.name = "Date"
    return ordered.to_csv(index=True, float_format="%.10f", lineterminator="\n")


def normalize_v8_provider_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw provider OHLC/actions into the frozen v8 column contract."""

    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PanelValidationError("raw provider frame missing columns: " + ", ".join(missing))
    source = frame.sort_index().copy()
    source = source.loc[~source.index.duplicated(keep=False)]
    source = source.dropna(subset=list(required))
    if source.empty:
        raise PanelValidationError("raw provider frame has no complete rows")
    raw_close = source["Close"].to_numpy(dtype=float)
    adjusted_close = source["Adj Close"].to_numpy(dtype=float)
    if not np.isfinite(raw_close).all() or (raw_close <= 0).any():
        raise PanelValidationError("raw provider close contains invalid values")
    factor = adjusted_close / raw_close
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise PanelValidationError("provider adjustment factor contains invalid values")

    index = pd.DatetimeIndex(source.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    normalized = pd.DataFrame(index=index)
    for output, raw in (("Open", "Open"), ("High", "High"), ("Low", "Low")):
        normalized[output] = source[raw].to_numpy(dtype=float) * factor
    normalized["Close"] = adjusted_close
    normalized["Volume"] = source["Volume"].to_numpy(dtype=float)
    normalized["RawOpen"] = source["Open"].to_numpy(dtype=float)
    normalized["RawHigh"] = source["High"].to_numpy(dtype=float)
    normalized["RawLow"] = source["Low"].to_numpy(dtype=float)
    normalized["RawClose"] = raw_close
    normalized["AdjustedClose"] = adjusted_close
    normalized["Dividends"] = (
        source["Dividends"].to_numpy(dtype=float) if "Dividends" in source else 0.0
    )
    normalized["SplitFactor"] = (
        source["Stock Splits"].to_numpy(dtype=float) if "Stock Splits" in source else 0.0
    )
    values = normalized[list(V8_MARKET_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PanelValidationError("normalized v8 market frame contains non-finite values")
    if (normalized[["RawOpen", "RawHigh", "RawLow", "RawClose"]] <= 0).any().any():
        raise PanelValidationError("normalized v8 market frame contains non-positive raw prices")
    validate_ohlcv("provider-frame", normalized)
    return normalized


def _universe_maps(universe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(member["ticker"]).strip().upper(): member
        for member in universe["members"]
        if isinstance(member, dict)
    }


def build_v8_market_snapshot(
    frames: Mapping[str, pd.DataFrame],
    *,
    universe_manifest: dict[str, Any],
    provider: str,
    provider_snapshot_id: str,
    provider_license_id: str,
    license_acknowledged: bool,
    v8_protocol_version: str = V8_PROTOCOL_VERSION_NUMERIC,
    allow_incomplete_diagnostic: bool = False,
    derived_from_panel_checksum: str | None = None,
) -> dict[str, Any]:
    """Validate frames and assemble the complete content-addressed manifest."""

    if not license_acknowledged:
        raise LicenseNotAcknowledged("v8 market snapshot requires explicit license acknowledgement")
    universe = verify_universe_manifest(universe_manifest)
    if v8_protocol_version not in _VALID_PROTOCOLS:
        raise ValueError(f"unsupported v8 protocol {v8_protocol_version!r}")
    for field_name, value in (
        ("provider", provider),
        ("provider_snapshot_id", provider_snapshot_id),
        ("provider_license_id", provider_license_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} is required")

    members = _universe_maps(universe)
    normalized_frames = {str(ticker).strip().upper(): frame for ticker, frame in frames.items()}
    unknown = sorted(set(normalized_frames) - set(members))
    if unknown:
        raise PanelValidationError(
            "market snapshot contains securities outside universe: " + ", ".join(unknown)
        )
    missing_tickers = sorted(set(members) - set(normalized_frames))
    incomplete_history: list[str] = []
    legacy_adjusted_only: list[str] = []
    ticker_metadata: dict[str, Any] = {}
    pooled = hashlib.sha256()
    for ticker in sorted(normalized_frames):
        frame = normalized_frames[ticker]
        provenance = validate_ohlcv(ticker, frame)
        has_complete_v8_schema = set(V8_MARKET_COLUMNS).issubset(frame.columns)
        text = (
            canonical_v8_market_csv(frame)
            if has_complete_v8_schema
            else _legacy_canonical_csv(frame)
        )
        if not has_complete_v8_schema:
            legacy_adjusted_only.append(ticker)
        required_rows = int(members[ticker]["required_history_sessions"])
        if len(frame) < required_rows:
            incomplete_history.append(ticker)
        checksum = _sha256_bytes(text.encode("utf-8"))
        ticker_metadata[ticker] = {
            "security_id": members[ticker]["security_id"],
            "primary_exchange_mic": members[ticker]["primary_exchange_mic"],
            "currency": members[ticker]["currency"],
            "timezone": members[ticker]["timezone"],
            "rows": provenance.rows,
            "required_history_sessions": required_rows,
            "start": provenance.start,
            "end": provenance.end,
            "checksum": checksum,
            "data_contract": (
                "raw_plus_adjusted_ohlcv_actions_v2"
                if has_complete_v8_schema
                else "legacy_adjusted_ohlcv_v1"
            ),
            "suspicious_adjustment_rows": provenance.suspicious_adjustment_rows,
        }
        pooled.update(ticker.encode("utf-8"))
        pooled.update(b"\0")
        pooled.update(checksum.encode("ascii"))

    reasons: list[str] = []
    if not universe.get("coverage_certifiable"):
        reasons.append("universe_not_certifiable")
    if missing_tickers:
        reasons.append("missing_universe_tickers")
    if incomplete_history:
        reasons.append("insufficient_history")
    if legacy_adjusted_only:
        reasons.append("raw_and_adjusted_history_not_preserved")
    coverage_certifiable = not reasons
    if reasons and not allow_incomplete_diagnostic:
        raise PanelValidationError("v8 market snapshot is not certifiable: " + "; ".join(reasons))

    pooled_digest = pooled.hexdigest()
    return {
        "schema_version": 1,
        "panel_id": f"panel-v8-{pooled_digest[:16]}",
        "pooled_checksum": f"sha256:{pooled_digest}",
        "provider": provider,
        "provider_snapshot_id": provider_snapshot_id,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "timezone": "per_security",
        "adjust_mode": "model_ohlc_adjusted; raw_ohlc_and_actions_preserved",
        "license": {
            "acknowledged": True,
            "license_id": provider_license_id,
            "derived_weight_redistribution_requires_review": True,
        },
        "tickers": ticker_metadata,
        "ticker_count": len(ticker_metadata),
        V8_MARKET_EXTRA_KEY: {
            "schema": V8_MARKET_SNAPSHOT_SCHEMA,
            "v8_protocol_version": v8_protocol_version,
            "universe_manifest_sha256": universe["sha256"],
            "requested_ticker_count": len(members),
            "acquired_ticker_count": len(ticker_metadata),
            "missing_tickers": missing_tickers,
            "incomplete_history_tickers": sorted(incomplete_history),
            "legacy_adjusted_only_tickers": sorted(legacy_adjusted_only),
            "coverage_certifiable": coverage_certifiable,
            "coverage_reasons": reasons,
            "derived_from_panel_checksum": derived_from_panel_checksum,
        },
    }


def write_v8_market_snapshot(
    root: Path,
    frames: Mapping[str, pd.DataFrame],
    **manifest_kwargs: Any,
) -> Path:
    """Write a complete immutable v8 snapshot using atomic directory promotion."""

    manifest = build_v8_market_snapshot(frames, **manifest_kwargs)
    root.mkdir(parents=True, exist_ok=True)
    target = root / manifest["panel_id"]
    if target.exists():
        raise FileExistsError(f"v8 market snapshot already exists at {target}")
    temporary = Path(tempfile.mkdtemp(prefix=".v8-market-", dir=root))
    try:
        raw_dir = temporary / "raw"
        raw_dir.mkdir()
        normalized_frames = {str(ticker).strip().upper(): frame for ticker, frame in frames.items()}
        for ticker, metadata in sorted(manifest["tickers"].items()):
            frame = normalized_frames[ticker]
            text = (
                canonical_v8_market_csv(frame)
                if metadata["data_contract"] == "raw_plus_adjusted_ohlcv_actions_v2"
                else _legacy_canonical_csv(frame)
            )
            path = raw_dir / f"{ticker}.csv"
            path.write_bytes(text.encode("utf-8"))
            if _sha256_bytes(path.read_bytes()) != metadata["checksum"]:
                raise RuntimeError(f"checksum mismatch writing {ticker}")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def _legacy_canonical_csv(frame: pd.DataFrame) -> str:
    from backend.panel.snapshots import canonical_csv

    return canonical_csv(frame)


def verify_v8_market_snapshot(
    snapshot_dir: Path,
    *,
    universe_manifest: dict[str, Any],
    require_certifiable: bool = True,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Verify files, universe binding, identities, and v8 coverage metadata."""

    universe = verify_universe_manifest(universe_manifest)
    manifest, frames = load_snapshot(snapshot_dir)
    v8_market = manifest.get(V8_MARKET_EXTRA_KEY)
    if not isinstance(v8_market, dict) or v8_market.get("schema") != V8_MARKET_SNAPSHOT_SCHEMA:
        raise PanelValidationError("panel is not a v8 market snapshot")
    if v8_market.get("universe_manifest_sha256") != universe["sha256"]:
        raise PanelValidationError("v8 market snapshot universe identity mismatch")
    members = _universe_maps(universe)
    if set(manifest.get("tickers", {})) - set(members):
        raise PanelValidationError("v8 market snapshot includes unknown securities")
    for ticker, metadata in manifest["tickers"].items():
        member = members[ticker]
        for field_name in ("security_id", "primary_exchange_mic", "currency", "timezone"):
            if metadata.get(field_name) != member[field_name]:
                raise PanelValidationError(
                    f"{ticker}: v8 market {field_name} differs from universe"
                )
    if require_certifiable:
        if not universe.get("coverage_certifiable"):
            raise PanelValidationError("v8 universe is not certifiable")
        if v8_market.get("coverage_certifiable") is not True:
            raise PanelValidationError("v8 market snapshot is diagnostic-only")
        if set(frames) != set(members):
            raise PanelValidationError("v8 market snapshot does not cover the complete universe")
    return manifest, frames


def fetch_v8_market_frames(
    universe_manifest: dict[str, Any],
    *,
    years: int,
) -> dict[str, pd.DataFrame]:
    """Download raw/action histories and normalize them for local research."""

    require_license_acknowledged()
    if years < 3:
        raise ValueError("v8 market acquisition requires at least three years")
    universe = verify_universe_manifest(universe_manifest)
    tickers = sorted(_universe_maps(universe))
    import yfinance as yf  # type: ignore[import-untyped]

    downloaded = yf.download(
        tickers,
        period=f"{years}y",
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        block = downloaded[ticker] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
        if block is None or block.empty:
            continue
        try:
            normalized = normalize_v8_provider_frame(block)
        except PanelValidationError:
            continue
        frames[ticker] = normalized
    if not frames:
        raise PanelValidationError("provider returned no usable v8 market histories")
    return frames
