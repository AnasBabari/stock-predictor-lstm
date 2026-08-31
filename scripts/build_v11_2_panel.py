#!/usr/bin/env python3
"""Build the V11.2 panel NPZ from an immutable OHLCV snapshot.

This adapter is deliberately offline: it never downloads data and refuses a
current-constituent list in place of the audited PIT64 manifest.  The output
contains one causal 60-session deployable-v5 feature window per security and
origin, with cumulative four-horizon variance/return targets.  It is the
bridge between ``backend.panel.snapshots`` and the V11.2 sealing command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5  # noqa: E402
from backend.panel.snapshots import load_snapshot  # noqa: E402
from research.volatility_forecasting.contracts import VolatilityForecastProtocol  # noqa: E402
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.v11_2_attestation import (  # noqa: E402
    AttestationError,
    verify_v11_2_inputs,
)
from research.volatility_forecasting.v11_2_protocol import (  # noqa: E402
    V11_2_HORIZONS,
    V11_2_PROTOCOL_ID,
    V112Protocol,
    feature_schema_digest,
)
from research.volatility_forecasting.v11_2_universe import (  # noqa: E402
    V112UniverseManifest,
    load_universe_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_snapshot_manifest(path: Path) -> Path:
    """Resolve either an immutable snapshot directory or its parent."""

    direct = path / "manifest.json"
    if direct.is_file():
        return direct
    children = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "manifest.json").is_file()
    )
    if len(children) != 1:
        raise ValueError("snapshot directory must contain exactly one content-addressed snapshot")
    return children[0] / "manifest.json"


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _evidence_args(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise SystemExit("evidence must use NAME=PATH")
        normalized = name.strip()
        if normalized in parsed:
            raise SystemExit(f"duplicate evidence name: {normalized}")
        parsed[normalized] = Path(path).resolve()
    return parsed


def _snapshot_manifest(path: Path) -> tuple[Path, dict[str, object], dict[str, pd.DataFrame]]:
    resolved = _resolve_snapshot_manifest(path)
    return resolved, *load_snapshot(resolved.parent)


def _security_frames(
    frames: dict[str, pd.DataFrame], manifest: V112UniverseManifest
) -> dict[str, pd.DataFrame]:
    """Resolve aliases and membership intervals into one frame per security."""
    output: dict[str, pd.DataFrame] = {}
    for security in manifest.securities:
        pieces: list[pd.DataFrame] = []
        for interval in security.ticker_intervals:
            source = frames.get(interval.ticker.upper())
            if source is None:
                continue
            frame = source.copy()
            dates = pd.DatetimeIndex(frame.index).date
            keep = [
                security.is_member(day.isoformat())
                and security.ticker_at(day.isoformat()) == interval.ticker.upper()
                for day in dates
            ]
            if any(keep):
                pieces.append(frame.loc[keep])
        if not pieces:
            raise ValueError(f"no point-in-time OHLCV rows found for {security.security_id}")
        combined = pd.concat(pieces).sort_index()
        if combined.index.has_duplicates:
            combined = combined[~combined.index.duplicated(keep="first")]
        if not combined.index.is_monotonic_increasing:
            raise ValueError(f"{security.security_id}: resolved dates are not chronological")
        output[security.security_id] = combined
    return output


def _write_manifest(
    output_path: Path,
    *,
    input_snapshot_sha256: str,
    universe: V112UniverseManifest,
    rows: int,
    dates: list[str],
    panel_sha256: str,
    attestation_summary: dict[str, object] | None = None,
) -> Path:
    payload = {
        "protocol_id": V11_2_PROTOCOL_ID,
        "feature_schema_version": "deployable_v5",
        "feature_names": list(DEPLOYABLE_FEATURE_COLUMNS_V5),
        "schema_sha256": feature_schema_digest(V112Protocol()),
        "horizons": list(V11_2_HORIZONS),
        "window_size": 60,
        "snapshot_manifest_sha256": input_snapshot_sha256,
        "panel_sha256": panel_sha256,
        "universe_manifest_sha256": universe.manifest_sha256,
        "certification_eligible": universe.certification_eligible,
        "universe_size": universe.universe_size,
        "stock_origin_observations": rows,
        "unique_sessions": len(set(dates)),
        "date_span": [min(dates), max(dates)],
    }
    if attestation_summary:
        payload["attestation_summary"] = attestation_summary
    output = output_path.with_suffix(output_path.suffix + ".manifest.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market-attestation", type=Path)
    parser.add_argument("--market-public-key", type=Path)
    parser.add_argument("--pit64-attestation", type=Path)
    parser.add_argument("--pit64-public-key", type=Path)
    parser.add_argument("--market-evidence", action="append", default=[])
    parser.add_argument("--pit64-evidence", action="append", default=[])
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable panel output: {output}")
    snapshot_dir = args.snapshot_dir.resolve()
    if _under(snapshot_dir, ROOT / "data" / "ndx100" / "cache"):
        raise SystemExit("the secondary data/ndx100/cache cannot be used for V11.2")
    snapshot_manifest_path, snapshot_manifest, frames = _snapshot_manifest(snapshot_dir)
    license_payload = snapshot_manifest.get("license")
    if not isinstance(license_payload, dict) or license_payload.get("acknowledged") is not True:
        raise SystemExit("snapshot license acknowledgement is missing")
    universe = load_universe_manifest(args.universe_manifest.resolve())
    if universe.protocol_id != V11_2_PROTOCOL_ID:
        raise SystemExit("universe manifest protocol does not match V11.2")
    allowed_aliases = {
        interval.ticker.upper()
        for security in universe.securities
        for interval in security.ticker_intervals
    }
    unknown_snapshot_tickers = sorted(set(frames) - allowed_aliases)
    if unknown_snapshot_tickers:
        raise SystemExit(
            "snapshot contains ticker aliases outside the audited PIT64 universe: "
            + ", ".join(unknown_snapshot_tickers)
        )
    attestation_summary: dict[str, object] = {}
    if universe.certification_eligible:
        required = (
            args.market_attestation,
            args.market_public_key,
            args.pit64_attestation,
            args.pit64_public_key,
        )
        if any(path is None for path in required):
            raise SystemExit(
                "certification-eligible panels require two signed attestations and pinned keys"
            )

        try:
            attestation_summary = verify_v11_2_inputs(
                snapshot_manifest_path=snapshot_manifest_path,
                universe_manifest_path=args.universe_manifest.resolve(),
                market_receipt_path=args.market_attestation,  # type: ignore[arg-type]
                market_public_key_path=args.market_public_key,  # type: ignore[arg-type]
                pit64_receipt_path=args.pit64_attestation,  # type: ignore[arg-type]
                pit64_public_key_path=args.pit64_public_key,  # type: ignore[arg-type]
                market_evidence_files=_evidence_args(args.market_evidence),
                pit64_evidence_files=_evidence_args(args.pit64_evidence),
            )
        except (AttestationError, OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"signed V11.2 input attestation failed: {exc}") from exc
    resolved = _security_frames(frames, universe)
    protocol = VolatilityForecastProtocol(
        horizons=V11_2_HORIZONS,
        feature_names=tuple(DEPLOYABLE_FEATURE_COLUMNS_V5),
        window_size=60,
    )
    examples = build_volatility_panel_examples(resolved, protocol, minimum_har_history=60)
    if examples.horizons != V11_2_HORIZONS:
        raise SystemExit("panel adapter produced an unexpected horizon contract")
    dates = [pd.Timestamp(value).date().isoformat() for value in examples.origin_dates]
    if not dates:
        raise SystemExit("panel adapter produced no origin observations")
    security_ids = [str(value) for value in examples.tickers]
    if len(set(security_ids)) != universe.universe_size:
        raise SystemExit("panel adapter did not produce all 64 accepted securities")
    if len(set(dates)) < 200:
        raise SystemExit("panel adapter produced fewer than 200 unique market sessions")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dates=np.asarray(dates, dtype="U10"),
        security_ids=np.asarray(security_ids, dtype="U128"),
        features=np.asarray(examples.features, dtype=np.float32),
        returns=np.asarray(examples.cumulative_returns, dtype=np.float32),
        rv=np.asarray(examples.realized_variance, dtype=np.float32),
        feature_names=np.asarray(DEPLOYABLE_FEATURE_COLUMNS_V5, dtype="U64"),
        horizons=np.asarray(V11_2_HORIZONS, dtype=np.int64),
    )
    sidecar = _write_manifest(
        output,
        input_snapshot_sha256=_sha256(snapshot_manifest_path),
        universe=universe,
        rows=len(dates),
        dates=dates,
        panel_sha256=_sha256(output),
        attestation_summary=attestation_summary,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(sidecar),
                "universe_manifest_sha256": universe.manifest_sha256,
                "certification_eligible": universe.certification_eligible,
                "stock_origin_observations": len(dates),
                "unique_sessions": len(set(dates)),
                "security_count": len(set(security_ids)),
                "feature_count": len(DEPLOYABLE_FEATURE_COLUMNS_V5),
                "horizons": list(V11_2_HORIZONS),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
