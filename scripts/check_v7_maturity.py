#!/usr/bin/env python3
"""Read-only v7 reserve maturity monitor (fail-closed, never opens the holdout).

This checker proves whether the future temporal reserve required by the frozen
v7 preregistration is mature. It never creates a certification marker, never
trains, and never modifies candidate state. It is safe to run nightly after the
prospective start and before every certification attempt.

Output states (exactly one):
  - not_mature
  - partially_mature
  - mature_but_incomplete_coverage
  - ready_for_one_shot_certification

See docs/VOLATILITY_V7_PREREGISTRATION.md and docs/DEPLOYMENT_GATE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for candidate_path in (ROOT, ROOT / "research"):
    if str(candidate_path) not in sys.path:
        sys.path.insert(0, str(candidate_path))

from volatility_forecasting.cache import (  # noqa: E402
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
)
from volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from volatility_forecasting.folds import build_prospective_certification_fold_plan  # noqa: E402
from volatility_forecasting.prospective import (  # noqa: E402
    ProspectiveCycleSettings,
    prospective_protocol,
)

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v7 maturity monitor (read-only, never opens holdout)"
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        required=True,
        help="Path to the latest immutable panel snapshot directory (contains manifest.json + raw/)",
    )
    parser.add_argument(
        "--development-panel-dir",
        type=Path,
        default=None,
        help="Optional: path to the frozen development panel for prefix drift detection",
    )
    parser.add_argument(
        "--example-cache-root",
        type=Path,
        default=None,
        help="Optional example-cache root (not required, rebuilt if missing)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout in addition to human-readable summary",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless ready_for_one_shot_certification",
    )
    return parser.parse_args()


def _trading_calendar_projection(
    start: np.datetime64,
    trading_days: int,
) -> str | None:
    """Project calendar date by adding trading_days sessions using NYSE calendar.

    Falls back to simple weekday approximation if pandas_market_calendars unavailable.
    """
    try:
        import pandas_market_calendars as mcal

        calendar = mcal.get_calendar("NYSE")
        start_ts = pd.Timestamp(str(start))
        # Search window wide enough for weekends/holidays
        horizon = max(trading_days * 4 + 30, 400)
        schedule = calendar.schedule(
            start_date=start_ts, end_date=start_ts + pd.Timedelta(days=horizon)
        )
        # schedule.index are session opens; filter strictly after start? For projection,
        # we want start + trading_days sessions inclusive/exclusive matching prereg spec:
        # start is the first certifiable origin; the 252nd is start +251 sessions;
        # the target-complete date adds max_horizon sessions.
        sessions = schedule.index
        # Find first session >= start
        future_sessions = [d for d in sessions if d.date() >= start_ts.date()]
        if len(future_sessions) >= trading_days:
            target = future_sessions[trading_days - 1]
            return target.strftime("%Y-%m-%d")
    except Exception:
        pass
    # Fallback: add trading days approximating 5 per 7 calendar days
    try:
        # Rough: 252 trading days ≈ 352 calendar days (including weekends/holidays)
        # Use numpy busday_offset if available
        candidate = np.busday_offset(start, trading_days - 1, roll="forward")
        return str(candidate)
    except Exception:
        return None


def main() -> int:
    args = _parse_args()
    cycle = ProspectiveCycleSettings()
    protocol = prospective_protocol()
    panel_dir = args.panel_dir.resolve()
    development_panel_dir = (
        args.development_panel_dir.resolve() if args.development_panel_dir else None
    )

    if not panel_dir.exists():
        print(f"panel directory does not exist: {panel_dir}", file=sys.stderr)
        return 2

    # Load panel manifests for dates and provenance
    try:
        development_fingerprint = None
        if development_panel_dir is not None:
            if not development_panel_dir.exists():
                print(f"development panel missing: {development_panel_dir}", file=sys.stderr)
                return 2
            development_fingerprint = panel_fingerprint(development_panel_dir)
    except Exception as error:
        print(f"development panel fingerprint failed: {error}", file=sys.stderr)
        return 2

    try:
        current_fingerprint = (
            panel_fingerprint(panel_dir) if (panel_dir / "manifest.json").exists() else None
        )
    except Exception:
        current_fingerprint = None

    # Verify immutable prefix if development panel supplied (read-only check)
    prefix_ok: bool | None = None
    prefix_detail: str | None = None
    if development_panel_dir is not None:
        try:
            from backend.panel.snapshots import canonical_csv

            dev_panel = load_panel_from_directory(development_panel_dir)
            cert_panel = load_panel_from_directory(panel_dir)
            cutoff = np.datetime64(cycle.development_cutoff, "D")
            # Check universe equality
            if set(dev_panel) != set(cert_panel):
                prefix_ok = False
                prefix_detail = (
                    f"ticker universe differs dev={len(dev_panel)} cert={len(cert_panel)}"
                )
            else:
                for ticker in sorted(dev_panel):
                    dev_prefix = dev_panel[ticker].loc[
                        np.asarray(dev_panel[ticker].index, dtype="datetime64[D]") <= cutoff
                    ]
                    cert_prefix = cert_panel[ticker].loc[
                        np.asarray(cert_panel[ticker].index, dtype="datetime64[D]") <= cutoff
                    ]
                    if canonical_csv(dev_prefix) != canonical_csv(cert_prefix):
                        prefix_ok = False
                        prefix_detail = f"immutable prefix drift for {ticker}"
                        break
                else:
                    prefix_ok = True
                    prefix_detail = "immutable prefix preserved"
        except Exception as error:
            prefix_ok = False
            prefix_detail = f"prefix check failed: {error}"

    # Load panel and build examples (target-complete origins only)
    try:
        panel = load_panel_from_directory(panel_dir)
    except Exception as error:
        print(f"failed to load panel: {error}", file=sys.stderr)
        return 2

    # Derive ticker universe from manifest if available
    manifest_path = panel_dir / "manifest.json"
    if manifest_path.exists():
        try:
            _ = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Build examples to get target-complete origin_dates (cache-accelerated)
    examples = None
    # Try to load from example cache if available (10-100x faster)
    candidate_cache_roots: list[Path] = []
    if args.example_cache_root:
        candidate_cache_roots.append(args.example_cache_root.resolve())
    # Well-known local cache locations from the durable run
    candidate_cache_roots.extend(
        [
            Path(r"C:\tmp\stocklstm-volatility-panel-v1\example-cache"),
            Path(r"C:\Users\Babar\AppData\Local\Temp\stocklstm-volatility-panel-v1\example-cache"),
            ROOT / "research" / ".cache" / "volatility-examples",
            ROOT / ".cache" / "volatility-examples",
        ]
    )
    # Deduplicate
    seen_roots: set[Path] = set()
    deduped_roots: list[Path] = []
    for root in candidate_cache_roots:
        if root not in seen_roots:
            seen_roots.add(root)
            deduped_roots.append(root)

    # Also try the candidate-manifest embedded cache location if present
    try:
        for root in deduped_roots:
            if not root.is_dir():
                continue
            try:
                fp = panel_fingerprint(panel_dir)
            except Exception:
                fp = current_fingerprint or ""
            compatible = find_compatible_example_cache(root, panel_checksum=fp, protocol=protocol)
            if compatible is not None:
                try:
                    examples = load_example_cache(compatible, panel_checksum=fp, protocol=protocol)
                    break
                except Exception:
                    continue
    except Exception:
        examples = None

    if examples is None:
        try:
            examples = build_volatility_panel_examples(panel, protocol)
        except Exception as error:
            print(f"failed to build volatility examples: {error}", file=sys.stderr)
            # If we cannot build examples, we are not_mature by definition
            status = "not_mature"
            result = {
                "status": status,
                "protocol_version": protocol.protocol_version,
                "architecture_version": protocol.architecture_version,
                "development_cutoff": cycle.development_cutoff,
                "certification_start": cycle.prospective_certification_start,
                "required_temporal_sessions": protocol.temporal_holdout_sessions,
                "max_horizon": max(protocol.horizons),
                "error": str(error),
                "prefix_ok": prefix_ok,
                "prefix_detail": prefix_detail,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1 if args.strict else 0

    unique_dates = np.unique(examples.origin_dates)
    unique_dates.sort()
    cutoff = np.datetime64(cycle.development_cutoff, "D")
    cert_start = np.datetime64(cycle.prospective_certification_start, "D")
    development_dates = unique_dates[unique_dates <= cutoff]
    future_dates = unique_dates[unique_dates >= cert_start]
    required_sessions = protocol.temporal_holdout_sessions
    max_horizon = max(protocol.horizons)

    # Per-ticker coverage for future window (read-only, does not open holdout)
    holdout_tickers: tuple[str, ...] = ()
    train_tickers: tuple[str, ...] = ()
    try:
        from volatility_forecasting.folds import select_asset_holdouts

        all_tickers_np = np.array(sorted(panel.keys()))
        train_tickers, holdout_tickers = select_asset_holdouts(
            all_tickers_np,
            fraction=protocol.asset_holdout_fraction,
            seed=42,
            required=("NMM", "MSFT"),
        )
    except Exception:
        pass

    # Compute maturity state
    status: str
    detail: str
    fold_plan_error: str | None = None

    if len(future_dates) == 0:
        status = "not_mature"
        detail = (
            f"no target-complete origins on or after {cycle.prospective_certification_start}; "
            f"need {required_sessions} sessions plus {max_horizon}-session target completion"
        )
    elif len(future_dates) < required_sessions:
        status = "partially_mature"
        detail = (
            f"{len(future_dates)}/{required_sessions} target-complete sessions available "
            f"on/after {cycle.prospective_certification_start}"
        )
    else:
        # We have at least 252 origins - check coverage via prospective certification fold plan
        # This is the exact same validation as certify_prospective_volatility_candidate.py but
        # without creating markers or predictions.
        try:
            fold_plan = build_prospective_certification_fold_plan(
                examples,
                protocol,
                development_cutoff=cutoff,
                prospective_certification_start=cert_start,
            )
            # If we got here, coverage is complete for every locked date and NMM/MSFT
            status = "ready_for_one_shot_certification"
            detail = (
                f"ready: {len(future_dates)} sessions available, "
                f"locked window {fold_plan.certification_start} "
                f"with {len(fold_plan.temporal_certification_indices)} temporal and "
                f"{len(fold_plan.asset_transfer_certification_indices)} asset-transfer rows"
            )
        except ValueError as error:
            msg = str(error)
            if "not mature" in msg or "need" in msg.lower():
                status = "partially_mature"
                detail = msg
            elif (
                "missing" in msg.lower()
                or "incomplete" in msg.lower()
                or "do not cover" in msg.lower()
            ):
                status = "mature_but_incomplete_coverage"
                detail = msg
            else:
                status = "mature_but_incomplete_coverage"
                detail = msg
            fold_plan_error = msg

    # Projected certification-ready date (read-only estimate)
    projected_ready: str | None = _trading_calendar_projection(
        np.datetime64(cycle.prospective_certification_start, "D"),
        required_sessions + max_horizon,
    )
    # Also project per current panel's last origin if partially mature
    # The wall-clock ready date is independent of current panel's last date;
    # it is start + (252+30-1) trading sessions. We report it statically.
    # For partially mature, also estimate remaining trading days
    remaining_sessions = max(0, required_sessions - len(future_dates))

    # Check NMM/MSFT coverage explicitly for human readability
    nmm_msft_complete: bool | None = None
    if holdout_tickers and len(future_dates) >= required_sessions:
        try:
            # Quick check: each required ticker has rows covering every locked date
            locked_end = (
                future_dates[required_sessions - 1]
                if len(future_dates) >= required_sessions
                else None
            )
            if locked_end is not None:
                locked_dates = future_dates[:required_sessions]
                nmm_ok = True
                msft_ok = True
                for ticker in ("NMM", "MSFT"):
                    ticker_mask = examples.tickers == ticker
                    ticker_dates = np.unique(
                        examples.origin_dates[ticker_mask & (examples.origin_dates >= cert_start)]
                    )
                    # Check contains all locked_dates
                    if not np.array_equal(np.intersect1d(ticker_dates, locked_dates), locked_dates):
                        if ticker == "NMM":
                            nmm_ok = False
                        else:
                            msft_ok = False
                nmm_msft_complete = nmm_ok and msft_ok
        except Exception:
            nmm_msft_complete = None

    # Panel last dates
    panel_first: str | None = None
    panel_last: str | None = None
    try:
        panel_first = str(
            min(
                pd.Timestamp(str(idx)).date()
                for frame in panel.values()
                for idx in [frame.index.min()]
            )
        )
        panel_last = str(
            max(
                pd.Timestamp(str(idx)).date()
                for frame in panel.values()
                for idx in [frame.index.max()]
            )
        )
    except Exception:
        pass

    # Gather ticker counts
    ticker_count = len(panel)
    example_rows = len(examples.features) if examples else 0

    result = {
        "status": status,
        "detail": detail,
        "protocol_version": protocol.protocol_version,
        "architecture_version": protocol.architecture_version,
        "development_cutoff": cycle.development_cutoff,
        "certification_start": cycle.prospective_certification_start,
        "required_temporal_sessions": required_sessions,
        "required_horizons": list(cycle.required_horizons),
        "max_horizon": max_horizon,
        "panel_dir": str(panel_dir),
        "panel_fingerprint": current_fingerprint,
        "development_panel_dir": str(development_panel_dir) if development_panel_dir else None,
        "development_fingerprint": development_fingerprint,
        "prefix_ok": prefix_ok,
        "prefix_detail": prefix_detail,
        "panel_first_date": panel_first,
        "panel_last_date": panel_last,
        "ticker_count": ticker_count,
        "holdout_tickers": list(holdout_tickers) if holdout_tickers else None,
        "train_ticker_count": len(train_tickers) if train_tickers else None,
        "example_rows": example_rows,
        "unique_origin_sessions": len(unique_dates),
        "development_sessions": len(development_dates),
        "future_target_complete_sessions": len(future_dates),
        "remaining_sessions_required": remaining_sessions,
        "projected_certification_ready_date": projected_ready,
        "projected_note": (
            f"requires {required_sessions} target-complete origins on/after "
            f"{cycle.prospective_certification_start} plus {max_horizon}-session target completion "
            f"(total {required_sessions + max_horizon} trading sessions beyond start)"
        ),
        "nmm_msft_coverage_complete": nmm_msft_complete,
        "first_future_date": str(future_dates[0]) if len(future_dates) else None,
        "last_future_date": str(future_dates[-1]) if len(future_dates) else None,
        "fold_plan_error": fold_plan_error,
        "one_shot_marker_created": False,
        "holdout_opened": False,
        "candidate_modified": False,
    }

    # Human-readable summary
    print("=" * 72)
    print("V7 MATURITY MONITOR (read-only, never opens holdout)")
    print("=" * 72)
    print(f"Status:              {status}")
    print(f"Detail:              {detail}")
    print(f"Protocol:            {protocol.protocol_version}")
    print(f"Architecture:        {protocol.architecture_version}")
    print(f"Development cutoff:  {cycle.development_cutoff}")
    print(f"Cert start:          {cycle.prospective_certification_start}")
    print(f"Panel dir:           {panel_dir}")
    print(f"Panel fingerprint:   {current_fingerprint or 'unknown'}")
    if development_panel_dir:
        print(f"Dev panel:           {development_panel_dir}")
        print(f"Prefix OK:           {prefix_ok} ({prefix_detail})")
    print(f"Panel range:         {panel_first} .. {panel_last} ({ticker_count} tickers)")
    print(f"Example rows:        {example_rows}")
    print(
        f"Unique origins:      {len(unique_dates)} (dev={len(development_dates)} future={len(future_dates)})"
    )
    print(f"Required sessions:   {required_sessions} (remaining {remaining_sessions})")
    print(f"Max horizon:         {max_horizon}")
    print(
        f"Projected ready:     {projected_ready}  ({required_sessions + max_horizon} sessions after start)"
    )
    print(f"Holdout tickers:     {list(holdout_tickers) if holdout_tickers else 'unknown'}")
    print(f"NMM/MSFT coverage:   {nmm_msft_complete}")
    if len(future_dates):
        print(
            f"Future window:       {future_dates[0]} .. {future_dates[min(required_sessions, len(future_dates)) - 1] if len(future_dates) >= 1 else future_dates[-1]}"
        )
    print("=" * 72)
    if status == "not_mature":
        print("Action: Keep production abstention; no certification attempted.")
    elif status == "partially_mature":
        print(
            "Action: Continue collecting immutable daily snapshots; do not attempt one-shot certification yet."
        )
    elif status == "mature_but_incomplete_coverage":
        print(
            "Action: Panel has enough sessions but per-ticker coverage incomplete; investigate missing tickers/dates."
        )
    elif status == "ready_for_one_shot_certification":
        print(
            "Action: READY - run certify_prospective_volatility_candidate.py exactly once with empty output dir."
        )
    print("=" * 72)
    print("Safety: No marker created, no holdout opened, no candidate modified.")
    print("=" * 72)

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(result, indent=2, sort_keys=True))

    # Return code: 0 for all except strict mode
    if args.strict:
        return 0 if status == "ready_for_one_shot_certification" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
