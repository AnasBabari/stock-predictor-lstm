import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "research"), str(ROOT_DIR / "backend")])

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from research.ndx100.data import load_ticker_history  # noqa: E402
from research.volatility_forecasting.architecture_ablation import (  # noqa: E402
    garch_coverage_diagnostics,
    reset_garch_diagnostics,
    run_architecture_ablation,
    select_numeric_champion,
)
from research.volatility_forecasting.contracts import VolatilityForecastProtocol  # noqa: E402
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.folds import VolatilityFold  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "CSCO",
    "ADBE",
    "NFLX",
    "AMD",
    "QCOM",
    "INTC",
    "TXN",
    "AVGO",
    "COST",
    "PEP",
    "TMUS",
    "AMAT",
    "ISRG",
    "CMCSA",
    "HON",
    "AMGN",
    "INTU",
)
HORIZONS = (1, 3, 5, 7, 14, 30)
SEEDS = (41, 42, 43)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V9 Architecture Ablation")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke run with fewer stocks")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Running on device: %s (CUDA: %s)", device, torch.cuda.is_available())

    # Deterministic representative sampling: hashing avoids the "first N tickers"
    # bias by ticker ordering, capitalization, or listing age.
    ordered = sorted(DEFAULT_TICKERS)
    if args.smoke:
        tickers = sorted(ordered, key=lambda t: hashlib.sha256(t.encode()).hexdigest())[:8]
        logger.warning(
            "SMOKE RUN: %d hash-sampled tickers. Output is engineering evidence "
            "only and is never a benchmark or certification result.",
            len(tickers),
        )
    else:
        tickers = ordered
    logger.info("Building panel examples for %d tickers...", len(tickers))

    panel: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = load_ticker_history(t)
        if df is not None and not df.empty:
            panel[t] = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

    protocol = VolatilityForecastProtocol(horizons=HORIZONS)
    examples = build_volatility_panel_examples(panel, protocol)
    reset_garch_diagnostics()
    logger.info(
        "Panel examples built: %s, horizons: %s, date range: %s to %s",
        examples.features.shape,
        examples.horizons,
        str(examples.origin_dates.min()),
        str(examples.origin_dates.max()),
    )

    # Construct development expanding fold: train through 2024, validate on 2025/2026 dev
    # (Sealed test partitions are completely excluded and untouched)
    dates = pd.to_datetime(examples.origin_dates)
    train_mask = dates < "2025-01-01"
    val_mask = (dates >= "2025-01-01") & (dates <= "2026-06-30")

    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)

    fold = VolatilityFold(
        fold=1,
        train_indices=train_indices,
        validation_indices=val_indices,
        train_end=examples.origin_dates[train_indices[-1]],
        validation_start=examples.origin_dates[val_indices[0]],
        validation_end=examples.origin_dates[val_indices[-1]],
    )
    logger.info(
        "Fold 1: train rows: %d (%s), val rows: %d (%s to %s)",
        len(fold.train_indices),
        fold.train_end,
        len(fold.validation_indices),
        fold.validation_start,
        fold.validation_end,
    )

    t0 = time.perf_counter()
    df_results = run_architecture_ablation(
        examples=examples,
        fold=fold,
        seeds=SEEDS,
        device=device,
    )
    duration = time.perf_counter() - t0
    logger.info("Architecture ablation completed in %.2f seconds", duration)

    # Descriptive summaries only. Ranking here is DIAGNOSTIC and must never be
    # used to pick a winner: an average across horizons can hide a loss at a
    # required horizon.
    summary_by_family = (
        df_results.groupby("family")
        .agg(
            mean_qlike=("mean_qlike", "mean"),
            relative_ratio=("relative_qlike_ratio", "mean"),
            p05=("bootstrap_p05", "mean"),
            p95=("bootstrap_p95", "mean"),
            low_vol_ratio=("low_vol_ratio", "mean"),
            normal_vol_ratio=("normal_vol_ratio", "mean"),
            high_vol_ratio=("high_vol_ratio", "mean"),
            avg_duration=("duration_seconds", "mean"),
        )
        .sort_values("relative_ratio")
        .reset_index()
    )

    print("\n" + "=" * 90)
    print(
        "V9 Architecture Ablation — DESCRIPTIVE summary "
        "(averaged across horizons and seeds; NOT a selection)"
    )
    print("=" * 90)
    print(
        summary_by_family.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )
    print("=" * 90 + "\n")

    summary_by_horizon = (
        df_results.groupby(["horizon", "family"])["relative_qlike_ratio"]
        .mean()
        .unstack(level="family")
    )
    print("Relative QLIKE Ratio vs HAR by Horizon (lower is better, < 1.0 beats HAR):")
    print(summary_by_horizon.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n")

    # Selection is delegated to the gated selector. It requires every required
    # horizon {1,3,5,7}, five folds, and a bootstrap upper bound below the
    # no-skill threshold. With a single fold it correctly returns HAR.
    decision = select_numeric_champion(df_results)
    print(f"Selected family      : {decision.selected_family}")
    print(f"Selection state      : {decision.selection_state}")
    print(f"Required horizons    : {list(decision.required_horizons)}")
    print(f"Eligible families    : {list(decision.eligible_families)}")
    for fam, reasons in sorted(decision.reasons_by_family.items()):
        status = "ELIGIBLE" if not reasons else "REJECTED"
        print(f"  {status:<9} {fam:<18} {'; '.join(reasons)}")
    print()

    garch_diag = garch_coverage_diagnostics()
    if garch_diag:
        print(f"GARCH/GJR substitutions recorded: {len(garch_diag)}")
        for item in garch_diag[:10]:
            print(
                f"  {item['family']} {item['ticker']}: {item['reason']} -> {item['substituted_with']}"
            )
        print()

    results_dir = Path("research/results/ndx100-v9")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_file = results_dir / "ablation_summary.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_role": "development_diagnostic_only",
        "certification_eligible": False,
        "selected_family": decision.selected_family,
        "selection_state": decision.selection_state,
        "eligible_families": list(decision.eligible_families),
        "required_horizons": list(decision.required_horizons),
        "reasons_by_family": {k: list(v) for k, v in decision.reasons_by_family.items()},
        "folds_present": sorted(int(v) for v in df_results["fold"].unique()),
        "families_evaluated": sorted(str(v) for v in df_results["family"].unique()),
        "garch_substitutions": list(garch_diag),
        "device": str(device),
        "total_duration_seconds": duration,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "family_summary": summary_by_family.to_dict(orient="records"),
        "by_horizon": summary_by_horizon.to_dict(),
    }
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    report_file = Path("reports/V9_ARCHITECTURE_ABLATION_REPORT.md")
    report_md = f"""# V9 Architecture Ablation — development diagnostic

> **Evidence role:** development diagnostic only.
> **Certification eligible:** no.
> **Sealed test:** not opened.

## Selection outcome

- **Selected family:** `{decision.selected_family}`
- **Selection state:** `{decision.selection_state}`
- **Required horizons:** {list(decision.required_horizons)}
- **Eligible families:** {list(decision.eligible_families)}
- **Folds present:** {sorted(int(v) for v in df_results["fold"].unique())}

Selection requires positive skill at *every* required horizon, across five
development folds, with a bootstrap upper bound below the no-skill threshold.
HAR is retained whenever no learned candidate clears every gate, and that
outcome is reported as HAR.

## Rejection reasons

| Family | Status | Reasons |
| :--- | :--- | :--- |
"""
    for fam, reasons in sorted(decision.reasons_by_family.items()):
        status = "ELIGIBLE" if not reasons else "REJECTED"
        report_md += f"| `{fam}` | {status} | {'; '.join(reasons) or '—'} |\n"

    report_md += f"""
## Descriptive family summary (not a ranking used for selection)

{summary_by_family.to_string(index=False, float_format=lambda v: f"{v:.4f}")}

## Relative QLIKE ratio vs HAR by horizon

```text
{summary_by_horizon.to_string(float_format=lambda v: f"{v:.4f}")}
```

## GARCH/GJR coverage

{len(garch_diag)} ticker-level substitution(s) recorded. Every substitution is
listed; a HAR substitute is never silently scored as a fitted GARCH forecast.

```
{garch_diag if garch_diag else "none"}
```
"""
    report_file.write_text(report_md, encoding="utf-8")
    logger.info("Report written to %s", report_file)

    if decision.selected_family != "har":
        logger.info(
            "Learned candidate %s cleared every development gate. Freezing is still "
            "disabled until a family-specific serializer exists.",
            decision.selected_family,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
