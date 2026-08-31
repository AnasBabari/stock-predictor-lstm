"""Run the simplified volatility benchmark on immutable per-symbol CSV files.

The command is deliberately offline and writes nothing unless ``--output`` is
provided.  Model selection uses validation QLIKE only; the test partition is
reported once, after selection, and is never used to choose a model.

Example::

    # PowerShell (from the repository root):
    backend/.venv/Scripts/python.exe scripts/run_volatility_benchmark.py `
      --csv snapshot_MSFT.csv --csv snapshot_SPY.csv `
      --horizon 5 --include-lstm --output reports/volatility.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from research.volatility_forecasting.simple_pipeline import (  # noqa: E402
    LSTMConfig,
    VolatilityConfig,
    build_examples,
    chronological_split,
    evaluate_benchmark,
    experiment_metadata,
    select_validation_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", action="append", required=True, help="OHLCV CSV (repeat per symbol)"
    )
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in sessions (e.g. 1, 5, 10, 20)")
    parser.add_argument("--lookback", type=int, default=22)
    parser.add_argument("--include-lstm", action="store_true")
    parser.add_argument("--without-boosting", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _ticker(path: Path) -> str:
    stem = path.stem.upper()
    return stem.split("_", 1)[1] if stem.startswith("SNAPSHOT_") else stem


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV is empty: {path}")
    return frame


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    model_names = sorted({name for result in results for name in result["metrics"]})
    output: dict[str, Any] = {}
    for model in model_names:
        output[model] = {}
        for partition in ("validation", "test"):
            rows = [result["metrics"][model][partition] for result in results]
            numeric: dict[str, Any] = {}
            for key in rows[0]:
                if key == "rows":
                    numeric["rows"] = int(sum(int(row.get("rows", 0)) for row in rows))
                elif isinstance(rows[0][key], (int, float, np.number)):
                    values = [
                        float(row[key])
                        for row in rows
                        if key in row and isinstance(row[key], (int, float, np.number))
                    ]
                    if values:
                        numeric[key] = float(np.mean(values))
            if partition == "test":
                if "volatility_interval" in rows[0] and isinstance(rows[0]["volatility_interval"], dict):
                    covs = [
                        float(row["volatility_interval"]["empirical_coverage"])
                        for row in rows
                        if "volatility_interval" in row
                        and row["volatility_interval"].get("empirical_coverage") is not None
                    ]
                    widths = [
                        float(row["volatility_interval"]["average_width"])
                        for row in rows
                        if "volatility_interval" in row
                        and row["volatility_interval"].get("average_width") is not None
                    ]
                    numeric["volatility_interval"] = {
                        "nominal_coverage": rows[0]["volatility_interval"].get("nominal_coverage"),
                        "empirical_coverage": float(np.mean(covs)) if covs else None,
                        "average_width": float(np.mean(widths)) if widths else None,
                    }
                if "price_cone" in rows[0] and isinstance(rows[0]["price_cone"], dict):
                    covs = [
                        float(row["price_cone"]["empirical_coverage"])
                        for row in rows
                        if "price_cone" in row
                        and row["price_cone"].get("empirical_coverage") is not None
                    ]
                    widths = [
                        float(row["price_cone"]["average_width_pct"])
                        for row in rows
                        if "price_cone" in row
                        and row["price_cone"].get("average_width_pct") is not None
                    ]
                    numeric["price_cone"] = {
                        "nominal_coverage": rows[0]["price_cone"].get("nominal_coverage"),
                        "empirical_coverage": float(np.mean(covs)) if covs else None,
                        "average_width_pct": float(np.mean(widths)) if widths else None,
                    }
            output[model][partition] = numeric
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = VolatilityConfig(horizon=args.horizon, lookback=args.lookback)
    results: list[dict[str, Any]] = []
    for raw_path in args.csv:
        path = Path(raw_path)
        ticker = _ticker(path)
        examples = build_examples(_load(path), config)
        split = chronological_split(
            len(examples.target),
            horizon=config.horizon,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            embargo_sessions=config.embargo,
        )
        metrics = evaluate_benchmark(
            examples,
            split,
            include_boosting=not args.without_boosting,
            include_lstm=args.include_lstm,
            lstm_config=LSTMConfig(seed=config.seed) if args.include_lstm else None,
        )
        selected = select_validation_model(metrics)
        selected_record = experiment_metadata(
            examples,
            config,
            split,
            model=selected,
            metrics=metrics[selected],
            git_commit=_git_commit(),
        )
        results.append(
            {
                "ticker": ticker,
                "source": str(path),
                "selected_model": selected,
                "metrics": metrics,
                "experiment": selected_record,
            }
        )

    report: dict[str, Any] = {
        "report_version": "simple-volatility-benchmark-v1",
        "selection_rule": "minimum validation QLIKE; test is report-only",
        "configuration": config.__dict__,
        "assets": results,
        "aggregate": _aggregate(results),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
