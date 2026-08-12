"""Run the final multi-window block-bootstrap holdout confirmation on a frozen CSV snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add research directory to Python path if needed
research_dir = Path(__file__).resolve().parent.parent / "research"
if str(research_dir) not in sys.path:
    sys.path.insert(0, str(research_dir))

from stock_autoresearch.candidates import (  # noqa: E402 - imported after research path setup
    CompactMLPCandidate,
    DLinearCandidate,
    ElasticNetCandidate,
    PersistenceCandidate,
    RidgeCandidate,
    SmallTCNCandidate,
    elastic_net_family_factories,
)
from stock_autoresearch.holdout import (  # noqa: E402 - imported after research path setup
    evaluate_multi_window_holdout,
)


def get_factory(family: str):
    if family == "persistence":
        return lambda seed: PersistenceCandidate()
    elif family == "ridge":
        return lambda seed: RidgeCandidate()
    elif family == "elastic_net":
        return lambda seed: ElasticNetCandidate()
    elif family == "compact_mlp":
        return lambda seed: CompactMLPCandidate()
    elif family == "dlinear":
        return lambda seed: DLinearCandidate()
    elif family == "small_tcn":
        return lambda seed: SmallTCNCandidate()
    elif family in elastic_net_family_factories():
        return elastic_net_family_factories()[family]
    else:
        raise ValueError(f"Unknown candidate family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stock Autoresearch multi-window holdout confirmation."
    )
    parser.add_argument("snapshot", type=Path, help="Path to frozen CSV market snapshot.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon (default: 5).")
    parser.add_argument("--family", required=True, help="Candidate family to evaluate.")
    args = parser.parse_args()

    if not args.snapshot.exists():
        print(f"Error: Snapshot file {args.snapshot} does not exist.", file=sys.stderr)
        return 1

    frame = pd.read_csv(args.snapshot, index_col=0, parse_dates=True)

    try:
        factory = get_factory(args.family)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(
        f"Running multi-window holdout for '{args.family}' on {args.snapshot.name} (h={args.horizon})..."
    )
    try:
        res = evaluate_multi_window_holdout(frame, factory, horizon=args.horizon)
    except Exception as e:
        print(f"Error during holdout evaluation: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 50)
    print("HOLDOUT VERDICT:", res.verdict())
    print("=" * 50)
    print(f"Pooled Relative RMSE: {res.pooled_relative_rmse:.5f}")
    print(f"Pooled RMSE 95% CI: [{res.rmse_ci['lower']:.5f}, {res.rmse_ci['upper']:.5f}]")
    print(f"Pooled Relative MAE: {res.pooled_relative_mae:.5f}")
    print(f"Pooled MAE 95% CI: [{res.mae_ci['lower']:.5f}, {res.mae_ci['upper']:.5f}]")
    print(
        f"Windows Passing Gate: {res.windows_passing_gate()} / {res.window_count} (Majority required: {res.majority_required()})"
    )

    print("\nPer-Window Breakdown:")
    for i, w in enumerate(res.windows):
        status = "PASS" if w.passes_gate() else "FAIL"
        print(
            f"  Window {i}: {status} (Rel RMSE: {w.relative_rmse:.5f}, Rel MAE: {w.relative_mae:.5f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
