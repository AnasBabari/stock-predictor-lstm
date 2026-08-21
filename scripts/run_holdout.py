"""Run the final multi-window block-bootstrap holdout confirmation on a frozen CSV snapshot.

Every completed holdout writes a durable evidence record to the experiment
ledger (snapshot content hash, git commit, code hash, window definitions,
per-window metrics, bootstrap CIs, verdict, and the multiplicity policy) so
survivor claims can be reconstructed from committed artifacts alone.
"""

from __future__ import annotations

import argparse
import hashlib
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
    RandomFeaturesRidgeCandidate,
    RidgeCandidate,
    canonical_family,
    elastic_net_family_factories,
)
from stock_autoresearch.holdout import (  # noqa: E402 - imported after research path setup
    evaluate_multi_window_holdout,
)
from stock_autoresearch.ledger import (  # noqa: E402 - imported after research path setup
    MULTIPLICITY_POLICY,
    PROTOCOL_VERSION,
    append_record,
    export_tsv_summary,
    generate_markdown_report,
)

DEFAULT_LEDGER = (
    Path(__file__).resolve().parent.parent / "research" / "results" / "experiments.jsonl"
)


def get_factory(family: str):
    # Legacy family names are accepted at this CLI boundary and mapped to the
    # renamed implementation (small_tcn -> random_features_ridge).
    canonical = canonical_family(family)
    if canonical == "persistence":
        return lambda seed: PersistenceCandidate()
    elif canonical == "ridge":
        return lambda seed: RidgeCandidate()
    elif canonical == "elastic_net":
        return lambda seed: ElasticNetCandidate()
    elif canonical == "compact_mlp":
        return lambda seed: CompactMLPCandidate()
    elif canonical == "dlinear":
        return lambda seed: DLinearCandidate()
    elif canonical == "random_features_ridge":
        return lambda seed: RandomFeaturesRidgeCandidate()
    elif canonical in elastic_net_family_factories():
        return elastic_net_family_factories()[canonical]
    else:
        raise ValueError(f"Unknown candidate family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stock Autoresearch multi-window holdout confirmation."
    )
    parser.add_argument("snapshot", type=Path, help="Path to frozen CSV market snapshot.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon (default: 5).")
    parser.add_argument("--family", required=True, help="Candidate family to evaluate.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="Experiment ledger JSONL to append the evidence record to.",
    )
    parser.add_argument(
        "--window-count", type=int, default=None, help="Override holdout window count."
    )
    parser.add_argument(
        "--window-rows", type=int, default=None, help="Override per-window test origins."
    )
    parser.add_argument(
        "--min-train-rows", type=int, default=None, help="Override minimum training rows."
    )
    parser.add_argument("--window", type=int, default=None, help="Override input window length.")
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

    canonical = canonical_family(args.family)
    print(
        f"Running multi-window holdout for '{args.family}' on {args.snapshot.name} (h={args.horizon})..."
    )
    try:
        overrides = {
            key: value
            for key, value in {
                "window_count": args.window_count,
                "window_rows": args.window_rows,
                "min_train_rows": args.min_train_rows,
                "window": args.window,
            }.items()
            if value is not None
        }
        res = evaluate_multi_window_holdout(frame, factory, horizon=args.horizon, **overrides)
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

    # ── Durable evidence record ─────────────────────────────────────────
    snapshot_digest = hashlib.sha256()
    snapshot_digest.update(frame.to_csv(index=True).encode("utf-8"))
    survives = res.verdict() == "edge_survives"
    passing = res.windows_passing_gate()
    decision_reason = (
        f"{passing}/{res.window_count} windows pass the 0.98 gate "
        f"(majority required {res.majority_required()}); pooled CI upper bounds "
        f"mae={res.mae_ci['upper']:.4f} rmse={res.rmse_ci['upper']:.4f} vs gate <1.0"
    )
    per_window = [
        {
            "window_index": w.window_index,
            "test_origin_start": int(w.test_origin_start),
            "test_origin_end": int(w.test_origin_end),
            "train_count": int(w.train_count),
            "sample_count": int(w.sample_count),
            "relative_mae": float(w.relative_mae),
            "relative_rmse": float(w.relative_rmse),
            "passes_gate": bool(w.passes_gate()),
        }
        for w in res.windows
    ]
    record = append_record(
        args.ledger,
        {
            "run_tag": f"holdout-{canonical}",
            "candidate_family": canonical,
            "hypothesis": (
                f"Locked multi-window holdout confirmation of {canonical} at h={args.horizon}"
                + (
                    f" (requested as legacy name '{args.family}')"
                    if canonical != args.family
                    else ""
                )
            ),
            "horizon": args.horizon,
            "horizons": [args.horizon],
            "snapshot_id": "sha256:" + snapshot_digest.hexdigest(),
            "status": "success",
            "median_relative_mae": float(res.pooled_relative_mae),
            "median_relative_rmse": float(res.pooled_relative_rmse),
            "worst_fold_relative_rmse": float(max(w.relative_rmse for w in res.windows)),
            "folds_beating_persistence": f"{passing}/{res.window_count}",
            "promotable": survives,
            "decision": "keep" if survives else "discard",
            "decision_reason": decision_reason,
            "protocol_version": PROTOCOL_VERSION,
            "multiplicity_policy": MULTIPLICITY_POLICY,
            "evidence": {
                "snapshot_file": args.snapshot.name,
                "window_count": int(res.window_count),
                "window_rows": int(res.window_rows),
                "min_train_rows": int(res.min_train_rows),
                "pooled_sample_count": int(res.pooled_sample_count),
                "mae_ci": dict(res.mae_ci),
                "rmse_ci": dict(res.rmse_ci),
                "per_window": per_window,
            },
        },
    )
    export_tsv_summary(args.ledger, args.ledger.with_suffix(".tsv"))
    generate_markdown_report(args.ledger, args.ledger.parent / "REPORT.md")
    print(
        f"\nEvidence record {record['experiment_id']} appended to {args.ledger} "
        f"(commit {str(record['commit'])[:7]}, snapshot sha256:{snapshot_digest.hexdigest()[:12]}…)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
