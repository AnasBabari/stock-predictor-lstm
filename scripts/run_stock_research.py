"""Run a baseline-relative stock research experiment on a frozen CSV snapshot.

Supports persistence, Ridge, Elastic Net, Compact MLP, DLinear, and Small TCN model families,
multi-fidelity evaluation levels, and controller execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# Add research directory to Python path if needed
research_dir = Path(__file__).resolve().parent.parent / "research"
if str(research_dir) not in sys.path:
    sys.path.insert(0, str(research_dir))

from stock_autoresearch.candidates import (  # noqa: E402 - research dir added to sys.path above
    CompactMLPCandidate,
    DLinearCandidate,
    ElasticNetCandidate,
    PersistenceCandidate,
    RidgeCandidate,
    SmallTCNCandidate,
)
from stock_autoresearch.config import EVALUATION_POLICY  # noqa: E402
from stock_autoresearch.controller import ExperimentController  # noqa: E402
from stock_autoresearch.data import Snapshot  # noqa: E402
from stock_autoresearch.ledger import (  # noqa: E402
    append_record,
    export_tsv_summary,
    generate_markdown_report,
)
from stock_autoresearch.multi_seed import evaluate_multi_seed  # noqa: E402


def compute_snapshot_id(frame: pd.DataFrame, feature_names: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(frame.to_csv(index=True).encode("utf-8"))
    digest.update(json.dumps(feature_names, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stock Autoresearch candidate evaluations.")
    parser.add_argument("snapshot", type=Path, help="Path to frozen CSV market snapshot.")
    parser.add_argument("--features", nargs="+", required=True, help="Feature column names.")
    parser.add_argument("--ticker", default="unknown", help="Ticker symbol for research metadata.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon (default: 5).")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["persistence", "ridge", "elastic_net", "compact_mlp", "dlinear", "small_tcn"],
        help="Candidate families to evaluate.",
    )
    parser.add_argument("--run-tag", default="dev_run", help="Run tag for experiment grouping.")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Multi-fidelity level (0: smoke, 1: screen, 2: confirm).",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help=(
            "Evaluate each family across policy.seed_count seeds via "
            "evaluate_multi_seed instead of the single-seed controller path. "
            "This is an unguarded in-process fast path: no subprocess "
            "isolation, no timeout or VRAM guard, and no harness-lock gate, "
            "unlike the controller path."
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("research/results/experiments.jsonl"),
        help="Path to experiment ledger JSONL.",
    )
    args = parser.parse_args()

    if not args.snapshot.exists():
        print(f"Error: Snapshot file {args.snapshot} does not exist.", file=sys.stderr)
        return 1

    frame = pd.read_csv(args.snapshot, index_col=0, parse_dates=True)
    sn_id = compute_snapshot_id(frame, args.features)
    snapshot = Snapshot(frame=frame, snapshot_id=sn_id, feature_names=tuple(args.features))

    factories = {
        "persistence": lambda seed: PersistenceCandidate(),
        "ridge": lambda seed: RidgeCandidate(),
        "elastic_net": lambda seed: ElasticNetCandidate(),
        "compact_mlp": lambda seed: CompactMLPCandidate(),
        "dlinear": lambda seed: DLinearCandidate(),
        "small_tcn": lambda seed: SmallTCNCandidate(seed=seed),
    }

    controller = ExperimentController(
        snapshot_path=args.snapshot,
        ledger_path=args.ledger,
        run_tag=args.run_tag,
        policy=EVALUATION_POLICY,
    )

    summaries = []
    for family in args.families:
        if family not in factories:
            print(f"Warning: Unknown candidate family '{family}', skipping.", file=sys.stderr)
            continue

        print(f"Running candidate family '{family}' (Level {args.level})...")
        if args.multi_seed:
            multi_seed_record = evaluate_multi_seed(
                snapshot,
                factories[family],
                horizon=args.horizon,
                policy=EVALUATION_POLICY,
            )
            entry = append_record(
                args.ledger,
                {
                    "run_tag": args.run_tag,
                    "candidate_family": family,
                    "hypothesis": f"Multi-seed evaluation of {family} candidate family",
                    "snapshot_id": sn_id,
                    "horizon": args.horizon,
                    "seeds": multi_seed_record["seeds"],
                    "status": multi_seed_record["status"],
                    "failure_reason": (
                        f"{multi_seed_record['failure_count']} seed run(s) failed"
                        if multi_seed_record["failure_count"]
                        else ""
                    ),
                    "median_relative_mae": multi_seed_record["median_relative_mae"],
                    "median_relative_rmse": multi_seed_record["median_relative_rmse"],
                    "worst_fold_relative_rmse": multi_seed_record["worst_fold_relative_rmse"],
                    "folds_beating_persistence": multi_seed_record["folds_beating_persistence"],
                    "promotable": multi_seed_record["promotable"],
                    "seed_aggregate": multi_seed_record["seed_aggregate"],
                    "per_seed": multi_seed_record["per_seed"],
                    "promotable_seed_count": multi_seed_record["promotable_seed_count"],
                    "failure_count": multi_seed_record["failure_count"],
                },
            )
            summaries.append(entry)
            continue
        entry = controller.execute_trial(
            candidate_family=family,
            hypothesis=f"Evaluation of {family} candidate family",
            level=args.level,
        )
        summaries.append(entry)
        if entry.get("status") == "violates_harness_lock":
            print(
                f"Error: trial for '{family}' has status 'violates_harness_lock'. "
                "Uncommitted changes to protected harness or production files block "
                "trials (see PROHIBITED_FILES in "
                "research/stock_autoresearch/controller.py: locked research modules, "
                "backend/api.py, backend/model.py, render.yaml, frontend/). Commit "
                "or revert those changes before running trials.",
                file=sys.stderr,
            )
            return 1

    export_tsv_summary(args.ledger, args.ledger.with_suffix(".tsv"))
    generate_markdown_report(args.ledger, args.ledger.parent / "REPORT.md")

    print(
        json.dumps(
            {"snapshot_id": sn_id, "results_count": len(summaries), "run_tag": args.run_tag},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
