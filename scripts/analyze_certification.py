"""Post-certification descriptive analysis tool.

IMPORTANT: This tool performs descriptive, post-hoc statistical analysis
on an ALREADY-OPENED certification artifact. It is NOT part of the precommitted
certification gate and does NOT modify or supersede the original certification decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze_certification_artifact(
    cert_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Inspect and format descriptive analysis of an existing certification artifact."""
    if not cert_path.exists():
        raise FileNotFoundError(f"Certification file not found: {cert_path}")

    raw_data = json.loads(cert_path.read_text(encoding="utf-8"))
    protocol_version = raw_data.get("certification_protocol_version", "global-cert-v1")
    decisions = raw_data.get("decisions", {})

    analysis: dict[str, Any] = {
        "analysis_type": "post_certification",
        "affects_original_certification": False,
        "precommitted_gate": False,
        "source_artifact": str(cert_path),
        "source_protocol_version": protocol_version,
        "source_decision": raw_data.get("decision", "unknown"),
        "horizons_analyzed": list(decisions.keys()),
        "horizon_breakdown": {},
    }

    for h, dec in decisions.items():
        temporal_rmse = dec.get("temporal_relative_rmse", 1.0)
        temporal_mae = dec.get("temporal_relative_mae", 1.0)
        transfer_rmse = dec.get("transfer_relative_rmse", 1.0)
        transfer_mae = dec.get("transfer_relative_mae", 1.0)
        dir_acc = dec.get("temporal_direction_acc", 0.5)
        pos_prev = dec.get("positive_prevalence", 0.5)
        maj_acc = dec.get("majority_class_accuracy", 0.5)
        dir_delta = dec.get("direction_accuracy_delta_vs_majority", 0.0)
        bal_acc = dec.get("balanced_accuracy", 0.5)

        analysis["horizon_breakdown"][h] = {
            "candidate_name": dec.get("candidate_name", "none"),
            "decision": dec.get("decision", "abstain"),
            "point_metrics": {
                "temporal_relative_rmse": temporal_rmse,
                "temporal_relative_mae": temporal_mae,
                "transfer_relative_rmse": transfer_rmse,
                "transfer_relative_mae": transfer_mae,
                "temporal_rmse_improvement_pct": (1.0 - temporal_rmse) * 100.0,
                "transfer_rmse_improvement_pct": (1.0 - transfer_rmse) * 100.0,
            },
            "directional_diagnostics": {
                "temporal_direction_acc": dir_acc,
                "positive_prevalence": pos_prev,
                "majority_class_accuracy": maj_acc,
                "direction_delta_vs_majority": dir_delta,
                "balanced_accuracy": bal_acc,
                "prevalence_note": (
                    "Constant positive drift predictions match positive prevalence and "
                    "do not represent stock-by-stock directional discrimination skill."
                ),
            },
            "probabilistic_diagnostics": {
                "brier_score": dec.get("temporal_brier"),
                "status": dec.get("direction_probability_status", "not_available"),
            },
            "passed_gates": dec.get("passed_gates", []),
            "failed_gates": dec.get("failed_gates", []),
        }

    if out_path is not None:
        out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze certification artifact.")
    parser.add_argument(
        "--cert-file",
        type=Path,
        required=True,
        help="Path to 07_certification.json artifact",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Optional path to write 08_post_certification_analysis.json",
    )
    args = parser.parse_args()

    result = analyze_certification_artifact(args.cert_file, args.out_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
