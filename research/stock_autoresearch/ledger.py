"""Structured append-only experiment ledger, TSV exporter, and Markdown reporter."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


def append_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append a structured JSONL experiment record and return the formatted entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()

    full_record = {
        "experiment_id": record.get("experiment_id", f"exp_{int(datetime.now(UTC).timestamp() * 1000)}"),
        "parent_experiment": record.get("parent_experiment"),
        "commit": record.get("commit", "unknown"),
        "run_tag": record.get("run_tag", "default"),
        "candidate_family": record.get("candidate_family", record.get("candidate_name", "unknown")),
        "hypothesis": record.get("hypothesis", "Baseline comparison"),
        "snapshot_id": record.get("snapshot_id", "unknown"),
        "code_hash": record.get("code_hash", "none"),
        "fold_version": record.get("fold_version", "purged-expanding-v1"),
        "horizons": record.get("horizons", [record.get("horizon", 5)]),
        "seeds": record.get("seeds", [record.get("seed", 0)]),
        "training_budget": record.get("training_budget", {}),
        "status": record.get("status", "success"),
        "relative_mae": record.get("median_relative_mae", record.get("relative_mae")),
        "relative_rmse": record.get("median_relative_rmse", record.get("relative_rmse")),
        "worst_fold_rmse_ratio": record.get("worst_fold_relative_rmse", record.get("worst_fold_rmse_ratio")),
        "peak_vram_mb": record.get("peak_vram_mb", 0),
        "training_seconds": record.get("training_seconds", 0.0),
        "parameter_count": record.get("parameter_count", 0),
        "complexity_delta": record.get("complexity_delta", 0),
        "decision": record.get("decision", "keep" if record.get("promotable") else "discard"),
        "failure_reason": record.get("failure_reason", ""),
        "timestamp": timestamp,
        "recorded_at": timestamp,
    }

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(full_record, sort_keys=True) + "\n")

    return full_record


def export_tsv_summary(jsonl_path: Path, tsv_path: Path) -> None:
    """Export a TSV summary table from the experiment ledger JSONL."""
    if not jsonl_path.exists():
        return
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    headers = [
        "experiment_id",
        "run_tag",
        "candidate_family",
        "status",
        "decision",
        "relative_mae",
        "relative_rmse",
        "worst_fold_rmse_ratio",
        "peak_vram_mb",
        "training_seconds",
        "recorded_at",
    ]

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for r in records:
            writer.writerow([r.get(h, "") for h in headers])


def generate_markdown_report(jsonl_path: Path, report_path: Path) -> None:
    """Generate a concise Markdown report summarizing experiment outcomes."""
    if not jsonl_path.exists():
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total = len(records)
    kept = [r for r in records if r.get("decision") == "keep"]
    discarded = [r for r in records if r.get("decision") == "discard"]
    failed = [r for r in records if r.get("status") in ("crash", "invalid", "timeout", "oom")]

    lines = [
        "# Stock Autoresearch Ledger Report",
        "",
        f"- **Total Experiments**: {total}",
        f"- **Kept (Promotable)**: {len(kept)}",
        f"- **Discarded**: {len(discarded)}",
        f"- **Failed / Invalid**: {len(failed)}",
        "",
        "| ID | Family | Status | Decision | Rel MAE | Rel RMSE | Peak VRAM | Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in records:
        exp_id = str(r.get("experiment_id", ""))[:12]
        family = r.get("candidate_family", "-")
        status = r.get("status", "-")
        decision = r.get("decision", "-")
        rel_mae = f"{r.get('relative_mae'):.4f}" if isinstance(r.get("relative_mae"), (int, float)) else "-"
        rel_rmse = f"{r.get('relative_rmse'):.4f}" if isinstance(r.get("relative_rmse"), (int, float)) else "-"
        vram = f"{r.get('peak_vram_mb')} MB" if r.get("peak_vram_mb") else "-"
        t_sec = f"{r.get('training_seconds'):.1f}s" if isinstance(r.get("training_seconds"), (int, float)) else "-"
        lines.append(f"| `{exp_id}` | `{family}` | {status} | **{decision}** | {rel_mae} | {rel_rmse} | {vram} | {t_sec} |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
