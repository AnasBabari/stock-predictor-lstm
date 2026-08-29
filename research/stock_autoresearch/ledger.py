"""Structured append-only experiment ledger, TSV exporter, and Markdown reporter.

Schema version 2 adds auditable provenance (git commit, code hash), robust
experiment identifiers, decision-explaining fields, and an explicit
multiplicity policy so that reported claims reflect how many hypotheses were
evaluated. Corrupt ledger lines are never silently dropped: exporters fail
closed by default and must be asked explicitly to skip them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from filelock import FileLock

LEDGER_SCHEMA_VERSION = 2

# Claims policy: fold-level gates are exploratory screening. Each
# (family, ticker, horizon) claim requires exactly ONE locked multi-window
# holdout on a frozen snapshot; re-running the holdout after observing results
# invalidates prior certification; production claims require re-certification
# on new untouched data.
MULTIPLICITY_POLICY = (
    "screen-then-single-holdout-v1: fold gates are exploratory screening; "
    "exactly one locked multi-window holdout per (family, ticker, horizon); "
    "no post-hoc re-runs; re-certify on new untouched data before production claims"
)

PROTOCOL_VERSION = "multi-window-block-bootstrap-v1"

_UNAUDITED = "LEGACY_UNAUDITED"
_AUDITED = "audited"
_UNKNOWN_PROVENANCE = {"", "unknown", None}


class LedgerCorruptionError(ValueError):
    """Raised when ledger lines are corrupt and skipping was not requested."""

    def __init__(self, corrupt: list[tuple[int, str]]):
        self.corrupt = corrupt
        lines = ", ".join(str(line_no) for line_no, _ in corrupt)
        super().__init__(
            f"Corrupt experiment ledger line(s) {lines}; evidence would be "
            "silently lost. Fix or remove the offending lines, or pass "
            "skip_corrupt=True explicitly."
        )


def resolve_commit() -> str:
    """Best-effort git commit of the current working tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        commit = out.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def compute_code_hash() -> str:
    """Content hash over the research harness source (evaluator fingerprint)."""
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(package_dir.glob("*.py"))
    if not files:
        return "unavailable"
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _record_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"exp_{stamp}_{secrets.token_hex(4)}"


def _is_audited(record: Mapping[str, Any]) -> bool:
    return not (
        record.get("snapshot_id") in _UNKNOWN_PROVENANCE
        or record.get("commit") in _UNKNOWN_PROVENANCE
    )


def append_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append a structured JSONL experiment record and return the formatted entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()

    full_record = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "experiment_id": record.get("experiment_id", _record_id()),
        "parent_experiment": record.get("parent_experiment"),
        "commit": record.get("commit", resolve_commit()),
        "run_tag": record.get("run_tag", "default"),
        "candidate_family": record.get("candidate_family", record.get("candidate_name", "unknown")),
        "hypothesis": record.get("hypothesis", "Baseline comparison"),
        "snapshot_id": record.get("snapshot_id", "unknown"),
        "code_hash": record.get("code_hash", compute_code_hash()),
        "fold_version": record.get("fold_version", "purged-expanding-v1"),
        "protocol_version": record.get("protocol_version", ""),
        "multiplicity_policy": record.get("multiplicity_policy", ""),
        "horizons": record.get("horizons", [record.get("horizon", 5)]),
        "seeds": record.get("seeds", [record.get("seed", 0)]),
        "training_budget": record.get("training_budget", {}),
        "status": record.get("status", "success"),
        "relative_mae": record.get("median_relative_mae", record.get("relative_mae")),
        "relative_rmse": record.get("median_relative_rmse", record.get("relative_rmse")),
        "worst_fold_rmse_ratio": record.get(
            "worst_fold_relative_rmse", record.get("worst_fold_rmse_ratio")
        ),
        "folds_beating_persistence": record.get("folds_beating_persistence"),
        "promotable": bool(record.get("promotable", False)),
        "decision_reason": record.get("decision_reason", ""),
        "peak_vram_mb": record.get("peak_vram_mb", 0),
        "vram_source": record.get("vram_source", "unsampled"),
        "peak_rss_mb": record.get("peak_rss_mb", 0),
        "training_seconds": record.get("training_seconds", 0.0),
        "parameter_count": record.get("parameter_count", 0),
        "complexity_delta": record.get("complexity_delta", 0),
        "decision": record.get("decision", "keep" if record.get("promotable") else "discard"),
        "failure_reason": record.get("failure_reason", ""),
        "evidence": record.get("evidence", {}),
        "timestamp": timestamp,
        "recorded_at": timestamp,
    }

    lock = FileLock(f"{path}.lock", timeout=15)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(full_record, sort_keys=True) + "\n")

    return full_record


def read_records(jsonl_path: Path) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    """Parse a JSONL ledger into (records, corrupt[(line_number, reason)])."""
    records: list[dict[str, Any]] = []
    corrupt: list[tuple[int, str]] = []
    if not jsonl_path.exists():
        return records, corrupt
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("record is not a JSON object")
                records.append(parsed)
            except (json.JSONDecodeError, ValueError) as exc:
                corrupt.append((line_number, str(exc)))
    return records, corrupt


def _require_clean(corrupt: list[tuple[int, str]], skip_corrupt: bool) -> None:
    if corrupt and not skip_corrupt:
        raise LedgerCorruptionError(corrupt)


def export_tsv_summary(jsonl_path: Path, tsv_path: Path, *, skip_corrupt: bool = False) -> None:
    """Export a TSV summary table from the experiment ledger JSONL."""
    records, corrupt = read_records(jsonl_path)
    _require_clean(corrupt, skip_corrupt)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "experiment_id",
        "run_tag",
        "candidate_family",
        "status",
        "decision",
        "certification",
        "relative_mae",
        "relative_rmse",
        "worst_fold_rmse_ratio",
        "peak_vram_mb",
        "peak_rss_mb",
        "training_seconds",
        "recorded_at",
    ]

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for r in records:
            row = [r.get(h, "") for h in headers[:-1]]
            row.append(_AUDITED if _is_audited(r) else _UNAUDITED)
            writer.writerow(row)


def generate_markdown_report(
    jsonl_path: Path, report_path: Path, *, skip_corrupt: bool = False
) -> None:
    """Generate the Markdown report deterministically from ledger evidence."""
    records, corrupt = read_records(jsonl_path)
    _require_clean(corrupt, skip_corrupt)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records)
    kept = [r for r in records if r.get("decision") == "keep"]
    discarded = [r for r in records if r.get("decision") == "discard"]
    failed = [r for r in records if r.get("status") in ("crash", "invalid", "timeout", "oom")]
    unaudited_keeps = sum(1 for r in kept if not _is_audited(r))

    lines = [
        "# Stock Autoresearch Ledger Report",
        "",
        f"- **Total Experiments**: {total}",
        f"- **Kept (Promotable)**: {len(kept)}",
        f"- **Discarded**: {len(discarded)}",
        f"- **Failed / Invalid**: {len(failed)}",
        f"- **Kept records without auditable provenance**: {unaudited_keeps}",
        f"- **Ledger schema**: v{LEDGER_SCHEMA_VERSION}",
        f"- **Multiplicity policy**: {MULTIPLICITY_POLICY}",
        "",
        "> Records marked LEGACY_UNAUDITED predate provenance tracking; their",
        "> keep/discard decisions cannot be reconstructed from committed",
        "> evidence and must not be presented as certified.",
        "",
        "| ID | Family | Status | Decision | Certification | Rel MAE | Rel RMSE | Peak VRAM | Peak RSS | Time (s) | Reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in records:
        exp_id = str(r.get("experiment_id", ""))[:12]
        family = r.get("candidate_family", "-")
        status = r.get("status", "-")
        decision = r.get("decision", "-")
        certification = _AUDITED if _is_audited(r) else _UNAUDITED
        rel_mae = (
            f"{r.get('relative_mae'):.4f}"
            if isinstance(r.get("relative_mae"), (int, float))
            else "-"
        )
        rel_rmse = (
            f"{r.get('relative_rmse'):.4f}"
            if isinstance(r.get("relative_rmse"), (int, float))
            else "-"
        )
        vram = f"{r.get('peak_vram_mb')} MB" if r.get("peak_vram_mb") else "-"
        rss = f"{r.get('peak_rss_mb')} MB" if r.get("peak_rss_mb") else "-"
        t_sec = (
            f"{r.get('training_seconds'):.1f}s"
            if isinstance(r.get("training_seconds"), (int, float))
            else "-"
        )
        reason = str(r.get("decision_reason") or r.get("failure_reason") or "")
        reason = reason.replace("|", "\\|")
        lines.append(
            f"| `{exp_id}` | `{family}` | {status} | **{decision}** | {certification} | "
            f"{rel_mae} | {rel_rmse} | {vram} | {rss} | {t_sec} | {reason} |"
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
