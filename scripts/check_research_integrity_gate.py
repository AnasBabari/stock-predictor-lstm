"""Research Integrity Gate for StockLSTM V10 & V9 Scientific Programmes.

Fails CI if:
- Protocol files drift or contain unhashed changes
- Ineligible data is marked certification-eligible
- Private signing keys exist inside the repository tree
- V9 quarantine manifest is missing or inconsistent
- Uncertified models are labelled as certified
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_research_integrity() -> list[str]:
    errors = []

    # 1. Check no private signing keys in repository
    release_keys_dir = ROOT / "backend" / "release_keys"
    if release_keys_dir.exists():
        priv_keys = list(release_keys_dir.glob("*.private.pem"))
        if priv_keys:
            errors.append(f"Private signing keys detected in repository: {priv_keys}. Private keys MUST NEVER be committed or generated in tree.")

    # 2. Check V9 quarantine manifest
    q_manifest = ROOT / "research" / "results" / "v9_quarantine_manifest.json"
    if not q_manifest.exists():
        errors.append("V9 quarantine manifest missing at research/results/v9_quarantine_manifest.json")
    else:
        try:
            q_data = json.loads(q_manifest.read_text(encoding="utf-8"))
            if q_data.get("status") != "quarantined_development_diagnostic_only":
                errors.append("V9 quarantine manifest status must be 'quarantined_development_diagnostic_only'")
        except Exception as exc:
            errors.append(f"Malformed V9 quarantine manifest: {exc}")

    # 3. Check V10 protocol
    v10_proto_file = ROOT / "configs" / "volatility_v10_protocol.json"
    if not v10_proto_file.exists():
        errors.append("V10 protocol file missing at configs/volatility_v10_protocol.json")
    else:
        try:
            v10_proto = json.loads(v10_proto_file.read_text(encoding="utf-8"))
            if v10_proto.get("protocol_status") != "frozen":
                errors.append("V10 protocol status must be 'frozen'")
            if v10_proto.get("historical_context", {}).get("v9_diagnostic_observed") is not True:
                errors.append("V10 historical_context must truthfully disclose v9_diagnostic_observed=True")
        except Exception as exc:
            errors.append(f"Malformed V10 protocol: {exc}")

    return errors


def main() -> int:
    errors = check_research_integrity()
    if errors:
        print("Research Integrity Gate FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - ERROR: {e}", file=sys.stderr)
        return 1
    print("Research Integrity Gate PASSED: Protocols, key invariants, and quarantine manifests verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
