#!/usr/bin/env python3
"""One-shot v8 certification — sealed historical test + asset-transfer.

This is the v8 analogue of ``scripts/certify_prospective_volatility_candidate.py``
but for the chronological 70/15/15 split. It never trains.

It verifies:
- candidate role == prospective_v8_development_candidate
- protocol == global-volatility-distribution-v8-*
- panel/universe/split/news checksums match
- feature/target order matches
- test sealed (no prior holdout-opened marker)
- every member and horizon present
- no retraining, no future news
- NMM/MSFT coverage where applicable
- all required horizons pass

On success materializes ``locked_v8_certification_candidate`` with
``release_eligible=true`` and ``metric_source=
locked_historical_temporal_test_plus_asset_transfer``.
On any failure: ``release_eligible=false``, no partial candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402
from research.volatility_forecasting.cache import (  # noqa: E402
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
)
from research.volatility_forecasting.candidate_v8 import v8_ensemble_identity  # noqa: E402
from research.volatility_forecasting.certification import (  # noqa: E402
    LockedPopulationInput,
    certify_locked_predictions,
)
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.export import (  # noqa: E402
    load_prospective_v8_candidate_member,
)
from research.volatility_forecasting.folds import VolatilityFoldPlan  # noqa: E402
from research.volatility_forecasting.refit import FrozenEnsemble  # noqa: E402
from research.volatility_forecasting.split_v8 import build_v8_chronological_split  # noqa: E402
from research.volatility_forecasting.universe_v8 import verify_universe_manifest  # noqa: E402
from research.volatility_forecasting.v8_protocol import (  # noqa: E402
    V8_PROTOCOL_VERSION_NEWS,
    V8_PROTOCOL_VERSION_NUMERIC,
    v8_manifest,
    v8_protocol,
)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    """Durably replace one JSON file without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _split_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _v8_fold_plan(split, examples) -> VolatilityFoldPlan:
    """Adapt the frozen 70/15/15 reserve identities to the generic certifier."""
    return VolatilityFoldPlan(
        folds=(),
        train_tickers=split.train_tickers,
        asset_holdout_tickers=split.holdout_tickers,
        temporal_certification_indices=split.temporal_test_indices,
        asset_transfer_certification_indices=split.asset_transfer_test_indices,
        certification_start=np.min(examples.origin_dates[split.pooled_test_indices]),
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One-shot v8 certification (sealed test)")
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--panel-dir", type=Path, required=True)
    ap.add_argument("--universe-manifest", type=Path, required=True)
    ap.add_argument("--news-manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True, help="Empty output dir for certification")
    ap.add_argument("--holdouts", type=str, default="NMM,MSFT")
    ap.add_argument(
        "--open-sealed-test",
        action="store_true",
        help="Required acknowledgement: irreversibly opens sealed test",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.open_sealed_test:
        print("--open-sealed-test is required (acknowledges one-shot)", file=sys.stderr)
        return 2
    cand_dir = args.candidate_dir.resolve()
    panel_dir = args.panel_dir.resolve()
    uni_path = args.universe_manifest.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"--out must be empty: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    holdouts = tuple(sorted({t.strip().upper() for t in args.holdouts.split(",") if t.strip()}))
    if not holdouts:
        print("at least one required holdout is required", file=sys.stderr)
        return 2

    # Validate every immutable identity and every persisted weight before the
    # irreversible marker is written. None of these checks reads test targets.
    cand_manifest_path = cand_dir / "candidate-manifest.json"
    if not cand_manifest_path.exists():
        print(f"candidate manifest missing: {cand_manifest_path}", file=sys.stderr)
        return 2
    try:
        cand = json.loads(cand_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"candidate manifest is invalid: {error}", file=sys.stderr)
        return 2
    if cand.get("artifact_role") != "prospective_v8_development_candidate":
        print(
            f"candidate role must be prospective_v8_development_candidate, got {cand.get('artifact_role')}",
            file=sys.stderr,
        )
        return 2
    if cand.get("placeholder") or cand.get("model_type") in {
        "ridge_stub",
        "ridge_cpu",
        "dummy",
    }:
        print("placeholder candidates are not certifiable", file=sys.stderr)
        return 2
    members = cand.get("members")
    if not isinstance(members, list) or not members:
        print("candidate has no persisted model members", file=sys.stderr)
        return 2
    if any(
        not isinstance(member, dict)
        or not isinstance(member.get("weights_file"), str)
        or not str(member["weights_file"]).endswith(".pt")
        for member in members
    ):
        print("candidate members are placeholders or malformed", file=sys.stderr)
        return 2
    proto_version = cand.get("protocol_version")
    if proto_version not in (V8_PROTOCOL_VERSION_NEWS, V8_PROTOCOL_VERSION_NUMERIC):
        print(f"candidate protocol {proto_version} not v8", file=sys.stderr)
        return 2
    protocol_payload = cand.get("protocol")
    news_enabled = bool(
        protocol_payload.get("news_enabled") if isinstance(protocol_payload, dict) else False
    )
    protocol = v8_protocol(news_enabled=news_enabled)
    frozen_manifest = v8_manifest(news_enabled=news_enabled)
    if proto_version != protocol.protocol_version:
        print(
            f"candidate protocol {proto_version} != expected {protocol.protocol_version}",
            file=sys.stderr,
        )
        return 2
    if cand.get("protocol") != frozen_manifest:
        print("candidate protocol payload differs from the frozen v8 manifest", file=sys.stderr)
        return 2
    if news_enabled:
        print(
            "news-enabled certification requires the real aligned historical news matrix; "
            "only the numeric v8 path is currently certifiable",
            file=sys.stderr,
        )
        return 2

    try:
        uni = verify_universe_manifest(json.loads(uni_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"universe manifest is invalid: {error}", file=sys.stderr)
        return 2
    uni_sha = uni.get("sha256")
    if cand.get("universe_manifest_sha256") != uni_sha:
        print(
            f"candidate universe sha mismatch {cand.get('universe_manifest_sha256')} vs {uni_sha}",
            file=sys.stderr,
        )
        return 2
    if not uni.get("coverage_certifiable") or not cand.get("universe_certifiable"):
        print("candidate universe is diagnostic-only and cannot be certified", file=sys.stderr)
        return 2

    panel_fp = panel_fingerprint(panel_dir)
    if cand.get("panel_checksum") != panel_fp:
        print(
            f"candidate panel checksum mismatch {cand.get('panel_checksum')} vs {panel_fp}",
            file=sys.stderr,
        )
        return 2

    candidate_split = cand.get("split_manifest")
    cand_split_sha = cand.get("split_manifest_sha256")
    if not isinstance(candidate_split, dict) or not isinstance(cand_split_sha, str):
        print("candidate split provenance is missing", file=sys.stderr)
        return 2
    if _split_digest(candidate_split) != cand_split_sha:
        print("candidate split manifest checksum does not match", file=sys.stderr)
        return 2
    if tuple(candidate_split.get("holdout_assets", ())) != tuple(sorted(candidate_split.get("holdout_assets", ()))):
        print("candidate holdout assets are not canonical", file=sys.stderr)
        return 2
    if not set(holdouts).issubset(set(candidate_split.get("holdout_assets", ()))):
        print("requested required holdouts differ from the frozen split", file=sys.stderr)
        return 2

    news_sha = cand.get("news_snapshot_checksum") or (
        "sha256:" + hashlib.sha256(b"no_news").hexdigest()
    )
    expected_numeric_news_sha = "sha256:" + hashlib.sha256(b"no_news").hexdigest()
    if news_sha != expected_numeric_news_sha:
        print("numeric candidate has an unexpected news snapshot identity", file=sys.stderr)
        return 2

    expected_seeds = tuple(int(value) for value in protocol.seeds)
    member_seeds = tuple(sorted(member.get("seed") for member in members if isinstance(member, dict)))
    if member_seeds != expected_seeds:
        print(f"candidate seeds {member_seeds} differ from protocol {expected_seeds}", file=sys.stderr)
        return 2
    try:
        loaded_members = tuple(
            load_prospective_v8_candidate_member(cand_dir, seed) for seed in expected_seeds
        )
        ensemble = FrozenEnsemble(
            members=loaded_members,
            model_identity=v8_ensemble_identity(loaded_members),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"candidate model verification failed: {error}", file=sys.stderr)
        return 2
    if ensemble.model_identity != cand.get("model_identity"):
        print("candidate ensemble identity does not match its members", file=sys.stderr)
        return 2

    development_report_path = cand_dir / "development-report.json"
    if not development_report_path.is_file():
        print("candidate development evidence is missing", file=sys.stderr)
        return 2
    development_evidence_sha = _sha256_file(development_report_path)

    # The next write is the irreversible boundary. Only after it succeeds may
    # derived examples or sealed target arrays be loaded.
    marker = {
        "candidate_manifest_sha256": _sha256_file(cand_manifest_path),
        "development_evidence_sha256": development_evidence_sha,
        "panel_checksum": panel_fp,
        "universe_sha256": uni_sha,
        "protocol_version": proto_version,
        "holdouts": list(holdouts),
        "one_shot": True,
    }
    _write_json_atomic(out / "v8-holdout-opened.json", marker)
    print(f"one-shot marker written to {out / 'v8-holdout-opened.json'}")

    report: dict[str, object]
    split = None
    try:
        for root in [Path(r"C:\tmp\stocklstm-volatility-panel-v1\example-cache")]:
            try:
                compat = find_compatible_example_cache(
                    root, panel_checksum=panel_fp, protocol=protocol
                )
                if compat:
                    examples = load_example_cache(
                        compat, panel_checksum=panel_fp, protocol=protocol
                    )
                    break
            except Exception:
                continue
        else:
            panel = load_panel_from_directory(panel_dir)
            examples = build_volatility_panel_examples(panel, protocol)

        split = build_v8_chronological_split(
            examples,
            protocol=protocol,
            required_asset_holdouts=holdouts,
            universe_manifest_sha256=str(uni_sha),
            universe_coverage_certifiable=True,
            panel_checksum=panel_fp,
            news_snapshot_checksum=str(news_sha),
        )
        recomputed_split = split.manifest.__dict__
        recomputed_sha = _split_digest(recomputed_split)
        if recomputed_sha != cand_split_sha:
            raise ValueError(
                f"split manifest SHA mismatch after open: {recomputed_sha} vs {cand_split_sha}"
            )
        if json.loads(json.dumps(recomputed_split, default=str)) != candidate_split:
            raise ValueError("recomputed split content differs from the frozen candidate")

        fold_plan = _v8_fold_plan(split, examples)
        temporal_predictions = ensemble.predict(examples, split.temporal_test_indices)
        temporal_baseline, temporal_return_baseline = ensemble.matched_baselines(
            examples, split.temporal_test_indices
        )
        transfer_predictions = ensemble.predict(examples, split.asset_transfer_test_indices)
        transfer_baseline, transfer_return_baseline = ensemble.matched_baselines(
            examples, split.asset_transfer_test_indices
        )
        certification = certify_locked_predictions(
            examples=examples,
            fold_plan=fold_plan,
            temporal=LockedPopulationInput(
                population="temporal",
                indices=split.temporal_test_indices,
                predictions=temporal_predictions,
                baseline_variance=temporal_baseline,
                baseline_return_variance=temporal_return_baseline,
            ),
            asset_transfer=LockedPopulationInput(
                population="asset_transfer",
                indices=split.asset_transfer_test_indices,
                predictions=transfer_predictions,
                baseline_variance=transfer_baseline,
                baseline_return_variance=transfer_return_baseline,
            ),
            model_identity=ensemble.model_identity,
            development_evidence_sha256=development_evidence_sha,
            required_asset_holdouts=holdouts,
            eligible_horizons=tuple(
                int(value) for value in frozen_manifest["required_horizons"]
            ),
            seed=20260827,
        )
        report = {
            **certification.to_dict(),
            "protocol_version": proto_version,
            "model_version": cand.get("model_version"),
            "metric_source": "locked_historical_temporal_test_plus_asset_transfer",
            "certification_scope": "historical_temporal_test_plus_asset_transfer",
            "panel_checksum": panel_fp,
            "universe_sha256": uni_sha,
            "split_manifest": recomputed_split,
            "candidate_manifest_sha256": marker["candidate_manifest_sha256"],
            "holdout_opened": str(out / "v8-holdout-opened.json"),
            "release_eligible": certification.status == "passed",
        }
    except Exception as error:
        report = {
            "status": "failed",
            "release_eligible": False,
            "protocol_version": proto_version,
            "model_version": cand.get("model_version"),
            "metric_source": "locked_historical_temporal_test_plus_asset_transfer",
            "certification_scope": "historical_temporal_test_plus_asset_transfer",
            "panel_checksum": panel_fp,
            "universe_sha256": uni_sha,
            "candidate_manifest_sha256": marker["candidate_manifest_sha256"],
            "holdout_opened": str(out / "v8-holdout-opened.json"),
            "failure": str(error),
        }

    report_path = out / "v8-locked-certification.json"
    _write_json_atomic(report_path, report)
    status = str(report["status"])
    print(f"certification report {status} written to {report_path}")
    if status != "passed":
        print("certification failed — no release eligible candidate materialized", file=sys.stderr)
        return 1

    # Materialize into a temporary sibling and rename only after every model
    # checksum and sidecar is present. A crash cannot expose a partial candidate.
    candidate_target = out / "candidate"
    temporary = Path(tempfile.mkdtemp(prefix=".locked-v8-", dir=out))
    try:
        for member in members:
            filename = str(member["weights_file"])
            shutil.copyfile(cand_dir / filename, temporary / filename)
            if _sha256_file(temporary / filename) != member["weights_sha256"]:
                raise RuntimeError(f"copied member checksum mismatch: {filename}")
        for sidecar in (
            "development-report.json",
            "split-v8-manifest.json",
            "universe-v8-manifest.json",
        ):
            source = cand_dir / sidecar
            if source.is_file():
                shutil.copyfile(source, temporary / sidecar)
        shutil.copyfile(report_path, temporary / "v8-locked-certification.json")
        locked_manifest = dict(cand)
        locked_manifest.update(
            {
                "artifact_role": "locked_v8_certification_candidate",
                "release_eligible": True,
                "metric_source": "locked_historical_temporal_test_plus_asset_transfer",
                "certification_report_sha256": _sha256_file(report_path),
            }
        )
        _write_json_atomic(temporary / "candidate-manifest.json", locked_manifest)
        temporary.replace(candidate_target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"locked candidate materialized at {candidate_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
