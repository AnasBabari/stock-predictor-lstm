"""Fail-closed contracts for authentic external certification evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from volatility_forecasting.certification_gate_v12 import (
    CertificationGateError,
    CosignVerification,
    verify_certification_evidence,
)


def _evidence() -> dict[str, object]:
    return {
        "generation": "v12",
        "ohlcv": {
            "permission_status": "VERIFIED",
            "source_is_external": True,
            "hash_verified": True,
        },
        "pit64": {
            "completeness_verified": True,
            "external_reviewer_verified": True,
        },
        "signature": {"real_cosign_verification": True, "simulated": False},
        "holdout": {"generated_externally": True, "previously_revealed": False},
        "reserve": {"previously_opened": False},
        "code_hash_matches": True,
        "dataset_hash_matches": True,
    }


def _files(tmp_path: Path, payload: dict[str, object] | None = None) -> tuple[Path, Path, Path]:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload or _evidence()), encoding="utf-8")
    bundle = tmp_path / "review.sigstore.json"
    bundle.write_text("real bundle bytes are opaque to this gate", encoding="utf-8")
    public_key = tmp_path / "reviewer.pub"
    public_key.write_text("public key", encoding="utf-8")
    return evidence, bundle, public_key


def _success(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="Verified OK", stderr="")


def test_valid_evidence_requires_real_cosign_command(tmp_path: Path) -> None:
    evidence, bundle, public_key = _files(tmp_path)
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _success(command, **kwargs)

    verified = verify_certification_evidence(
        evidence,
        CosignVerification(evidence, bundle, public_key_path=public_key),
        runner=runner,
    )
    assert verified["generation"] == "v12"
    assert commands == [
        [
            "cosign",
            "verify-blob",
            str(evidence.resolve()),
            "--bundle",
            str(bundle.resolve()),
            "--key",
            str(public_key.resolve()),
        ]
    ]


@pytest.mark.parametrize(
    "marker",
    [
        "permission_pending",
        "simulated",
        "demo",
        "self_attested",
        "copied_diagnostic",
        "locally_generated",
        "previously_opened",
        "invalidated",
    ],
)
def test_forbidden_marker_fails_before_cosign(tmp_path: Path, marker: str) -> None:
    payload = _evidence()
    payload["note"] = marker
    evidence, bundle, public_key = _files(tmp_path, payload)
    with pytest.raises(CertificationGateError, match="forbidden markers"):
        verify_certification_evidence(
            evidence,
            CosignVerification(evidence, bundle, public_key_path=public_key),
            runner=lambda *args, **kwargs: pytest.fail("Cosign must not run"),
        )


def test_v11_generation_and_unverified_fields_fail_closed(tmp_path: Path) -> None:
    payload = _evidence()
    payload["generation"] = "v11.2"
    evidence, bundle, public_key = _files(tmp_path, payload)
    with pytest.raises(CertificationGateError, match="post-V11.2"):
        verify_certification_evidence(
            evidence,
            CosignVerification(evidence, bundle, public_key_path=public_key),
            runner=_success,
        )

    payload = _evidence()
    payload["signature"]["simulated"] = True  # type: ignore[index]
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CertificationGateError, match="signature.simulated"):
        verify_certification_evidence(
            evidence,
            CosignVerification(evidence, bundle, public_key_path=public_key),
            runner=_success,
        )

    payload = _evidence()
    payload["ohlcv"]["hash_verified"] = False  # type: ignore[index]
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CertificationGateError, match="ohlcv.hash_verified"):
        verify_certification_evidence(
            evidence,
            CosignVerification(evidence, bundle, public_key_path=public_key),
            runner=_success,
        )


def test_cosign_failure_and_mismatched_artifact_fail_closed(tmp_path: Path) -> None:
    evidence, bundle, public_key = _files(tmp_path)
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(CertificationGateError, match="exact evidence manifest"):
        verify_certification_evidence(
            evidence,
            CosignVerification(other, bundle, public_key_path=public_key),
            runner=_success,
        )

    def failure(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="bad proof")

    with pytest.raises(CertificationGateError, match="bad proof"):
        verify_certification_evidence(
            evidence,
            CosignVerification(evidence, bundle, public_key_path=public_key),
            runner=failure,
        )
