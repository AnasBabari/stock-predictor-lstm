"""Tests for the frozen V10 protocol preregistration and canonical hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.volatility_forecasting.protocol_hashing import (
    protocol_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "configs" / "volatility_v10_protocol.json"
PREREG_DOC_PATH = REPO_ROOT / "docs" / "VOLATILITY_V10_PREREGISTRATION.md"


@pytest.fixture(scope="module")
def protocol() -> dict:
    assert PROTOCOL_PATH.exists(), f"v10 protocol missing at {PROTOCOL_PATH}"
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prereg_doc() -> str:
    assert PREREG_DOC_PATH.exists(), f"v10 preregistration doc missing at {PREREG_DOC_PATH}"
    return PREREG_DOC_PATH.read_text(encoding="utf-8")


def test_v10_protocol_structure_and_status(protocol: dict) -> None:
    assert protocol["protocol_version"] == "volatility-v10"
    assert protocol["protocol_status"] == "frozen"
    assert protocol["independent_per_horizon_selection"] is True


def test_v10_historical_truth_and_v9_disclosure(protocol: dict) -> None:
    hist = protocol["historical_context"]
    assert hist["v9_diagnostic_observed"] is True
    assert hist["v9_certification_valid"] is False
    assert hist["v10_design_informed_by_v9"] is True
    assert hist["v10_test_opened"] is False


def test_canonical_protocol_hashing_is_deterministic(protocol: dict) -> None:
    h1 = protocol_sha256(protocol)
    h2 = protocol_sha256(json.loads(json.dumps(protocol)))
    assert h1 == h2
    assert len(h1) == 64


def test_prereg_doc_matches_v10_protocol_identities(protocol: dict, prereg_doc: str) -> None:
    assert "volatility-v10" in prereg_doc
    assert protocol["target"]["target_contract_version"] in prereg_doc
    assert protocol["feature_schema"]["feature_contract_version"] in prereg_doc
    assert protocol["split"]["split_contract_version"] in prereg_doc
