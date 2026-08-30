"""Tests for V11.2 route-freeze invariants."""

from __future__ import annotations

import pytest

from research.volatility_forecasting.v11_2_freezer import (
    V112Route,
    freeze_routing_bundle,
)
from research.volatility_forecasting.v11_2_protocol import V112Protocol, feature_schema_digest


def _route(horizon: int, *, family: str = "M0_HAR_BASELINE", promoted: bool = False) -> V112Route:
    return V112Route(
        horizon=horizon,
        family=family,
        model_digest="a" * 64,
        scaler_digest="b" * 64,
        selection_record_digest="c" * 64,
        learned_promotion=promoted,
        artifact_path=f"baselines/h{horizon}.json",
    )


def _freeze_kwargs(tmp_path):
    protocol = V112Protocol()
    return {
        "protocol": protocol,
        "universe_sha256": "1" * 64,
        "panel_sha256": "2" * 64,
        "schema_sha256": feature_schema_digest(protocol),
        "split_sha256": "4" * 64,
        "development_evidence_sha256": "5" * 64,
        "routes": [_route(horizon) for horizon in protocol.horizons],
        "seed_evidence_sha256": [f"{value:x}".zfill(64) for value in range(1, 13)],
        "sealed_ciphertext_sha256": "6" * 64,
        "output_dir": tmp_path,
        "git_sha": "7" * 40,
        "git_dirty": False,
    }


def test_freeze_requires_complete_digest_bound_routes(tmp_path) -> None:
    bundle = freeze_routing_bundle(**_freeze_kwargs(tmp_path))
    assert len(bundle.routes) == 4
    assert bundle.master_freeze_sha256 == (tmp_path / "v11_2_routing_bundle.sha256").read_text()


def test_freeze_rejects_inconsistent_learned_promotion(tmp_path) -> None:
    kwargs = _freeze_kwargs(tmp_path)
    kwargs["routes"] = [_route(1, promoted=True), _route(3), _route(5), _route(7)]
    with pytest.raises(ValueError, match="learned_promotion"):
        freeze_routing_bundle(**kwargs)


def test_freeze_rejects_schema_digest_from_another_contract(tmp_path) -> None:
    kwargs = _freeze_kwargs(tmp_path)
    kwargs["schema_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="schema digest"):
        freeze_routing_bundle(**kwargs)
