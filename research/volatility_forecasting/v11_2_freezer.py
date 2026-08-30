"""Immutable V11.2 per-horizon routing bundle freezer."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .v11_2_protocol import V112Protocol, canonical_json_digest, protocol_manifest


@dataclass(frozen=True)
class V112Route:
    horizon: int
    family: str
    model_digest: str
    scaler_digest: str
    selection_record_digest: str
    learned_promotion: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "family": self.family,
            "model_digest": self.model_digest,
            "scaler_digest": self.scaler_digest,
            "selection_record_digest": self.selection_record_digest,
            "learned_promotion": self.learned_promotion,
        }


@dataclass(frozen=True)
class V112RoutingBundle:
    protocol: dict[str, Any]
    universe_sha256: str
    panel_sha256: str
    schema_sha256: str
    split_sha256: str
    development_evidence_sha256: str
    routes: tuple[V112Route, ...]
    seed_evidence_sha256: tuple[str, ...]
    sealed_ciphertext_sha256: str
    master_freeze_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "universe_sha256": self.universe_sha256,
            "panel_sha256": self.panel_sha256,
            "schema_sha256": self.schema_sha256,
            "split_sha256": self.split_sha256,
            "development_evidence_sha256": self.development_evidence_sha256,
            "routes": [route.to_dict() for route in self.routes],
            "seed_evidence_sha256": list(self.seed_evidence_sha256),
            "sealed_ciphertext_sha256": self.sealed_ciphertext_sha256,
            "sealed_test_status": "LOCKED_UNOPENED",
            "master_freeze_sha256": self.master_freeze_sha256,
        }


def _state_digest(state: Any) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def freeze_routing_bundle(
    *,
    protocol: V112Protocol,
    universe_sha256: str,
    panel_sha256: str,
    schema_sha256: str,
    split_sha256: str,
    development_evidence_sha256: str,
    routes: list[V112Route],
    seed_evidence_sha256: list[str],
    sealed_ciphertext_sha256: str,
    output_dir: Path,
    git_sha: str,
    git_dirty: bool,
) -> V112RoutingBundle:
    """Freeze exactly one complete route for every required horizon."""
    if git_dirty:
        raise ValueError("cannot freeze V11.2 with a dirty Git tree")
    if len(routes) != len(protocol.horizons):
        raise ValueError("one V11.2 route is required for every horizon")
    horizons = [route.horizon for route in routes]
    if sorted(horizons) != sorted(protocol.horizons):
        raise ValueError("V11.2 routes must cover horizons 1, 3, 5, and 7 exactly once")
    if not sealed_ciphertext_sha256 or len(sealed_ciphertext_sha256) < 64:
        raise ValueError("sealed ciphertext digest is required before candidate freeze")
    payload: dict[str, Any] = {
        "protocol": protocol_manifest(protocol),
        "universe_sha256": universe_sha256,
        "panel_sha256": panel_sha256,
        "schema_sha256": schema_sha256,
        "split_sha256": split_sha256,
        "development_evidence_sha256": development_evidence_sha256,
        "routes": [route.to_dict() for route in sorted(routes, key=lambda item: item.horizon)],
        "seed_evidence_sha256": sorted(seed_evidence_sha256),
        "sealed_ciphertext_sha256": sealed_ciphertext_sha256,
        "git_sha": git_sha,
        "sealed_test_status": "LOCKED_UNOPENED",
    }
    digest = canonical_json_digest(payload)
    bundle = V112RoutingBundle(
        protocol=payload["protocol"],
        universe_sha256=universe_sha256,
        panel_sha256=panel_sha256,
        schema_sha256=schema_sha256,
        split_sha256=split_sha256,
        development_evidence_sha256=development_evidence_sha256,
        routes=tuple(sorted(routes, key=lambda item: item.horizon)),
        seed_evidence_sha256=tuple(sorted(seed_evidence_sha256)),
        sealed_ciphertext_sha256=sealed_ciphertext_sha256,
        master_freeze_sha256=digest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v11_2_routing_bundle.json").write_text(
        json.dumps({**bundle.to_dict(), "git_sha": git_sha}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "v11_2_routing_bundle.sha256").write_text(digest, encoding="ascii")
    return bundle


def state_digest(state: Any) -> str:
    """Public helper for per-seed model evidence."""
    return _state_digest(state)
