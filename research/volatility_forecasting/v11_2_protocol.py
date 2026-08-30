"""Frozen methodology contract for the V11.2 numeric PIT64 experiment.

This module is intentionally additive.  V11.1 imports and artifacts are not
changed by the V11.2 protocol.  A protocol object is serialized and hashed
before data construction so changing a scientific choice creates a new
experiment namespace instead of silently invalidating an old one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

V11_2_PROTOCOL_ID = "stocklstm-volatility-v11.2-numeric-pit64"
V11_2_PROTOCOL_VERSION = "v11.2-numeric-pit64"
V11_2_HORIZONS: tuple[int, ...] = (1, 3, 5, 7)
V11_2_SEEDS: tuple[int, ...] = (41, 42, 43)
V11_2_CANONICAL_SEED = 42
V11_2_UNIVERSE_SIZE = 64
V11_2_TRAIN_RATIO = 0.70
V11_2_VALIDATION_RATIO = 0.15
V11_2_TEST_RATIO = 0.15
V11_2_MAX_HORIZON = 7
V11_2_PURGE_SESSIONS = 7
V11_2_EMBARGO_SESSIONS = 30
V11_2_BOOTSTRAP_BLOCK_SESSIONS = 20
V11_2_BOOTSTRAP_REPLICATES = 10_000
V11_2_BOOTSTRAP_SEED = 42
V11_2_FAMILYWISE_ALPHA = 0.05
V11_2_NEWS_MODE = "M2_DISABLED_BY_PROTOCOL"
V11_2_MODEL_VERSION = "v11.2-numeric-residual-v1"


class HistoricalNewsFeatureProvider(Protocol):
    """Future V12 extension point; V11.2 rejects all providers."""

    def audit_provenance(self, *args: Any, **kwargs: Any) -> Any: ...

    def build_point_in_time_features(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class V112Protocol:
    protocol_id: str = V11_2_PROTOCOL_ID
    protocol_version: str = V11_2_PROTOCOL_VERSION
    model_scope: str = "numeric_only"
    universe_size: int = V11_2_UNIVERSE_SIZE
    horizons: tuple[int, ...] = V11_2_HORIZONS
    split: str = "chronological_70_15_15"
    train_ratio: float = V11_2_TRAIN_RATIO
    validation_ratio: float = V11_2_VALIDATION_RATIO
    test_ratio: float = V11_2_TEST_RATIO
    purge_sessions: int = V11_2_PURGE_SESSIONS
    embargo_sessions: int = V11_2_EMBARGO_SESSIONS
    selection: str = "per_horizon"
    candidate_families: tuple[str, ...] = (
        "ZERO_RETURN_CONST_VAR",
        "ZERO_RETURN_PERSISTENCE_VOL",
        "M0_HAR_BASELINE",
        "RIDGE_LOCATION_HAR_SCALE",
        "HISTGB_LOCATION_HAR_SCALE",
        "M1_NUMERIC_RESIDUAL",
    )
    seeds: tuple[int, ...] = V11_2_SEEDS
    canonical_seed: int = V11_2_CANONICAL_SEED
    bootstrap_block_sessions: int = V11_2_BOOTSTRAP_BLOCK_SESSIONS
    bootstrap_replicates: int = V11_2_BOOTSTRAP_REPLICATES
    bootstrap_seed: int = V11_2_BOOTSTRAP_SEED
    familywise_alpha: float = V11_2_FAMILYWISE_ALPHA
    news_mode: str = V11_2_NEWS_MODE
    model_version: str = V11_2_MODEL_VERSION

    def __post_init__(self) -> None:
        if self.universe_size != 64:
            raise ValueError("V11.2 requires exactly 64 accepted securities")
        if self.horizons != V11_2_HORIZONS:
            raise ValueError("V11.2 horizons are fixed to 1, 3, 5, and 7")
        if self.selection != "per_horizon":
            raise ValueError("V11.2 requires independent per-horizon selection")
        if self.news_mode != V11_2_NEWS_MODE:
            raise ValueError("V11.2 is numeric-only and structurally disables news")
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-12:
            raise ValueError("split ratios must sum to one")
        if self.purge_sessions < self.max_horizon or self.embargo_sessions < self.max_horizon:
            raise ValueError("purge and embargo must cover the maximum target horizon")
        if self.canonical_seed not in self.seeds:
            raise ValueError("canonical seed must be one of the robustness seeds")
        if self.bootstrap_block_sessions < self.max_horizon:
            raise ValueError("bootstrap blocks must cover the maximum horizon")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def canonical_json_digest(payload: Any) -> str:
    """Hash JSON-compatible payloads without formatting or key-order ambiguity."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require_numeric_only(news_provider: HistoricalNewsFeatureProvider | None) -> None:
    """Fail closed if a caller tries to activate the future news path in V11.2."""
    if news_provider is not None:
        raise ValueError("V11.2 is numeric-only; historical news is reserved for V12")


def protocol_manifest(protocol: V112Protocol | None = None) -> dict[str, Any]:
    selected = protocol or V112Protocol()
    payload = selected.to_dict()
    payload["protocol_sha256"] = selected.digest()
    return payload
