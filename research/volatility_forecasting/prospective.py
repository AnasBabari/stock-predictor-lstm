"""Pre-registered prospective retraining cycle after a consumed holdout.

The prior certification result is historical evidence only.  This module
defines the finite objective comparison, development cutoff, and selection
rule before any post-cutoff observation is loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .contracts import VolatilityForecastProtocol
from .model import VolatilityLossWeights

PROSPECTIVE_PROTOCOL_VERSION = "global-volatility-distribution-v7-prospective"
PROSPECTIVE_ARCHITECTURE_VERSION = "baseline-residual-tcn-v3-objective-selection"


@dataclass(frozen=True)
class ProspectiveObjectiveProfile:
    name: Literal["multitask_v1", "volatility_only_v1"]
    loss_weights: VolatilityLossWeights
    rationale: str


OBJECTIVE_PROFILES: dict[str, ProspectiveObjectiveProfile] = {
    "multitask_v1": ProspectiveObjectiveProfile(
        name="multitask_v1",
        loss_weights=VolatilityLossWeights(),
        rationale="Frozen v6 incumbent objective retained as the control.",
    ),
    "volatility_only_v1": ProspectiveObjectiveProfile(
        name="volatility_only_v1",
        loss_weights=VolatilityLossWeights(
            qlike=0.70,
            variance_crps=0.25,
            return_location=0.0,
            direction=0.0,
            baseline_regularization=0.05,
        ),
        rationale=(
            "Pre-certification v6 development evidence rejected every auxiliary "
            "return-distribution head, so the challenger removes their training gradients."
        ),
    ),
}


@dataclass(frozen=True)
class ProspectiveCycleSettings:
    development_cutoff: str = "2026-08-21"
    prospective_certification_start: str = "2026-08-27"
    required_horizons: tuple[int, ...] = (1, 3, 5, 7)
    profile_names: tuple[str, ...] = ("multitask_v1", "volatility_only_v1")
    incumbent_profile: str = "multitask_v1"
    challenger_profile: str = "volatility_only_v1"
    maximum_challenger_median_ratio: float = 0.995
    maximum_challenger_horizon_ratio: float = 1.01

    def __post_init__(self) -> None:
        if not self.required_horizons:
            raise ValueError("prospective cycle requires at least one horizon")
        if set(self.profile_names) != {self.incumbent_profile, self.challenger_profile}:
            raise ValueError("prospective cycle profiles do not match incumbent and challenger")
        if any(name not in OBJECTIVE_PROFILES for name in self.profile_names):
            raise ValueError("prospective cycle contains an unknown objective profile")
        if not 0 < self.maximum_challenger_median_ratio <= 1:
            raise ValueError("challenger median ratio must require improvement")
        if self.maximum_challenger_horizon_ratio < 1:
            raise ValueError("horizon guardrail cannot be stricter than non-degradation")


def prospective_protocol() -> VolatilityForecastProtocol:
    """Return the immutable protocol identity for the new development cycle."""
    return VolatilityForecastProtocol(
        protocol_version=PROSPECTIVE_PROTOCOL_VERSION,
        architecture_version=PROSPECTIVE_ARCHITECTURE_VERSION,
    )


def objective_manifest(profile: ProspectiveObjectiveProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "loss_weights": asdict(profile.loss_weights),
        "rationale": profile.rationale,
    }


def select_prospective_profile(
    consensus_by_profile: Mapping[str, Mapping[str, Mapping[str, object]]],
    settings: ProspectiveCycleSettings | None = None,
) -> dict[str, object]:
    """Apply the frozen whole-profile selection rule to all-seed evidence."""
    cycle = settings or ProspectiveCycleSettings()
    evidence: dict[str, dict[str, object]] = {}
    for profile_name in cycle.profile_names:
        raw_consensus = consensus_by_profile.get(profile_name)
        if not isinstance(raw_consensus, Mapping):
            raise ValueError(f"missing consensus for profile {profile_name}")
        horizon_values: dict[str, float] = {}
        promoted = True
        for horizon in cycle.required_horizons:
            row = raw_consensus.get(str(horizon))
            if not isinstance(row, Mapping):
                raise ValueError(f"profile {profile_name} is missing horizon {horizon}")
            value = row.get("relative_qlike_median")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
            ):
                raise ValueError(f"profile {profile_name} has invalid QLIKE evidence")
            horizon_values[str(horizon)] = float(value)
            promoted = promoted and row.get("promoted_all_seeds") is True
        evidence[profile_name] = {
            "eligible": promoted,
            "median_required_horizon_qlike": float(np.median(list(horizon_values.values()))),
            "relative_qlike_by_horizon": horizon_values,
        }

    incumbent = evidence[cycle.incumbent_profile]
    challenger = evidence[cycle.challenger_profile]
    if not incumbent["eligible"] and not challenger["eligible"]:
        return {
            "status": "abstain_no_all_horizon_profile",
            "selected_profile": None,
            "evidence": evidence,
            "reasons": ["neither pre-registered profile passed every required horizon and seed"],
        }
    if challenger["eligible"] and not incumbent["eligible"]:
        return {
            "status": "selected",
            "selected_profile": cycle.challenger_profile,
            "evidence": evidence,
            "reasons": ["challenger passed every gate while the incumbent did not"],
        }
    if incumbent["eligible"] and not challenger["eligible"]:
        return {
            "status": "selected",
            "selected_profile": cycle.incumbent_profile,
            "evidence": evidence,
            "reasons": ["incumbent passed every gate while the challenger did not"],
        }

    incumbent_values = incumbent["relative_qlike_by_horizon"]
    challenger_values = challenger["relative_qlike_by_horizon"]
    ratios = {
        str(horizon): float(challenger_values[str(horizon)] / incumbent_values[str(horizon)])
        for horizon in cycle.required_horizons
    }
    median_ratio = float(np.median(list(ratios.values())))
    worst_ratio = float(max(ratios.values()))
    challenger_wins = (
        median_ratio <= cycle.maximum_challenger_median_ratio
        and worst_ratio <= cycle.maximum_challenger_horizon_ratio
    )
    selected = cycle.challenger_profile if challenger_wins else cycle.incumbent_profile
    reason = (
        "challenger cleared the pre-registered median-improvement and horizon guardrails"
        if challenger_wins
        else "challenger did not clear the pre-registered incumbent displacement rule"
    )
    return {
        "status": "selected",
        "selected_profile": selected,
        "evidence": evidence,
        "challenger_to_incumbent": {
            "relative_qlike_by_horizon": ratios,
            "median_ratio": median_ratio,
            "worst_horizon_ratio": worst_ratio,
        },
        "reasons": [reason],
    }
