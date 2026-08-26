from __future__ import annotations

import pytest
from volatility_forecasting.prospective import (
    OBJECTIVE_PROFILES,
    ProspectiveCycleSettings,
    objective_manifest,
    prospective_protocol,
    select_prospective_profile,
)


def _consensus(values: tuple[float, float, float, float], *, promoted: bool = True):
    return {
        str(horizon): {
            "promoted_all_seeds": promoted,
            "relative_qlike_median": value,
        }
        for horizon, value in zip((1, 3, 5, 7), values, strict=True)
    }


def test_prospective_protocol_and_objectives_are_frozen() -> None:
    protocol = prospective_protocol()
    assert protocol.protocol_version == "global-volatility-distribution-v7-prospective"
    assert set(OBJECTIVE_PROFILES) == {"multitask_v1", "volatility_only_v1"}
    challenger = objective_manifest(OBJECTIVE_PROFILES["volatility_only_v1"])
    assert challenger["loss_weights"]["return_location"] == 0.0
    assert challenger["loss_weights"]["direction"] == 0.0


def test_challenger_must_clear_median_and_worst_horizon_guardrails() -> None:
    result = select_prospective_profile(
        {
            "multitask_v1": _consensus((0.95, 0.94, 0.93, 0.92)),
            "volatility_only_v1": _consensus((0.94, 0.93, 0.92, 0.91)),
        }
    )
    assert result["selected_profile"] == "volatility_only_v1"

    retained = select_prospective_profile(
        {
            "multitask_v1": _consensus((0.95, 0.94, 0.93, 0.92)),
            "volatility_only_v1": _consensus((0.97, 0.92, 0.91, 0.90)),
        }
    )
    assert retained["selected_profile"] == "multitask_v1"


def test_selection_abstains_when_neither_profile_passes_all_horizons() -> None:
    result = select_prospective_profile(
        {
            "multitask_v1": _consensus((0.95, 0.94, 0.93, 0.92), promoted=False),
            "volatility_only_v1": _consensus((0.90, 0.89, 0.88, 0.87), promoted=False),
        }
    )
    assert result["status"] == "abstain_no_all_horizon_profile"
    assert result["selected_profile"] is None


def test_selection_rejects_missing_or_nonfinite_evidence() -> None:
    with pytest.raises(ValueError, match="missing consensus"):
        select_prospective_profile({"multitask_v1": _consensus((1, 1, 1, 1))})
    bad = _consensus((0.9, 0.9, 0.9, 0.9))
    bad["3"]["relative_qlike_median"] = float("nan")
    with pytest.raises(ValueError, match="invalid QLIKE"):
        select_prospective_profile(
            {
                "multitask_v1": _consensus((0.9, 0.9, 0.9, 0.9)),
                "volatility_only_v1": bad,
            },
            ProspectiveCycleSettings(),
        )
