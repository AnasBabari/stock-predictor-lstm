"""Guards for the frozen v9 preregistration.

A preregistration is only worth writing if it cannot be quietly revised after an
inconvenient result appears.  These tests pin the machine-readable protocol in
``configs/volatility_v9_protocol.json`` to the values committed in
``docs/VOLATILITY_V9_PREREGISTRATION.md`` and fail closed on any drift.

What is deliberately **not** pinned here: a byte-level hash of the protocol
file.  A preregistration may legitimately be amended, but an amendment must be
visible — it changes ``protocol_version`` and it is recorded.  What must never
change silently are the fields that determine whether a result is real: the
metric and its argument order, the required horizons, the seed policy, the
split rule, and the sealed-test flag.  Those are pinned below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "configs" / "volatility_v9_protocol.json"
PREREG_DOC_PATH = REPO_ROOT / "docs" / "VOLATILITY_V9_PREREGISTRATION.md"


@pytest.fixture(scope="module")
def protocol() -> dict:
    if not PROTOCOL_PATH.exists():
        pytest.fail(f"v9 protocol missing at {PROTOCOL_PATH}")
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prereg_doc() -> str:
    if not PREREG_DOC_PATH.exists():
        pytest.fail(f"v9 preregistration document missing at {PREREG_DOC_PATH}")
    return PREREG_DOC_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Existence and structure
# --------------------------------------------------------------------------


def test_protocol_is_valid_json_with_expected_keys(protocol: dict) -> None:
    """The protocol must carry every section the preregistration promises."""
    required = {
        "protocol_version",
        "protocol_status",
        "preregistered_before_any_v9_training",
        "task",
        "target",
        "horizons",
        "feature_schema",
        "split",
        "periods",
        "asset_transfer_policy",
        "missing_data_policy",
        "folds",
        "seeding",
        "metrics",
        "promotion_gates",
        "model_family_namespace",
        "candidate_families",
        "experiment_budget",
        "decision_hierarchy",
        "news_controls",
        "data_eligibility",
        "integrity",
        "terminal_outcomes",
    }
    missing = required - set(protocol)
    assert not missing, f"protocol missing required sections: {sorted(missing)}"


def test_protocol_status_and_diagnostic_quarantine(protocol: dict) -> None:
    """The protocol status must be valid, and quarantined diagnostics remain quarantined."""
    assert protocol["protocol_status"] in ("draft_pre_freeze", "frozen")
    assert protocol["builds_on"] == ["volatility-v8"]
    assert protocol["does_not_supersede"] == ["volatility-v7", "volatility-v8"]
    assert "quarantined_pre_protocol_diagnostics" in protocol
    quarantine = protocol["quarantined_pre_protocol_diagnostics"]
    assert quarantine["status"] == "invalid_and_quarantined"
    assert "numeric_companion_freeze.json" in quarantine["files"]
    assert "weights.pt" in quarantine["files"]
    assert "QUARANTINE.md" in quarantine["files"]
    # No certified model may exist without a passed certification record.
    if protocol["integrity"].get("certified_model") is not None:
        assert protocol["integrity"].get("sealed_test_opened") is True


# --------------------------------------------------------------------------
# Target and metric — the fields that decide whether a result is real
# --------------------------------------------------------------------------


def test_target_definition_is_frozen(protocol: dict) -> None:
    target = protocol["target"]
    assert target["target_contract_version"] == "future_annualized_realized_variance_log_v1"
    assert target["transformation"] == "log"
    assert target["annualization_basis"] == 252
    # epsilon must be strictly positive (otherwise log(0)) and small enough that
    # it does not materially shift the target.
    assert 0 < target["epsilon"] <= 1e-6


def test_primary_metric_and_argument_order_are_frozen(protocol: dict) -> None:
    """QLIKE is asymmetric; reversing its arguments reverses rankings."""
    metrics = protocol["metrics"]
    assert metrics["primary_selection_metric"] == "qlike"
    assert metrics["qlike_argument_order"] == "qlike_losses(forecast, realized)"


def test_statistical_evidence_requirements(protocol: dict) -> None:
    stats = protocol["metrics"]["statistical_evidence"]
    assert stats["minimum_resamples"] >= 2000
    assert stats["confidence_level"] == 0.95
    # Daily volatility losses are autocorrelated; an i.i.d. bootstrap would
    # produce intervals that are far too narrow.
    assert "block" in stats["method"]
    assert stats["multiple_comparison_control"] == "holm"


# --------------------------------------------------------------------------
# Horizons
# --------------------------------------------------------------------------


def test_required_horizons_are_a_subset_of_evaluated(protocol: dict) -> None:
    horizons = protocol["horizons"]
    evaluated = horizons["evaluated"]
    required = horizons["required_for_promotion"]
    assert tuple(required) == (1, 3, 5, 7)
    assert set(required).issubset(set(evaluated))
    assert len(set(evaluated)) == len(evaluated), "duplicate evaluated horizon"


def test_embargo_covers_longest_required_horizon(protocol: dict) -> None:
    longest_required = max(protocol["horizons"]["required_for_promotion"])
    assert protocol["split"]["embargo_sessions"] >= longest_required


# --------------------------------------------------------------------------
# Split, folds, seeding
# --------------------------------------------------------------------------


def test_split_is_chronological_and_sealed(protocol: dict) -> None:
    split = protocol["split"]
    assert split["split_by"] == "forecast_origin_timestamp"
    assert split["random_split_allowed"] is False
    assert split["embargo_sessions"] > 0
    assert split["purge"]
    # The sealed test must not have been opened at preregistration time.
    assert split["sealed_test_opened"] is False


def test_split_fractions_sum_to_one(protocol: dict) -> None:
    split = protocol["split"]
    total = split["train_fraction"] + split["validation_fraction"] + split["test_fraction"]
    assert total == pytest.approx(1.0)


def test_inner_splits_are_date_aligned(protocol: dict) -> None:
    folds = protocol["folds"]
    assert folds["scheme"] == "expanding_window_chronological"
    assert folds["development_folds"] >= 3
    assert folds["inner_early_stopping_split"] == "date_aligned_purged_embargoed"
    assert folds["row_based_inner_split_allowed"] is False


def test_seed_policy_forbids_fake_replication(protocol: dict) -> None:
    seeding = protocol["seeding"]
    neural = seeding["neural_seeds"]
    assert len(neural) == 3
    assert len(set(neural)) == 3, "duplicate neural seed fabricates evidence volume"
    assert seeding["deterministic_seed"] == 0
    assert 0 not in neural, "deterministic seed must not masquerade as a neural replicate"


def test_periods_are_not_invented_before_the_panel_exists(protocol: dict) -> None:
    """Periods are derived at Stage 4, not guessed at preregistration time."""
    periods = protocol["periods"]
    assert periods["train_period"] is None
    assert periods["validation_period"] is None
    assert periods["sealed_test_period"] is None
    assert periods["status"] == "derived_from_panel_at_stage_4_then_frozen"


def test_asset_transfer_is_an_evaluation_surface_only(protocol: dict) -> None:
    policy = protocol["asset_transfer_policy"]
    assert policy["holdout_declared_before_training"] is True
    assert policy["evaluated_after_freeze"] is True
    for forbidden in ("training", "validation", "hyperparameter_search", "candidate_selection"):
        assert forbidden in policy["excluded_from"], f"{forbidden} must be excluded"


def test_missing_data_policy_forbids_silent_imputation(protocol: dict) -> None:
    policy = protocol["missing_data_policy"]
    assert policy["imputation"] == "none_silent"
    assert any("indicator" in rule for rule in policy["rules"])
    assert "abstention" in policy and policy["abstention"]


# --------------------------------------------------------------------------
# Feature schema and candidate families
# --------------------------------------------------------------------------


def test_feature_schema_counts(protocol: dict) -> None:
    schema = protocol["feature_schema"]
    assert schema["numeric_feature_count"] == 26
    assert schema["news_feature_count"] == 22
    assert schema["total_feature_count"] == 48
    assert (
        schema["numeric_feature_count"] + schema["news_feature_count"]
        == schema["total_feature_count"]
    )
    # News is defined in the schema but excluded from the numeric cycle.
    assert schema["news_included_in_this_cycle"] is False


def test_mislabelled_families_are_handled_explicitly(protocol: dict) -> None:
    families = protocol["candidate_families"]
    # A linear model relabelled as DLinear must not be evaluated.
    assert "dlinear" in families["conditional"]
    assert "conditional_rule" in families
    # A fixed GARCH/HAR blend is not a GARCH-LSTM.
    assert "garch_lstm" in families["excluded"]
    assert "excluded_rule" in families
    overlap = set(families["baselines"]) & set(families["neural"])
    assert not overlap, f"family listed as both baseline and neural: {sorted(overlap)}"


def test_model_family_namespace_is_bound_to_content(protocol: dict) -> None:
    ns = protocol["model_family_namespace"]
    assert ns["namespace"] == "global-volatility-v9"
    rules = " ".join(ns["rules"]).lower()
    assert "digest" in rules, "identities must be bound to a content digest"
    assert "reuse" in rules, "v9 must not reuse v7/v8 weights or identities"


# --------------------------------------------------------------------------
# Budget, hierarchy, integrity
# --------------------------------------------------------------------------


def test_experiment_budget_is_finite(protocol: dict) -> None:
    budget = protocol["experiment_budget"]
    assert budget["baseline_cycle"] == 1
    assert budget["primary_neural_cycle"] == 1
    assert budget["corrective_neural_cycle"] == 1
    assert "defect" in budget["corrective_cycle_condition"].lower()


def test_decision_hierarchy_is_total_and_ordered(protocol: dict) -> None:
    ranks = [entry["rank"] for entry in protocol["decision_hierarchy"]]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks), "duplicate decision rank"
    outcomes = [entry["outcome"] for entry in protocol["decision_hierarchy"]]
    # Abstention must be reachable: without it a failing programme has no exit.
    assert "retain_abstention" in outcomes


def test_data_ineligibility_is_recorded_not_hidden(protocol: dict) -> None:
    """Development-only data may not masquerade as certification-grade input."""
    eligibility = protocol["data_eligibility"]
    assert eligibility["universe_certification_eligible"] is False
    assert eligibility["market_panel_certification_eligible"] is False
    assert eligibility["blocker"]
    labelling = eligibility["labelling_requirement"].lower()
    assert "development_diagnostic_only" in labelling
    assert "certification_eligible=false" in labelling.replace(" ", "")


def test_integrity_flags_track_the_full_cycle(protocol: dict) -> None:
    integrity = protocol["integrity"]
    assert integrity["sealed_test_opened"] is False
    assert integrity["v8_sealed_test_opened"] is False
    assert integrity["v7_modified"] is False
    assert integrity["numeric_companion_frozen"] is False


def test_terminal_outcomes_include_a_credible_negative(protocol: dict) -> None:
    joined = " ".join(protocol["terminal_outcomes"]).lower()
    assert "baseline" in joined, "a baseline-only outcome must be preregistered as acceptable"
    assert "abstention" in joined, "abstention must be a preregistered outcome, not a failure"


# --------------------------------------------------------------------------
# Document / protocol consistency
# --------------------------------------------------------------------------


def test_preregistration_doc_matches_protocol_identities(protocol: dict, prereg_doc: str) -> None:
    """The prose document must state the frozen identities, not paraphrase them."""
    assert "global-volatility-v9" in prereg_doc
    assert protocol["target"]["target_contract_version"] in prereg_doc
    assert protocol["feature_schema"]["feature_contract_version"] in prereg_doc
    assert protocol["split"]["split_contract_version"] in prereg_doc


def test_preregistration_doc_states_the_data_blocker(protocol: dict, prereg_doc: str) -> None:
    """A reader of the document alone must learn that the data is not certifiable."""
    lowered = prereg_doc.lower()
    assert "certification eligible" in lowered
    assert "development-only" in lowered or "development only" in lowered
    assert protocol["data_eligibility"]["current_market_source"] in prereg_doc


def test_preregistration_doc_rejects_manufactured_winners(prereg_doc: str) -> None:
    lowered = prereg_doc.lower()
    assert "every" in lowered and "required horizon" in lowered
    assert "abstention" in lowered
