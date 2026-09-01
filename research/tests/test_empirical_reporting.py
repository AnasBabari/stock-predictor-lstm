from __future__ import annotations

from scripts.run_comprehensive_empirical_study import (
    _paired_bootstrap_summary,
    build_ablation_breadth,
    build_phase_3_5_audit,
)


def _study_rows(*, include_lstm_for_second: bool = True, feature_count: int = 11) -> dict:
    rows = []
    for ticker, qlike in (("AAA", 2.0), ("BBB", 3.0)):
        metrics = {
            "persistence": {"test": {"qlike": qlike, "mae": 1.0, "rmse": 1.2}},
            "ridge": {"test": {"qlike": qlike - 0.2, "mae": 0.9, "rmse": 1.1}},
        }
        if ticker == "AAA" or include_lstm_for_second:
            metrics["lstm"] = {"test": {"qlike": qlike - 0.1, "mae": 0.95, "rmse": 1.15}}
        rows.append({"ticker": ticker, "feature_count": feature_count, "metrics": metrics})
    return {"horizons": [1], "raw_results_by_horizon": {"h1": rows}}


def test_paired_bootstrap_is_reproducible_and_uses_asset_deltas() -> None:
    first = _paired_bootstrap_summary([2.0, 3.0, 5.0], [1.0, 2.0, 6.0], seed=7)
    second = _paired_bootstrap_summary([2.0, 3.0, 5.0], [1.0, 2.0, 6.0], seed=7)

    assert first == second
    assert first is not None
    assert first["asset_count"] == 3
    assert first["improved_assets"] == 2
    assert first["mean_delta"] == 1.0 / 3.0
    assert first["bootstrap_unit"] == "asset"


def test_ablation_breadth_requires_a_model_on_every_paired_asset() -> None:
    left = _study_rows(include_lstm_for_second=False, feature_count=11)
    right = _study_rows(include_lstm_for_second=True, feature_count=26)

    result = build_ablation_breadth(left, right, None)
    comparisons = result["comparisons"]
    models = {item["model"] for item in comparisons}

    assert models == {"persistence", "ridge"}
    persistence = next(item for item in comparisons if item["model"] == "persistence")
    assert persistence["from_feature_count"] == 11
    assert persistence["to_feature_count"] == 26
    assert persistence["metrics"]["qlike"]["asset_count"] == 2


def test_phase_audit_freezes_complete_validation_configuration_without_test_access() -> None:
    def study(mode: str, validation_qlike: float) -> dict:
        return {
            "universe_size": 2,
            "per_horizon_aggregates": {
                "h1": {
                    "horizon": 1,
                    "asset_count": 2,
                    "models": {
                        "ridge": {"val_qlike": validation_qlike, "asset_count": 2},
                    },
                }
            },
            "feature_mode": mode,
        }

    audit = build_phase_3_5_audit(
        study("price_plus_ohlc", 0.8),
        None,
        {
            "price_plus_ohlc": study("price_plus_ohlc", 0.8),
            "price_plus_ohlc_plus_market": study("price_plus_ohlc_plus_market", 0.7),
        },
    )

    frozen = audit["frozen_feature_configuration_by_horizon"]["1"]
    assert frozen["feature_mode"] == "price_plus_ohlc_plus_market"
    assert frozen["model"] == "ridge"
    assert frozen["selection_basis"] == "complete_coverage_aggregate_validation_qlike"
    assert frozen["test_partition_used"] is False
