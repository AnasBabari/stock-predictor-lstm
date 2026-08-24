from __future__ import annotations

import numpy as np
import pytest
from volatility_forecasting.certification import (
    LockedCertificationGate,
    LockedPopulationInput,
    certify_locked_predictions,
)
from volatility_forecasting.data import VolatilityPanelExamples
from volatility_forecasting.folds import VolatilityFoldPlan
from volatility_forecasting.metrics import DistributionPredictions


def _fixture():
    rng = np.random.default_rng(4)
    horizons = (1, 7)
    sessions = np.arange(140, dtype="timedelta64[D]") + np.datetime64("2025-01-01")
    tickers = ("AAA", "BBB", "NMM", "MSFT")
    dates = np.repeat(sessions, len(tickers))
    names = np.tile(np.asarray(tickers), len(sessions))
    rows = len(dates)
    realized = rng.lognormal(mean=-7.0, sigma=0.35, size=(rows, 2))
    baseline = realized * rng.lognormal(mean=0.0, sigma=0.65, size=(rows, 2))
    model = realized * rng.lognormal(mean=0.0, sigma=0.08, size=(rows, 2))
    returns = rng.normal(0.0, np.sqrt(realized))
    classes = np.where(returns < -0.001, 0, np.where(returns > 0.001, 2, 1))
    examples = VolatilityPanelExamples(
        features=np.zeros((rows, 60, 1), dtype=np.float32),
        baseline_variance=baseline.astype(np.float32),
        realized_variance=realized.astype(np.float32),
        cumulative_returns=returns.astype(np.float32),
        direction_classes=classes.astype(np.int64),
        tickers=names,
        origin_dates=dates,
        origin_closes=np.full(rows, 100.0),
        horizons=horizons,
        feature_names=("x",),
    )
    temporal = np.flatnonzero(np.isin(names, ("AAA", "BBB")))
    transfer = np.flatnonzero(np.isin(names, ("NMM", "MSFT")))
    plan = VolatilityFoldPlan(
        folds=(),
        train_tickers=("AAA", "BBB"),
        asset_holdout_tickers=("MSFT", "NMM"),
        temporal_certification_indices=temporal,
        asset_transfer_certification_indices=transfer,
        certification_start=sessions[0],
    )

    def population(label, indices):
        probs = np.full((len(indices), 2, 3), 1 / 3, dtype=np.float64)
        predictions = DistributionPredictions(
            variance=model[indices],
            return_location=np.zeros((len(indices), 2)),
            direction_probabilities=probs,
            return_variance=model[indices],
        )
        return LockedPopulationInput(
            population=label,
            indices=indices,
            predictions=predictions,
            baseline_variance=baseline[indices],
            baseline_return_variance=baseline[indices],
        )

    return examples, plan, population("temporal", temporal), population("asset_transfer", transfer)


def test_locked_certification_passes_strong_candidate_on_complete_reserves() -> None:
    examples, plan, temporal, transfer = _fixture()
    report = certify_locked_predictions(
        examples=examples,
        fold_plan=plan,
        temporal=temporal,
        asset_transfer=transfer,
        model_identity="tcn-v6-seed-ensemble",
        development_evidence_sha256="a" * 64,
        gate=LockedCertificationGate(minimum_sessions=100),
        resamples=200,
    )
    assert report.status == "passed"
    assert report.certified_horizons == examples.horizons
    assert len(report.decisions) == 4
    assert {
        ticker for row in report.decisions for ticker in row.required_ticker_relative_qlike
    } == {
        "NMM",
        "MSFT",
    }


def test_locked_certification_rejects_reserve_slicing_and_missing_required_ticker() -> None:
    examples, plan, temporal, transfer = _fixture()
    sliced = LockedPopulationInput(
        population="asset_transfer",
        indices=transfer.indices[:-1],
        predictions=DistributionPredictions(
            variance=transfer.predictions.variance[:-1],
            return_location=transfer.predictions.return_location[:-1],
            direction_probabilities=transfer.predictions.direction_probabilities[:-1],
        ),
        baseline_variance=transfer.baseline_variance[:-1],
        baseline_return_variance=transfer.baseline_return_variance[:-1],
    )
    with pytest.raises(ValueError, match="locked reserve"):
        certify_locked_predictions(
            examples=examples,
            fold_plan=plan,
            temporal=temporal,
            asset_transfer=sliced,
            model_identity="candidate",
            development_evidence_sha256="b" * 64,
            gate=LockedCertificationGate(minimum_sessions=100),
            resamples=200,
        )


def test_locked_certification_fails_when_candidate_is_the_worse_forecast() -> None:
    examples, plan, temporal, transfer = _fixture()
    bad_variance = examples.realized_variance[temporal.indices] * 20
    bad_temporal = LockedPopulationInput(
        population="temporal",
        indices=temporal.indices,
        predictions=DistributionPredictions(
            variance=bad_variance,
            return_location=temporal.predictions.return_location,
            direction_probabilities=temporal.predictions.direction_probabilities,
        ),
        baseline_variance=temporal.baseline_variance,
        baseline_return_variance=temporal.baseline_return_variance,
    )
    report = certify_locked_predictions(
        examples=examples,
        fold_plan=plan,
        temporal=bad_temporal,
        asset_transfer=transfer,
        model_identity="bad-candidate",
        development_evidence_sha256="c" * 64,
        gate=LockedCertificationGate(minimum_sessions=100),
        resamples=200,
    )
    assert report.status == "failed"
    assert any(row.population == "temporal" and row.decision == "fail" for row in report.decisions)
