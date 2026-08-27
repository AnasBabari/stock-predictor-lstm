from __future__ import annotations

import numpy as np

from research.volatility_forecasting.candidate_v8 import (
    split_validation_for_selection,
    train_v8_numeric_ensemble,
)
from research.volatility_forecasting.data import VolatilityPanelExamples


def _examples() -> VolatilityPanelExamples:
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2022-01-01"))
    rows = len(dates)
    return VolatilityPanelExamples(
        features=np.ones((rows, 60, 26), dtype=np.float32),
        baseline_variance=np.ones((rows, 6), dtype=np.float32),
        realized_variance=np.ones((rows, 6), dtype=np.float32),
        cumulative_returns=np.zeros((rows, 6), dtype=np.float32),
        direction_classes=np.ones((rows, 6), dtype=np.int64),
        tickers=np.full(rows, "AAPL"),
        origin_dates=dates,
        origin_closes=np.ones(rows),
        horizons=(1, 3, 5, 7, 14, 30),
        feature_names=tuple(f"f{column}" for column in range(26)),
    )


def test_validation_selection_is_disjoint_and_purged() -> None:
    examples = _examples()
    partitions = split_validation_for_selection(examples, np.arange(len(examples.features)))

    assert not np.intersect1d(
        partitions.calibration_indices,
        partitions.selection_indices,
    ).size
    calibration_end = np.max(examples.origin_dates[partitions.calibration_indices])
    selection_start = np.min(examples.origin_dates[partitions.selection_indices])
    assert (selection_start - calibration_end).astype(int) > 30


def test_v8_numeric_model_identity_has_dedicated_namespace(monkeypatch) -> None:
    examples = _examples()
    validation = np.arange(200, len(examples.features))
    train = np.arange(0, 150)

    class _Member:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.model_identity = f"global-volatility:member-{seed}"

    monkeypatch.setattr(
        "research.volatility_forecasting.candidate_v8._fit_member",
        lambda **kwargs: _Member(kwargs["seed"]),
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.candidate_v8._evaluate_member",
        lambda candidate, *_args, **_kwargs: type(
            "Evidence",
            (),
            {"seed": candidate.seed},
        )(),
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.candidate_v8.FrozenEnsemble",
        lambda members, model_identity: type(
            "Ensemble",
            (),
            {"members": members, "model_identity": model_identity},
        )(),
    )

    ensemble, _evidence, _partitions = train_v8_numeric_ensemble(
        examples=examples,
        train_indices=train,
        validation_indices=validation,
        seeds=(41, 42, 43),
        required_horizons=(1, 3, 5, 7),
        device="cpu",
        maximum_epochs=1,
    )

    assert ensemble.model_identity.startswith("global-volatility-v8-numeric:")
