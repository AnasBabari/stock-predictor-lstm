"""Date-aligned expanding folds, purge/embargo, and asset-transfer reserves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import VolatilityForecastProtocol
from .data import VolatilityPanelExamples


@dataclass(frozen=True)
class VolatilityFold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end: np.datetime64
    validation_start: np.datetime64
    validation_end: np.datetime64

    def __post_init__(self) -> None:
        if len(self.train_indices) == 0 or len(self.validation_indices) == 0:
            raise ValueError("fold train and validation sets must be non-empty")
        if self.train_end >= self.validation_start:
            raise ValueError("fold train end must precede validation start")
        if np.intersect1d(self.train_indices, self.validation_indices).size:
            raise ValueError("fold train and validation rows overlap")


@dataclass(frozen=True)
class VolatilityFoldPlan:
    folds: tuple[VolatilityFold, ...]
    train_tickers: tuple[str, ...]
    asset_holdout_tickers: tuple[str, ...]
    temporal_certification_indices: np.ndarray
    asset_transfer_certification_indices: np.ndarray
    certification_start: np.datetime64


@dataclass(frozen=True)
class InnerTrainingSplit:
    """Fit/early-stop split constructed solely inside an outer training fold."""

    fit_indices: np.ndarray
    early_stopping_indices: np.ndarray
    fit_end: np.datetime64
    early_stopping_start: np.datetime64
    early_stopping_end: np.datetime64

    def __post_init__(self) -> None:
        if len(self.fit_indices) == 0 or len(self.early_stopping_indices) == 0:
            raise ValueError("inner fit and early-stopping sets must be non-empty")
        if self.fit_end >= self.early_stopping_start:
            raise ValueError("inner fit end must precede early-stopping start")
        if np.intersect1d(self.fit_indices, self.early_stopping_indices).size:
            raise ValueError("inner fit and early-stopping rows overlap")


def select_asset_holdouts(
    tickers: np.ndarray,
    *,
    fraction: float,
    seed: int,
    required: tuple[str, ...] = ("NMM", "MSFT"),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Choose deterministic unseen assets while reserving requested symbols."""
    universe = sorted({str(ticker).upper() for ticker in tickers})
    if len(universe) < 3:
        raise ValueError("asset-transfer evaluation requires at least three tickers")
    if not 0 < fraction < 0.5:
        raise ValueError("asset holdout fraction must be in (0, 0.5)")
    holdout_count = max(1, int(round(len(universe) * fraction)))
    required_present = [ticker for ticker in required if ticker in universe]
    holdout_count = max(holdout_count, len(required_present))
    if holdout_count >= len(universe):
        raise ValueError("asset holdout would leave no training tickers")

    remaining = [ticker for ticker in universe if ticker not in required_present]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    selected = sorted(required_present + remaining[: holdout_count - len(required_present)])
    training = sorted(set(universe) - set(selected))
    return tuple(training), tuple(selected)


def build_inner_training_split(
    examples: VolatilityPanelExamples,
    outer_train_indices: np.ndarray,
    protocol: VolatilityForecastProtocol,
) -> InnerTrainingSplit:
    """Reserve a purged inner tail for epoch selection without touching OOF rows."""
    outer = np.asarray(outer_train_indices, dtype=np.int64)
    if outer.ndim != 1 or len(outer) == 0:
        raise ValueError("outer training indices must be a non-empty vector")
    dates = np.unique(examples.origin_dates[outer])
    dates.sort()
    required = protocol.early_stopping_sessions + protocol.embargo_sessions + 1
    if len(dates) < required:
        raise ValueError("outer training fold is too short for the purged early-stopping reserve")
    early_start_position = len(dates) - protocol.early_stopping_sessions
    fit_end_position = early_start_position - protocol.embargo_sessions
    if fit_end_position < 1:
        raise ValueError("inner split leaves no fitting session")
    fit_end = dates[fit_end_position - 1]
    early_start = dates[early_start_position]
    early_end = dates[-1]
    outer_dates = examples.origin_dates[outer]
    fit = outer[outer_dates <= fit_end]
    early = outer[(outer_dates >= early_start) & (outer_dates <= early_end)]
    return InnerTrainingSplit(
        fit_indices=fit,
        early_stopping_indices=early,
        fit_end=fit_end,
        early_stopping_start=early_start,
        early_stopping_end=early_end,
    )


def build_volatility_fold_plan(
    examples: VolatilityPanelExamples,
    protocol: VolatilityForecastProtocol | None = None,
    *,
    asset_split_seed: int = 42,
    required_asset_holdouts: tuple[str, ...] = ("NMM", "MSFT"),
) -> VolatilityFoldPlan:
    """Build five expanding development folds and locked certification rows."""
    contract = protocol or VolatilityForecastProtocol()
    unique_dates = np.unique(examples.origin_dates)
    unique_dates.sort()
    required_sessions = (
        contract.temporal_holdout_sessions
        + contract.minimum_train_sessions
        + contract.embargo_sessions
        + contract.folds * contract.validation_sessions
    )
    if len(unique_dates) < required_sessions:
        raise ValueError(
            f"need at least {required_sessions} unique origin sessions, got {len(unique_dates)}"
        )

    train_tickers, holdout_tickers = select_asset_holdouts(
        examples.tickers,
        fraction=contract.asset_holdout_fraction,
        seed=asset_split_seed,
        required=required_asset_holdouts,
    )
    development_dates = unique_dates[: -contract.temporal_holdout_sessions]
    certification_dates = unique_dates[-contract.temporal_holdout_sessions :]
    first_validation_position = len(development_dates) - (
        contract.folds * contract.validation_sessions
    )
    training_ticker_mask = np.isin(examples.tickers, train_tickers)
    holdout_ticker_mask = np.isin(examples.tickers, holdout_tickers)
    folds: list[VolatilityFold] = []

    for fold in range(contract.folds):
        validation_start_position = first_validation_position + fold * contract.validation_sessions
        validation_end_position = validation_start_position + contract.validation_sessions
        train_end_position = validation_start_position - contract.embargo_sessions
        if train_end_position < contract.minimum_train_sessions:
            raise ValueError("fold plan does not satisfy minimum training history")

        train_end = development_dates[train_end_position - 1]
        validation_start = development_dates[validation_start_position]
        validation_end = development_dates[validation_end_position - 1]
        train_mask = training_ticker_mask & (examples.origin_dates <= train_end)
        validation_mask = (
            training_ticker_mask
            & (examples.origin_dates >= validation_start)
            & (examples.origin_dates <= validation_end)
        )
        folds.append(
            VolatilityFold(
                fold=fold + 1,
                train_indices=np.flatnonzero(train_mask),
                validation_indices=np.flatnonzero(validation_mask),
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )

    certification_start = certification_dates[0]
    temporal_mask = training_ticker_mask & (examples.origin_dates >= certification_start)
    asset_transfer_mask = holdout_ticker_mask & (examples.origin_dates >= certification_start)
    temporal_indices = np.flatnonzero(temporal_mask)
    transfer_indices = np.flatnonzero(asset_transfer_mask)
    if len(temporal_indices) == 0 or len(transfer_indices) == 0:
        raise ValueError("certification reserves must contain temporal and asset-transfer rows")

    return VolatilityFoldPlan(
        folds=tuple(folds),
        train_tickers=train_tickers,
        asset_holdout_tickers=holdout_tickers,
        temporal_certification_indices=temporal_indices,
        asset_transfer_certification_indices=transfer_indices,
        certification_start=certification_start,
    )
