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


def build_prospective_development_fold_plan(
    examples: VolatilityPanelExamples,
    protocol: VolatilityForecastProtocol,
    *,
    development_cutoff: np.datetime64,
    prospective_certification_start: np.datetime64,
    asset_split_seed: int = 42,
    required_asset_holdouts: tuple[str, ...] = ("NMM", "MSFT"),
) -> VolatilityFoldPlan:
    """Build development folds before a genuinely future certification window.

    This plan is used after an earlier locked reserve has been consumed.  All
    examples through ``development_cutoff`` become ordinary development data,
    while certification arrays remain empty until observations strictly after
    ``prospective_certification_start`` are collected in a later snapshot.
    The caller must separately prove that target observations were also
    bounded by the immutable panel cutoff.
    """
    cutoff = np.datetime64(str(development_cutoff), "D")
    certification_start = np.datetime64(str(prospective_certification_start), "D")
    if np.isnat(cutoff) or np.isnat(certification_start):
        raise ValueError("prospective fold dates must be finite")
    if certification_start <= cutoff:
        raise ValueError("prospective certification must start after the development cutoff")
    if np.any(examples.origin_dates >= certification_start):
        raise ValueError("prospective certification rows are already present in development data")

    unique_dates = np.unique(examples.origin_dates[examples.origin_dates <= cutoff])
    unique_dates.sort()
    required_sessions = (
        protocol.minimum_train_sessions
        + protocol.embargo_sessions
        + protocol.folds * protocol.validation_sessions
    )
    if len(unique_dates) < required_sessions:
        raise ValueError(
            f"need at least {required_sessions} prospective development sessions, "
            f"got {len(unique_dates)}"
        )

    train_tickers, holdout_tickers = select_asset_holdouts(
        examples.tickers,
        fraction=protocol.asset_holdout_fraction,
        seed=asset_split_seed,
        required=required_asset_holdouts,
    )
    first_validation_position = len(unique_dates) - protocol.folds * protocol.validation_sessions
    training_ticker_mask = np.isin(examples.tickers, train_tickers)
    folds: list[VolatilityFold] = []
    for fold in range(protocol.folds):
        validation_start_position = first_validation_position + fold * protocol.validation_sessions
        validation_end_position = validation_start_position + protocol.validation_sessions
        train_end_position = validation_start_position - protocol.embargo_sessions
        if train_end_position < protocol.minimum_train_sessions:
            raise ValueError("prospective fold plan does not satisfy minimum training history")

        train_end = unique_dates[train_end_position - 1]
        validation_start = unique_dates[validation_start_position]
        validation_end = unique_dates[validation_end_position - 1]
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

    return VolatilityFoldPlan(
        folds=tuple(folds),
        train_tickers=train_tickers,
        asset_holdout_tickers=holdout_tickers,
        temporal_certification_indices=np.empty(0, dtype=np.int64),
        asset_transfer_certification_indices=np.empty(0, dtype=np.int64),
        certification_start=certification_start,
    )


def build_prospective_certification_fold_plan(
    examples: VolatilityPanelExamples,
    protocol: VolatilityForecastProtocol,
    *,
    development_cutoff: np.datetime64,
    prospective_certification_start: np.datetime64,
    asset_split_seed: int = 42,
    required_asset_holdouts: tuple[str, ...] = ("NMM", "MSFT"),
) -> VolatilityFoldPlan:
    """Reconstruct frozen development folds and the first matured future reserve.

    The certification reserve is the first ``temporal_holdout_sessions`` unique
    forecast origins on or after the preregistered start.  ``examples`` only
    contains origins whose maximum-horizon target is already observable, so
    requiring the full origin count also enforces target maturity without
    admitting a partial terminal horizon.
    """
    cutoff = np.datetime64(str(development_cutoff), "D")
    certification_start = np.datetime64(str(prospective_certification_start), "D")
    if np.isnat(cutoff) or np.isnat(certification_start):
        raise ValueError("prospective fold dates must be finite")
    if certification_start <= cutoff:
        raise ValueError("prospective certification must start after the development cutoff")

    unique_dates = np.unique(examples.origin_dates)
    unique_dates.sort()
    development_dates = unique_dates[unique_dates <= cutoff]
    future_dates = unique_dates[unique_dates >= certification_start]
    required_development_sessions = (
        protocol.minimum_train_sessions
        + protocol.embargo_sessions
        + protocol.folds * protocol.validation_sessions
    )
    if len(development_dates) < required_development_sessions:
        raise ValueError(
            f"need at least {required_development_sessions} prospective development sessions, "
            f"got {len(development_dates)}"
        )
    if len(future_dates) < protocol.temporal_holdout_sessions:
        raise ValueError(
            "prospective certification is not mature: need "
            f"{protocol.temporal_holdout_sessions} target-complete origin sessions on or after "
            f"{certification_start}, got {len(future_dates)}"
        )

    train_tickers, holdout_tickers = select_asset_holdouts(
        examples.tickers,
        fraction=protocol.asset_holdout_fraction,
        seed=asset_split_seed,
        required=required_asset_holdouts,
    )
    first_validation_position = len(development_dates) - (
        protocol.folds * protocol.validation_sessions
    )
    training_ticker_mask = np.isin(examples.tickers, train_tickers)
    holdout_ticker_mask = np.isin(examples.tickers, holdout_tickers)
    folds: list[VolatilityFold] = []
    for fold in range(protocol.folds):
        validation_start_position = first_validation_position + fold * protocol.validation_sessions
        validation_end_position = validation_start_position + protocol.validation_sessions
        train_end_position = validation_start_position - protocol.embargo_sessions
        if train_end_position < protocol.minimum_train_sessions:
            raise ValueError("prospective fold plan does not satisfy minimum training history")

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

    locked_dates = future_dates[: protocol.temporal_holdout_sessions]
    locked_start = locked_dates[0]
    locked_end = locked_dates[-1]
    temporal_mask = (
        training_ticker_mask
        & (examples.origin_dates >= locked_start)
        & (examples.origin_dates <= locked_end)
    )
    asset_transfer_mask = (
        holdout_ticker_mask
        & (examples.origin_dates >= locked_start)
        & (examples.origin_dates <= locked_end)
    )
    temporal_indices = np.flatnonzero(temporal_mask)
    transfer_indices = np.flatnonzero(asset_transfer_mask)
    if len(temporal_indices) == 0 or len(transfer_indices) == 0:
        raise ValueError("prospective certification reserves must both be non-empty")
    temporal_dates = np.unique(examples.origin_dates[temporal_indices])
    transfer_dates = np.unique(examples.origin_dates[transfer_indices])
    if not np.array_equal(temporal_dates, locked_dates) or not np.array_equal(
        transfer_dates,
        locked_dates,
    ):
        raise ValueError("prospective certification populations do not cover every locked date")
    if not set(required_asset_holdouts).issubset(
        {str(value).upper() for value in examples.tickers[transfer_indices]}
    ):
        raise ValueError("prospective certification is missing a required asset holdout")
    for ticker in required_asset_holdouts:
        ticker_dates = np.unique(
            examples.origin_dates[transfer_indices][examples.tickers[transfer_indices] == ticker]
        )
        if not np.array_equal(ticker_dates, locked_dates):
            raise ValueError(f"prospective certification has incomplete {ticker} date coverage")

    return VolatilityFoldPlan(
        folds=tuple(folds),
        train_tickers=train_tickers,
        asset_holdout_tickers=holdout_tickers,
        temporal_certification_indices=temporal_indices,
        asset_transfer_certification_indices=transfer_indices,
        certification_start=certification_start,
    )
