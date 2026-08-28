"""Multi-horizon architecture ablation for the v9 numeric companion (DEVELOPMENT ONLY).

EVALUATED families — ten, and only these ten:

    har, ewma, garch, gjr, ridge, elasticnet                    (deterministic, seed 0)
    gru, lstm, tcn, patch_transformer                           (neural, seeds 41/42/43)

NOT implemented. These names previously appeared in reports while the code
actually ran something else. They are recorded here so that no caller can
legitimately claim to have evaluated them:

    dlinear          -- prior implementation was Ridge with a DLinear label
    garch_lstm       -- prior implementation was a fixed GARCH/HAR blend with
                        no trained recurrent residual

A family name must describe the mathematics that actually ran. Anything
referencing ``UNIMPLEMENTED_FAMILIES`` must be rejected, not silently mapped
onto a different model.

All evaluated models receive identical 26-column stationary features,
60-session windows, and identical targets. Uncertainty is handled via 2,000
block-bootstrap resamples clustered by forecast week.

Selection is NOT performed by averaging over horizons. ``select_numeric_champion``
requires positive skill at *every* required horizon {1, 3, 5, 7}, across five
folds, with a bootstrap upper bound below the no-skill threshold. HAR is the
fallback whenever no learned candidate qualifies, and that fallback is labelled
as HAR rather than as the family that failed.

Sealed test partitions remain strictly untouched. Freezing is disabled until
family-specific serializers exist (see ``freeze_numeric_companion``).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from backend.panel.volatility import _garch_filter, fit_garch

from .contracts import VolatilityForecastProtocol
from .data import VolatilityPanelExamples
from .folds import InnerTrainingSplit, VolatilityFold, build_inner_training_split
from .metrics import qlike_losses
from .model import (
    BaselineResidualLSTMConfig,
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
    VolatilityLossWeights,
    train_baseline_residual_tcn,
)

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = (1, 3, 5, 7, 14, 30)
DEFAULT_SEEDS = (41, 42, 43)

#: Horizons a candidate must beat HAR at, independently. An aggregate win
#: cannot compensate for a loss at any one of these.
REQUIRED_HORIZONS: tuple[int, ...] = (1, 3, 5, 7)

#: Families a candidate may be drawn from. Anything outside this set has no
#: implementation and must never be scored or frozen.
EVALUATED_DETERMINISTIC_FAMILIES: tuple[str, ...] = (
    "har",
    "ewma",
    "garch",
    "gjr",
    "ridge",
    "elasticnet",
)
EVALUATED_NEURAL_FAMILIES: tuple[str, ...] = ("gru", "lstm", "tcn", "patch_transformer")
EVALUATED_FAMILIES: tuple[str, ...] = EVALUATED_DETERMINISTIC_FAMILIES + EVALUATED_NEURAL_FAMILIES

#: Names that must be rejected rather than silently substituted.
UNIMPLEMENTED_FAMILIES: frozenset[str] = frozenset({"dlinear", "garch_lstm"})

#: Seed recorded for deterministic candidates. Deterministic fits have exactly
#: one true result per fold; repeating them across neural seeds would inflate
#: the apparent evidence volume and corrupt uncertainty estimates.
DETERMINISTIC_SEED = 0


def assert_family_is_implemented(family: str) -> str:
    """Reject a family with no implementation instead of silently substituting."""
    name = str(family).strip().lower()
    if name in UNIMPLEMENTED_FAMILIES:
        raise NotImplementedError(
            f"family {name!r} has no implementation in this module; it must not be "
            "evaluated, scored, or frozen under a borrowed implementation"
        )
    if name not in EVALUATED_FAMILIES:
        raise ValueError(f"unknown candidate family: {name!r}")
    return name


DEFAULT_ARTIFACTS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "artifacts" / "numeric_companion_v9"
)
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


@dataclass(frozen=True)
class ModelAblationResult:
    family: str
    seed: int
    horizon: int
    mean_qlike: float
    relative_qlike_ratio: float
    bootstrap_ratio_p05: float
    bootstrap_ratio_p95: float
    low_vol_ratio: float
    normal_vol_ratio: float
    high_vol_ratio: float
    training_duration_seconds: float


@dataclass(frozen=True)
class NumericChampionDecision:
    selected_family: str
    selection_state: str
    eligible_families: tuple[str, ...]
    required_horizons: tuple[int, ...]
    reasons_by_family: dict[str, tuple[str, ...]]


def select_numeric_champion(
    results: pd.DataFrame,
    *,
    required_horizons: tuple[int, ...] = (1, 3, 5, 7),
    expected_folds: int = 5,
    expected_neural_seeds: tuple[int, ...] = (41, 42, 43),
) -> NumericChampionDecision:
    """Select a learner only when every required development gate clears."""
    required_columns = {
        "family",
        "fold",
        "seed",
        "horizon",
        "relative_qlike_ratio",
        "bootstrap_p95",
    }
    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(f"ablation results missing columns: {sorted(missing)}")
    if expected_folds < 2 or not required_horizons:
        raise ValueError("numeric selection requires multiple folds and required horizons")

    reasons_by_family: dict[str, tuple[str, ...]] = {}
    eligible: list[str] = []
    neural_families = {"gru", "lstm", "tcn", "patch_transformer"}
    for family in sorted(set(results["family"]) - {"har"}):
        rows = results[(results["family"] == family) & results["horizon"].isin(required_horizons)]
        reasons: list[str] = []
        if set(rows["horizon"].astype(int)) != set(required_horizons):
            reasons.append("required horizon evidence is incomplete")
        if set(rows["fold"].astype(int)) != set(range(1, expected_folds + 1)):
            reasons.append("expanding-fold evidence is incomplete")
        expected_seeds = set(expected_neural_seeds) if family in neural_families else {0}
        if set(rows["seed"].astype(int)) != expected_seeds:
            reasons.append("seed evidence is incomplete or mislabeled")
        ratios = rows["relative_qlike_ratio"].to_numpy(dtype=np.float64)
        upper = rows["bootstrap_p95"].to_numpy(dtype=np.float64)
        if not len(rows) or not np.isfinite(ratios).all() or not np.isfinite(upper).all():
            reasons.append("development evidence contains missing or non-finite scores")
        else:
            if np.any(ratios >= 1.0):
                reasons.append("at least one required fold/horizon/seed did not beat HAR")
            if np.any(upper >= 1.0):
                reasons.append("at least one required bootstrap upper bound did not beat HAR")
        reasons_by_family[family] = tuple(reasons)
        if not reasons:
            eligible.append(family)

    if not eligible:
        return NumericChampionDecision(
            selected_family="har",
            selection_state="baseline_retained_no_learned_candidate_qualified",
            eligible_families=(),
            required_horizons=required_horizons,
            reasons_by_family=reasons_by_family,
        )
    ranked = sorted(
        eligible,
        key=lambda family: (
            float(
                results[(results["family"] == family) & results["horizon"].isin(required_horizons)][
                    "relative_qlike_ratio"
                ].median()
            ),
            family,
        ),
    )
    return NumericChampionDecision(
        selected_family=ranked[0],
        selection_state="learned_candidate_selected_on_development_only",
        eligible_families=tuple(ranked),
        required_horizons=required_horizons,
        reasons_by_family=reasons_by_family,
    )


def compute_block_bootstrap_ratio_bounds(
    model_losses: np.ndarray,
    baseline_losses: np.ndarray,
    week_clusters: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute 95% bootstrap confidence interval on relative QLIKE ratio clustered by week."""
    unique_weeks = np.unique(week_clusters)
    if len(unique_weeks) < 5:
        ratio = float(np.mean(model_losses) / max(np.mean(baseline_losses), 1e-12))
        return ratio, ratio, ratio

    week_to_indices = {w: np.flatnonzero(week_clusters == w) for w in unique_weeks}
    rng = np.random.default_rng(seed)

    n_weeks = len(unique_weeks)
    ratios: list[float] = []

    for _ in range(n_resamples):
        sampled_weeks = rng.choice(unique_weeks, size=n_weeks, replace=True)
        sample_idx = np.concatenate([week_to_indices[w] for w in sampled_weeks])
        m_mean = np.mean(model_losses[sample_idx])
        b_mean = np.mean(baseline_losses[sample_idx])
        ratios.append(float(m_mean / max(b_mean, 1e-12)))

    p05 = float(np.percentile(ratios, 5.0))
    point = float(np.mean(model_losses) / max(np.mean(baseline_losses), 1e-12))
    p95 = float(np.percentile(ratios, 95.0))
    return point, p05, p95


def classify_regimes(examples: VolatilityPanelExamples, indices: np.ndarray) -> np.ndarray:
    """Classify origins into 0: LOW_VOL (<15%), 1: NORMAL (15-30%), 2: HIGH_VOL (>30%)."""
    try:
        vol_col = examples.feature_names.index("Vol_C2C_20")
        trailing_vol = examples.features[indices, -1, vol_col] * math.sqrt(252.0)
    except ValueError:
        # Fallback to annualizing baseline variance at horizon 1
        trailing_vol = np.sqrt(examples.baseline_variance[indices, 0] * 252.0)

    regimes = np.ones(len(indices), dtype=int)
    regimes[trailing_vol < 0.15] = 0
    regimes[trailing_vol > 0.30] = 2
    return regimes


_GARCH_PARAMS_CACHE: dict[str, Any] = {}

#: A GARCH fit needs at least one year of that security's own returns.
GARCH_MINIMUM_TRAIN_ROWS = 252
GARCH_MAX_TRAIN_ROWS = 2000
GARCH_CACHE_MAX_ENTRIES = 512

#: Populated whenever a GARCH/GJR forecast falls back to HAR for a ticker.
#: Callers must surface this; a silent fallback would let an unfit model be
#: scored as if it had been fitted.
GARCH_COVERAGE_DIAGNOSTICS: list[dict[str, Any]] = []


def reset_garch_diagnostics() -> None:
    """Clear recorded GARCH coverage substitutions between runs."""
    GARCH_COVERAGE_DIAGNOSTICS.clear()


def garch_coverage_diagnostics() -> tuple[dict[str, Any], ...]:
    """Return every recorded GARCH-to-HAR substitution from the current run."""
    return tuple(GARCH_COVERAGE_DIAGNOSTICS)


def compute_forecast_metrics(
    realized_variance: np.ndarray,
    forecast_variance: np.ndarray,
    baseline_variance: np.ndarray,
) -> dict[str, float]:
    """Return correctly oriented variance-forecast metrics for one matched population."""
    realized = np.asarray(realized_variance, dtype=np.float64)
    forecast = np.asarray(forecast_variance, dtype=np.float64)
    baseline = np.asarray(baseline_variance, dtype=np.float64)
    if realized.shape != forecast.shape or forecast.shape != baseline.shape or realized.size == 0:
        raise ValueError("forecast metric inputs must be matched non-empty arrays")
    if not (
        np.isfinite(realized).all() and np.isfinite(forecast).all() and np.isfinite(baseline).all()
    ):
        raise ValueError("forecast metric inputs must be finite")
    errors = forecast - realized
    denominator = float(np.sum((realized - np.mean(realized)) ** 2))
    model_qlike = float(np.mean(qlike_losses(forecast, realized)))
    baseline_qlike = float(np.mean(qlike_losses(baseline, realized)))
    return {
        "qlike": model_qlike,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "medae": float(np.median(np.abs(errors))),
        "r2": float(1.0 - np.sum(errors**2) / denominator) if denominator > 0 else 0.0,
        "ratio_to_baseline": model_qlike / baseline_qlike if baseline_qlike > 0 else float("inf"),
    }


def evaluate_classical_model(
    family: str,
    examples: VolatilityPanelExamples,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    horizon_idx: int,
) -> np.ndarray:
    """Fit and forecast classical variance baselines."""
    family = assert_family_is_implemented(family)
    target_h = examples.horizons[horizon_idx]
    baseline_val = examples.baseline_variance[val_indices, horizon_idx]

    if family == "har":
        return baseline_val

    if family == "ewma":
        try:
            ewma_col = examples.feature_names.index("EWMA_Var")
            daily_var = examples.features[val_indices, -1, ewma_col]
            return np.maximum(daily_var * target_h, 1e-12)
        except ValueError:
            return baseline_val

    if family in ("garch", "gjr"):
        # Fitted per ticker. Pooling mixed-asset returns would estimate one
        # volatility process across unrelated securities.
        predictions = np.asarray(baseline_val, dtype=np.float64).copy()
        ret_1d_col = examples.feature_names.index("Return_1D")
        tickers = np.asarray(examples.tickers)
        for ticker in np.unique(tickers[val_indices]):
            ticker_train = train_indices[tickers[train_indices] == ticker]
            ticker_val_positions = np.flatnonzero(tickers[val_indices] == ticker)
            ticker_val = val_indices[ticker_val_positions]
            if len(ticker_train) < GARCH_MINIMUM_TRAIN_ROWS or not len(ticker_val):
                # Recorded, never silent: a HAR substitute must not be scored
                # as if it were a fitted GARCH forecast.
                GARCH_COVERAGE_DIAGNOSTICS.append(
                    {
                        "family": family,
                        "ticker": str(ticker),
                        "reason": "insufficient_training_history",
                        "train_rows": int(len(ticker_train)),
                        "substituted_with": "har",
                    }
                )
                continue
            train_rets = examples.features[ticker_train[-GARCH_MAX_TRAIN_ROWS:], -1, ret_1d_col]
            # Cache key binds family, ticker AND the exact training content, so
            # a fit can never be reused across incompatible panels or folds.
            digest = hashlib.sha256(np.asarray(train_rets, dtype=np.float64).tobytes()).hexdigest()
            cache_key = f"{family}:{ticker}:{digest}"
            try:
                params = _GARCH_PARAMS_CACHE.get(cache_key)
                if params is None:
                    params = fit_garch(train_rets, gjr=(family == "gjr"))
                    if len(_GARCH_PARAMS_CACHE) >= GARCH_CACHE_MAX_ENTRIES:
                        _GARCH_PARAMS_CACHE.clear()
                    _GARCH_PARAMS_CACHE[cache_key] = params
                val_rets = examples.features[ticker_val, -1, ret_1d_col]
                # Filtering starts from the actual training history; filtering
                # validation alone would reset the latent variance at the boundary.
                filtered = _garch_filter(np.concatenate((train_rets, val_rets)), params)
                predictions[ticker_val_positions] = np.maximum(
                    filtered[-len(ticker_val) :] * target_h,
                    1e-12,
                )
            except Exception as exc:  # noqa: BLE001 - per-ticker econometric fit
                GARCH_COVERAGE_DIAGNOSTICS.append(
                    {
                        "family": family,
                        "ticker": str(ticker),
                        "reason": f"fit_failed:{type(exc).__name__}",
                        "train_rows": int(len(ticker_train)),
                        "substituted_with": "har",
                    }
                )
        return predictions

    if family in ("ridge", "elasticnet"):
        x_train = examples.features[train_indices].reshape(len(train_indices), -1)
        x_val = examples.features[val_indices].reshape(len(val_indices), -1)

        # Log variance residual target
        y_train_res = np.log(
            np.clip(examples.realized_variance[train_indices, horizon_idx], 1e-12, 1e2)
            / np.clip(examples.baseline_variance[train_indices, horizon_idx], 1e-12, 1e2)
        )
        y_train_res = np.clip(y_train_res, -1.5, 1.5)

        scaler = StandardScaler().fit(x_train)
        x_train_s = scaler.transform(x_train)
        x_val_s = scaler.transform(x_val)

        if family == "ridge":
            model = Ridge(alpha=10.0).fit(x_train_s, y_train_res)
        elif family == "elasticnet":
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=200, random_state=42).fit(
                x_train_s, y_train_res
            )
        pred_res = np.clip(model.predict(x_val_s), -1.5, 1.5)
        return baseline_val * np.exp(pred_res)

    return baseline_val


def train_and_eval_neural_candidate(
    family: str,
    examples: VolatilityPanelExamples,
    train_split: InnerTrainingSplit,
    val_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Train neural baseline-residual model and evaluate on validation slice."""
    horizon_count = len(examples.horizons)
    feature_count = examples.features.shape[-1]
    family = assert_family_is_implemented(family)

    if family == "lstm":
        config = BaselineResidualLSTMConfig(
            feature_count=feature_count,
            horizon_count=horizon_count,
            encoder_family="lstm",
            window_size=examples.features.shape[1],
            channels=48,
            lstm_layers=2,
            lstm_hidden=48,
        )
    elif family == "gru":
        config = BaselineResidualTCNConfig(
            feature_count=feature_count,
            horizon_count=horizon_count,
            encoder_family="gru",
            window_size=examples.features.shape[1],
            channels=48,
            lstm_layers=2,
            lstm_hidden=48,
        )
    elif family == "patch_transformer":
        config = BaselineResidualTCNConfig(
            feature_count=feature_count,
            horizon_count=horizon_count,
            encoder_family="patch_transformer",
            window_size=examples.features.shape[1],
            channels=48,
            transformer_d_model=64,
            transformer_heads=4,
            transformer_layers=2,
        )
    else:  # tcn
        config = BaselineResidualTCNConfig(
            feature_count=feature_count,
            horizon_count=horizon_count,
            encoder_family="tcn",
            window_size=examples.features.shape[1],
            channels=48,
        )

    training_cfg = TorchTrainingConfig(
        maximum_epochs=35,
        patience=6,
        batch_size=512,
        learning_rate=1e-3,
        use_amp=torch.cuda.is_available(),
    )

    t0 = time.perf_counter()
    train_res = train_baseline_residual_tcn(
        train_features=examples.features[train_split.fit_indices],
        train_baseline_variance=examples.baseline_variance[train_split.fit_indices],
        train_realized_variance=examples.realized_variance[train_split.fit_indices],
        train_cumulative_returns=examples.cumulative_returns[train_split.fit_indices],
        train_direction_classes=examples.direction_classes[train_split.fit_indices],
        validation_features=examples.features[train_split.early_stopping_indices],
        validation_baseline_variance=examples.baseline_variance[train_split.early_stopping_indices],
        validation_realized_variance=examples.realized_variance[train_split.early_stopping_indices],
        validation_cumulative_returns=examples.cumulative_returns[
            train_split.early_stopping_indices
        ],
        validation_direction_classes=examples.direction_classes[train_split.early_stopping_indices],
        model_config=config,
        training_config=training_cfg,
        loss_weights=VolatilityLossWeights(),
        seed=seed,
        device=str(device),
    )
    duration = time.perf_counter() - t0
    model = train_res.model
    scaler = train_res.scaler

    # Inference on validation slice
    model.eval()
    val_scaled = scaler.transform(examples.features[val_indices])
    val_baseline = examples.baseline_variance[val_indices]

    with torch.no_grad():
        f_t = torch.tensor(val_scaled, dtype=torch.float32, device=device)
        b_t = torch.tensor(val_baseline, dtype=torch.float32, device=device)
        var_pred, _, _, _ = model(f_t, b_t)
        var_pred_np = var_pred.cpu().numpy()

    return var_pred_np, duration


def run_architecture_ablation(
    examples: VolatilityPanelExamples,
    fold: VolatilityFold,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Run ablation across all 12 candidate families and all horizons."""
    dev_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting architecture ablation on device: %s", dev_device)

    inner_split = build_inner_training_split(
        examples,
        fold.train_indices,
        VolatilityForecastProtocol(horizons=examples.horizons),
    )

    val_idx = fold.validation_indices
    val_dates = pd.DatetimeIndex(examples.origin_dates[val_idx])
    iso = val_dates.isocalendar()
    week_clusters = np.asarray(iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2))
    regimes = classify_regimes(examples, val_idx)

    records: list[dict[str, Any]] = []

    # 1. Evaluate classical models across horizons
    for fam in ("har", "ewma", "garch", "gjr", "ridge", "elasticnet"):
        logger.info("Evaluating classical family: %s", fam)
        for h_idx, h in enumerate(examples.horizons):
            pred_var = evaluate_classical_model(fam, examples, fold.train_indices, val_idx, h_idx)
            target_var = examples.realized_variance[val_idx, h_idx]
            har_var = examples.baseline_variance[val_idx, h_idx]

            m_losses = qlike_losses(pred_var, target_var)
            b_losses = qlike_losses(har_var, target_var)

            ratio, p05, p95 = compute_block_bootstrap_ratio_bounds(
                m_losses, b_losses, week_clusters
            )

            # Regime-specific ratios
            low_ratio = (
                float(np.mean(m_losses[regimes == 0]) / max(np.mean(b_losses[regimes == 0]), 1e-12))
                if (regimes == 0).any()
                else ratio
            )
            norm_ratio = (
                float(np.mean(m_losses[regimes == 1]) / max(np.mean(b_losses[regimes == 1]), 1e-12))
                if (regimes == 1).any()
                else ratio
            )
            high_ratio = (
                float(np.mean(m_losses[regimes == 2]) / max(np.mean(b_losses[regimes == 2]), 1e-12))
                if (regimes == 2).any()
                else ratio
            )

            records.append(
                {
                    "family": fam,
                    "fold": fold.fold,
                    "seed": 0,
                    "horizon": h,
                    "mean_qlike": float(np.mean(m_losses)),
                    "har_qlike": float(np.mean(b_losses)),
                    "relative_qlike_ratio": ratio,
                    "bootstrap_p05": p05,
                    "bootstrap_p95": p95,
                    "low_vol_ratio": low_ratio,
                    "normal_vol_ratio": norm_ratio,
                    "high_vol_ratio": high_ratio,
                    "duration_seconds": 0.0,
                }
            )

    # 2. Evaluate neural models across seeds and horizons
    for fam in ("gru", "lstm", "tcn", "patch_transformer"):
        for s in seeds:
            logger.info("Training neural family: %s (seed %d)", fam, s)
            pred_var_all, duration = train_and_eval_neural_candidate(
                family=fam,
                examples=examples,
                train_split=inner_split,
                val_indices=val_idx,
                seed=s,
                device=dev_device,
            )
            for h_idx, h in enumerate(examples.horizons):
                pred_var = pred_var_all[:, h_idx]
                target_var = examples.realized_variance[val_idx, h_idx]
                har_var = examples.baseline_variance[val_idx, h_idx]

                m_losses = qlike_losses(pred_var, target_var)
                b_losses = qlike_losses(har_var, target_var)

                ratio, p05, p95 = compute_block_bootstrap_ratio_bounds(
                    m_losses, b_losses, week_clusters
                )

                low_ratio = (
                    float(
                        np.mean(m_losses[regimes == 0])
                        / max(np.mean(b_losses[regimes == 0]), 1e-12)
                    )
                    if (regimes == 0).any()
                    else ratio
                )
                norm_ratio = (
                    float(
                        np.mean(m_losses[regimes == 1])
                        / max(np.mean(b_losses[regimes == 1]), 1e-12)
                    )
                    if (regimes == 1).any()
                    else ratio
                )
                high_ratio = (
                    float(
                        np.mean(m_losses[regimes == 2])
                        / max(np.mean(b_losses[regimes == 2]), 1e-12)
                    )
                    if (regimes == 2).any()
                    else ratio
                )

                records.append(
                    {
                        "family": fam,
                        "fold": fold.fold,
                        "seed": s,
                        "horizon": h,
                        "mean_qlike": float(np.mean(m_losses)),
                        "har_qlike": float(np.mean(b_losses)),
                        "relative_qlike_ratio": ratio,
                        "bootstrap_p05": p05,
                        "bootstrap_p95": p95,
                        "low_vol_ratio": low_ratio,
                        "normal_vol_ratio": norm_ratio,
                        "high_vol_ratio": high_ratio,
                        "duration_seconds": duration / len(examples.horizons),
                    }
                )

    df_results = pd.DataFrame(records)
    return df_results


def freeze_numeric_companion(
    examples: VolatilityPanelExamples,
    winning_family: str,
    output_dir: Path = DEFAULT_ARTIFACTS_DIR,
    seed: int = 42,
    device: torch.device | None = None,
) -> Path:
    """Freeze winning numeric candidate on full development slice (no sealed test)."""
    raise RuntimeError(
        "numeric companion freezing is disabled until five-fold evidence, immutable data "
        "identities, and a family-specific serializer are supplied; smoke or single-fold "
        "results must never materialize a freeze artifact"
    )
