"""Numeric V11.2 candidate training with an exact epoch-zero HAR prior.

This module is intentionally independent from the V11.1 multimodal trainer.
It trains one residual model per horizon and never accepts news features.
"""

from __future__ import annotations

import copy
import hashlib
import io
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch import nn

from .model import BaselineResidualLSTM, BaselineResidualTCNConfig
from .proper_scoring_v11 import student_t_crps
from .v11_2_evaluation import HorizonGate, evaluate_horizon_gates


@dataclass(frozen=True)
class V112Forecast:
    family: str
    horizon: int
    location: np.ndarray
    variance: np.ndarray
    crps: np.ndarray
    qlike: np.ndarray
    coverage_80: float

    @property
    def scale(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 1e-12) * (5.0 - 2.0) / 5.0)

    def metrics(self) -> dict[str, float]:
        return {
            "crps_mean": float(np.mean(self.crps)),
            "qlike_mean": float(np.mean(self.qlike)),
            "coverage_80": self.coverage_80,
        }


@dataclass(frozen=True)
class V112EpochEvidence:
    epoch: int
    validation_crps: float
    validation_qlike: float
    state_sha256: str


@dataclass(frozen=True)
class V112ResidualTrainingResult:
    model: BaselineResidualLSTM
    best_epoch: int
    epoch_evidence: tuple[V112EpochEvidence, ...]
    epoch_zero_crps: float
    epoch_zero_qlike: float
    epoch_zero_state_sha256: str
    best_state_sha256: str
    stop_reason: str


@dataclass(frozen=True)
class V112SelectionResult:
    horizon: int
    selected_family: str
    learned_promotion: bool
    gates: tuple[HorizonGate, ...]
    reason: str


def _state_digest(model: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _as_sequence(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3 or not len(values):
        raise ValueError("numeric features must have shape [rows, window, features]")
    if not np.isfinite(values).all():
        raise ValueError("numeric features must be finite")
    return values


def _metric_arrays(
    location: np.ndarray,
    variance: np.ndarray,
    returns: np.ndarray,
    realized_variance: np.ndarray,
    *,
    df: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    loc = np.asarray(location, dtype=np.float64).reshape(-1)
    var = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-12)
    y = np.asarray(returns, dtype=np.float64).reshape(-1)
    rv = np.maximum(np.asarray(realized_variance, dtype=np.float64).reshape(-1), 1e-12)
    if not (len(loc) == len(var) == len(y) == len(rv)):
        raise ValueError("forecast and target arrays must have equal length")
    scale = np.sqrt(var * (df - 2.0) / df)
    crps = np.asarray(student_t_crps(y, loc, scale, df=df), dtype=np.float64)
    qlike = rv / var - np.log(rv / var) - 1.0
    return crps, qlike


def make_forecast(
    family: str,
    horizon: int,
    location: np.ndarray,
    variance: np.ndarray,
    returns: np.ndarray,
    realized_variance: np.ndarray,
) -> V112Forecast:
    crps, qlike = _metric_arrays(location, variance, returns, realized_variance)
    location_values = np.asarray(location, dtype=np.float64).reshape(-1)
    variance_values = np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-12)
    returns_values = np.asarray(returns, dtype=np.float64).reshape(-1)
    critical_value = float(stats.t.ppf(0.90, df=5.0))
    scale = np.sqrt(variance_values * (5.0 - 2.0) / 5.0)
    lower = location_values - critical_value * scale
    upper = location_values + critical_value * scale
    coverage_80 = float(np.mean((returns_values >= lower) & (returns_values <= upper)))
    return V112Forecast(
        family=family,
        horizon=horizon,
        location=location_values,
        variance=variance_values,
        crps=crps,
        qlike=qlike,
        coverage_80=coverage_80,
    )


def _loss(
    prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    baseline_variance: torch.Tensor,
    realized_variance: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    forecast_variance, location, _direction, _residual = prediction
    safe_var = torch.clamp(forecast_variance, min=1e-8, max=1e2)
    safe_rv = torch.clamp(realized_variance, min=1e-8, max=1e2)
    qlike = torch.mean(safe_rv / safe_var - torch.log(safe_rv / safe_var) - 1.0)
    scale = torch.sqrt(torch.clamp(baseline_variance, min=1e-8))
    location_loss = torch.mean(nn.functional.smooth_l1_loss(location / scale, returns / scale))
    return qlike + 0.25 * location_loss


def train_epoch_zero_residual_model(
    *,
    x_train: np.ndarray,
    base_variance_train: np.ndarray,
    returns_train: np.ndarray,
    rv_train: np.ndarray,
    x_validation: np.ndarray,
    base_variance_validation: np.ndarray,
    returns_validation: np.ndarray,
    rv_validation: np.ndarray,
    max_epochs: int = 15,
    patience: int = 4,
    learning_rate: float = 0.003,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> V112ResidualTrainingResult:
    """Train one horizon while evaluating and preserving exact epoch zero."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    torch.manual_seed(seed)
    train_x = _as_sequence(x_train)
    validation_x = _as_sequence(x_validation)
    feature_count = train_x.shape[-1]
    window_size = train_x.shape[1]
    config = BaselineResidualTCNConfig(
        feature_count=feature_count,
        horizon_count=1,
        encoder_family="lstm",
        window_size=max(window_size, 2),
        channels=32,
        lstm_hidden=32,
        lstm_layers=1,
        dropout=0.15,
        patch_length=2,
        patch_stride=1,
    )
    model = BaselineResidualLSTM(config)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    model.to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    t_x = torch.tensor(train_x, dtype=torch.float32, device=selected_device)
    t_b = torch.tensor(
        np.asarray(base_variance_train, dtype=np.float32).reshape(-1, 1), device=selected_device
    )
    t_y = torch.tensor(
        np.asarray(returns_train, dtype=np.float32).reshape(-1, 1), device=selected_device
    )
    t_rv = torch.tensor(
        np.asarray(rv_train, dtype=np.float32).reshape(-1, 1), device=selected_device
    )
    v_x = torch.tensor(validation_x, dtype=torch.float32, device=selected_device)
    v_b_np = np.asarray(base_variance_validation, dtype=np.float64).reshape(-1, 1)
    v_y_np = np.asarray(returns_validation, dtype=np.float64).reshape(-1, 1)
    v_rv_np = np.asarray(rv_validation, dtype=np.float64).reshape(-1, 1)
    epoch_evidence: list[V112EpochEvidence] = []

    def evaluate(epoch: int) -> tuple[float, float, str]:
        model.eval()
        with torch.no_grad():
            variance, location, _direction, _residual = model(
                v_x, torch.tensor(v_b_np.astype(np.float32), device=selected_device)
            )
        crps, qlike = _metric_arrays(
            location.detach().cpu().numpy(),
            variance.detach().cpu().numpy(),
            v_y_np,
            v_rv_np,
        )
        return float(np.mean(crps)), float(np.mean(qlike)), _state_digest(model)

    # Epoch zero is evaluated before any optimizer update and is the exact HAR prior.
    epoch_zero_crps, epoch_zero_qlike, epoch_zero_digest = evaluate(0)
    epoch_evidence.append(
        V112EpochEvidence(0, epoch_zero_crps, epoch_zero_qlike, epoch_zero_digest)
    )
    best_crps = epoch_zero_crps
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stagnant = 0
    stop_reason = "max_epochs_reached"

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(t_x, t_b)
        loss = _loss(prediction, t_b, t_rv, t_y)
        if not torch.isfinite(loss):
            stop_reason = "nonfinite_training_loss"
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        val_crps, val_qlike, state_digest = evaluate(epoch)
        epoch_evidence.append(V112EpochEvidence(epoch, val_crps, val_qlike, state_digest))
        if val_crps < best_crps:
            best_crps = val_crps
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                stop_reason = "early_stopping"
                break

    model.load_state_dict(best_state)
    return V112ResidualTrainingResult(
        model=model,
        best_epoch=best_epoch,
        epoch_evidence=tuple(epoch_evidence),
        epoch_zero_crps=epoch_zero_crps,
        epoch_zero_qlike=epoch_zero_qlike,
        epoch_zero_state_sha256=epoch_zero_digest,
        best_state_sha256=_state_digest(model),
        stop_reason=stop_reason,
    )


def fit_ridge_location(
    x_train: np.ndarray,
    returns_train: np.ndarray,
    x_eval: np.ndarray,
    base_variance_eval: np.ndarray,
    returns_eval: np.ndarray,
    rv_eval: np.ndarray,
    *,
    horizon: int,
    alpha: float = 1.0,
) -> V112Forecast:
    model = Ridge(alpha=alpha)
    train_values = _as_sequence(x_train).reshape(len(x_train), -1)
    eval_values = _as_sequence(x_eval).reshape(len(x_eval), -1)
    model.fit(train_values, np.asarray(returns_train, dtype=np.float64).reshape(-1))
    location = model.predict(eval_values)
    return make_forecast(
        "RIDGE_LOCATION_HAR_SCALE",
        horizon,
        location,
        base_variance_eval,
        returns_eval,
        rv_eval,
    )


def fit_histgb_location(
    x_train: np.ndarray,
    returns_train: np.ndarray,
    x_eval: np.ndarray,
    base_variance_eval: np.ndarray,
    returns_eval: np.ndarray,
    rv_eval: np.ndarray,
    *,
    horizon: int,
    max_iter: int = 150,
) -> V112Forecast:
    model = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
    train_values = _as_sequence(x_train).reshape(len(x_train), -1)
    eval_values = _as_sequence(x_eval).reshape(len(x_eval), -1)
    model.fit(train_values, np.asarray(returns_train, dtype=np.float64).reshape(-1))
    location = model.predict(eval_values)
    return make_forecast(
        "HISTGB_LOCATION_HAR_SCALE",
        horizon,
        location,
        base_variance_eval,
        returns_eval,
        rv_eval,
    )


def evaluate_residual_model(
    result: V112ResidualTrainingResult,
    *,
    x_eval: np.ndarray,
    base_variance_eval: np.ndarray,
    returns_eval: np.ndarray,
    rv_eval: np.ndarray,
    horizon: int,
) -> V112Forecast:
    x_values = torch.tensor(_as_sequence(x_eval), dtype=torch.float32)
    base = np.asarray(base_variance_eval, dtype=np.float32).reshape(-1, 1)
    with torch.no_grad():
        model_device = next(result.model.parameters()).device
        variance, location, _direction, _residual = result.model(
            x_values.to(model_device),
            torch.tensor(base, dtype=torch.float32, device=model_device),
        )
    return make_forecast(
        "M1_NUMERIC_RESIDUAL",
        horizon,
        location.detach().cpu().numpy().reshape(-1),
        variance.detach().cpu().numpy().reshape(-1),
        returns_eval,
        rv_eval,
    )


def select_per_horizon_challenger(
    *,
    horizon: int,
    dates: list[str] | tuple[str, ...],
    har: V112Forecast,
    candidates: dict[str, V112Forecast],
    ranking_scores: dict[str, float] | None = None,
    block_sessions: int = 20,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> V112SelectionResult:
    """Select one learned route for a horizon or fail safely to HAR.

    This compatibility wrapper evaluates a single horizon.  The production
    V11.2 runner uses :func:`select_per_horizon_challengers` so Holm correction
    is applied jointly across all four horizon decisions.
    """
    return select_per_horizon_challengers(
        dates=dates,
        horizons=[horizon],
        har_by_horizon={horizon: har},
        candidates_by_horizon={horizon: candidates},
        ranking_scores_by_horizon={horizon: ranking_scores} if ranking_scores is not None else {},
        block_sessions=block_sessions,
        n_replicates=n_replicates,
        seed=seed,
    )[horizon]


def select_per_horizon_challengers(
    *,
    dates: Iterable[str],
    horizons: Iterable[int],
    har_by_horizon: dict[int, V112Forecast],
    candidates_by_horizon: dict[int, dict[str, V112Forecast]],
    ranking_scores_by_horizon: dict[int, dict[str, float]],
    block_sessions: int = 20,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> dict[int, V112SelectionResult]:
    """Rank candidates per horizon and apply one family-wise gate.

    Ranking is allowed to be horizon-specific, but the uncertainty tests are
    evaluated in one call so Holm step-down correction covers the complete
    frozen horizon family.  This prevents four independent one-test decisions
    from being mislabeled as a four-horizon family-wise gate.
    """
    horizon_list = [int(value) for value in horizons]
    if not horizon_list or len(set(horizon_list)) != len(horizon_list):
        raise ValueError("horizons must be a non-empty unique sequence")
    date_values = list(dates)
    selected: dict[int, V112Forecast] = {}
    selected_families: dict[int, str] = {}
    results: dict[int, V112SelectionResult] = {}
    learned_families = {
        "RIDGE_LOCATION_HAR_SCALE",
        "HISTGB_LOCATION_HAR_SCALE",
        "M1_NUMERIC_RESIDUAL",
    }
    for horizon in horizon_list:
        if horizon not in har_by_horizon or horizon not in candidates_by_horizon:
            raise ValueError(f"missing HAR or candidate forecasts for horizon {horizon}")
        candidates = candidates_by_horizon[horizon]
        scores = ranking_scores_by_horizon.get(horizon)
        if not isinstance(scores, dict):
            raise ValueError("candidate ranking must come from training-only evidence")
        learned = [
            candidate for family, candidate in candidates.items() if family in learned_families
        ]
        if not learned:
            results[horizon] = V112SelectionResult(
                horizon, "M0_HAR_BASELINE", False, (), "no learned candidates"
            )
            continue
        missing_scores = [
            candidate.family for candidate in learned if candidate.family not in scores
        ]
        if missing_scores:
            raise ValueError(f"missing training-only ranking scores: {missing_scores}")
        challenger = min(learned, key=lambda candidate: scores[candidate.family])
        selected[horizon] = challenger
        selected_families[horizon] = challenger.family

    if selected:
        gates = evaluate_horizon_gates(
            dates=date_values,
            horizons=sorted(selected),
            candidate="per_horizon_selected_challenger",
            comparator="M0_HAR_BASELINE",
            candidate_losses_by_horizon={h: selected[h].crps for h in selected},
            comparator_losses_by_horizon={h: har_by_horizon[h].crps for h in selected},
            candidate_crps_by_horizon={h: float(np.mean(selected[h].crps)) for h in selected},
            comparator_crps_by_horizon={
                h: float(np.mean(har_by_horizon[h].crps)) for h in selected
            },
            qlike_candidate_by_horizon={h: float(np.mean(selected[h].qlike)) for h in selected},
            qlike_comparator_by_horizon={
                h: float(np.mean(har_by_horizon[h].qlike)) for h in selected
            },
            coverage_candidate_by_horizon={h: selected[h].coverage_80 for h in selected},
            coverage_comparator_by_horizon={h: har_by_horizon[h].coverage_80 for h in selected},
            block_sessions=block_sessions,
            n_replicates=n_replicates,
            seed=seed,
        )
        for gate in gates:
            family = selected_families[gate.horizon]
            # Preserve the actual family in evidence while retaining the
            # globally corrected p-value produced by evaluate_horizon_gates.
            corrected_gate = HorizonGate(
                horizon=gate.horizon,
                candidate=family,
                comparator=gate.comparator,
                mean_crps_candidate=gate.mean_crps_candidate,
                mean_crps_comparator=gate.mean_crps_comparator,
                interval=gate.interval,
                holm_p_value=gate.holm_p_value,
                passed=gate.passed,
                reason=gate.reason,
                coverage_candidate_80=gate.coverage_candidate_80,
                coverage_comparator_80=gate.coverage_comparator_80,
            )
            results[gate.horizon] = V112SelectionResult(
                gate.horizon,
                family if gate.passed else "M0_HAR_BASELINE",
                gate.passed,
                (corrected_gate,),
                "learned challenger passed the family-wise horizon gates"
                if gate.passed
                else "learned challenger failed; HAR retained",
            )

    return {horizon: results[horizon] for horizon in horizon_list}
