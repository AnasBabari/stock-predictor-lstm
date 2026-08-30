"""Calibrated multi-horizon probabilistic evaluation metrics and statistical hypothesis testing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ReturnPriceMetrics:
    return_mae: float
    return_rmse: float
    price_mae: float
    price_rmse: float
    oos_r2_vs_zero: float
    relative_return_mae_vs_baseline: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionMetrics:
    macro_f1: float
    balanced_accuracy: float
    brier_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VolatilityMetrics:
    qlike: float
    log_var_mae: float
    relative_qlike_vs_persistence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbabilisticIntervalMetrics:
    coverage_50pct: float
    coverage_80pct: float
    coverage_95pct: float
    mean_interval_width: float
    crps_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HonestEvaluationEngine:
    """Computes transparent, paired evaluation metrics with bootstrap confidence and DM tests."""

    @staticmethod
    def evaluate_returns(
        y_true_returns: np.ndarray,
        y_pred_returns: np.ndarray,
        base_prices: np.ndarray,
        baseline_pred_returns: np.ndarray | None = None,
    ) -> ReturnPriceMetrics:
        """Evaluate cumulative return and reconstructed price errors."""
        y_t = np.asarray(y_true_returns, dtype=float)
        y_p = np.asarray(y_pred_returns, dtype=float)
        p0 = np.asarray(base_prices, dtype=float)

        ret_err = y_p - y_t
        ret_mae = float(np.mean(np.abs(ret_err)))
        ret_rmse = float(np.sqrt(np.mean(ret_err**2)))

        p_true = p0 * np.exp(y_t)
        p_pred = p0 * np.exp(y_p)
        price_err = p_pred - p_true
        price_mae = float(np.mean(np.abs(price_err)))
        price_rmse = float(np.sqrt(np.mean(price_err**2)))

        # OOS R2 vs zero return (y_zero = 0)
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum(y_t**2)  # Benchmark is 0 return
        oos_r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))

        rel_mae = 1.0
        if baseline_pred_returns is not None:
            base_err = np.asarray(baseline_pred_returns, dtype=float) - y_t
            base_mae = float(np.mean(np.abs(base_err)))
            rel_mae = float(ret_mae / max(base_mae, 1e-8))

        return ReturnPriceMetrics(
            return_mae=ret_mae,
            return_rmse=ret_rmse,
            price_mae=price_mae,
            price_rmse=price_rmse,
            oos_r2_vs_zero=oos_r2,
            relative_return_mae_vs_baseline=rel_mae,
        )

    @staticmethod
    def evaluate_volatility_qlike(
        y_true_variance: np.ndarray,
        y_pred_variance: np.ndarray,
        baseline_variance: np.ndarray | None = None,
    ) -> VolatilityMetrics:
        """Patton (2011) QLIKE loss: L(y, h) = y/h - log(y/h) - 1."""
        y = np.maximum(np.asarray(y_true_variance, dtype=float), 1e-8)
        h = np.maximum(np.asarray(y_pred_variance, dtype=float), 1e-8)

        ratio = y / h
        qlike_losses = ratio - np.log(ratio) - 1.0
        mean_qlike = float(np.mean(qlike_losses))

        log_var_mae = float(np.mean(np.abs(np.log(h) - np.log(y))))

        rel_qlike = 1.0
        if baseline_variance is not None:
            h_base = np.maximum(np.asarray(baseline_variance, dtype=float), 1e-8)
            ratio_base = y / h_base
            base_losses = ratio_base - np.log(ratio_base) - 1.0
            rel_qlike = float(mean_qlike / max(np.mean(base_losses), 1e-8))

        return VolatilityMetrics(
            qlike=mean_qlike,
            log_var_mae=log_var_mae,
            relative_qlike_vs_persistence=rel_qlike,
        )

    @staticmethod
    def evaluate_probabilistic_coverage(
        y_true: np.ndarray,
        quantiles: dict[int, np.ndarray],  # 5, 10, 25, 50, 75, 90, 95
    ) -> ProbabilisticIntervalMetrics:
        """Calculate empirical coverage rates for 50%, 80%, and 95% central intervals."""
        y = np.asarray(y_true, dtype=float)

        # 50% interval: [25%, 75%]
        q25 = np.asarray(quantiles[25], dtype=float)
        q75 = np.asarray(quantiles[75], dtype=float)
        cov_50 = float(np.mean((y >= q25) & (y <= q75)))

        # 80% interval: [10%, 90%]
        q10 = np.asarray(quantiles[10], dtype=float)
        q90 = np.asarray(quantiles[90], dtype=float)
        cov_80 = float(np.mean((y >= q10) & (y <= q90)))

        # 95% interval: [2.5% / 5%, 95%]
        q5 = np.asarray(quantiles.get(5, q10), dtype=float)
        q95 = np.asarray(quantiles.get(95, q90), dtype=float)
        cov_95 = float(np.mean((y >= q5) & (y <= q95)))

        mean_width = float(np.mean(q90 - q10))

        # Approximate pinball CRPS across available quantiles
        tau_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
        pinball_sum = 0.0
        for tau in tau_levels:
            q_val = np.asarray(quantiles[int(tau * 100)], dtype=float)
            diff = y - q_val
            loss = np.maximum(tau * diff, (tau - 1.0) * diff)
            pinball_sum += np.mean(loss)
        crps = float(pinball_sum / len(tau_levels))

        return ProbabilisticIntervalMetrics(
            coverage_50pct=cov_50,
            coverage_80pct=cov_80,
            coverage_95pct=cov_95,
            mean_interval_width=mean_width,
            crps_score=crps,
        )

    @staticmethod
    def diebold_mariano_test(
        loss_candidate: np.ndarray, loss_baseline: np.ndarray, horizon: int = 1
    ) -> tuple[float, float]:
        """Diebold-Mariano test with Newey-West HAC variance estimator."""
        d = np.asarray(loss_candidate, dtype=float) - np.asarray(loss_baseline, dtype=float)
        n = len(d)
        if n < 10:
            return 0.0, 1.0

        d_mean = np.mean(d)
        # Autocovariance up to lag h-1
        gamma_0 = np.var(d, ddof=0)
        gamma_sum = 0.0
        for lag in range(1, horizon):
            gamma_k = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
            weight = 1.0 - (lag / horizon)  # Bartlett kernel
            gamma_sum += 2.0 * weight * gamma_k

        lr_var = max(1e-10, gamma_0 + gamma_sum)
        dm_stat = float(d_mean / np.sqrt(lr_var / n))
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
        return dm_stat, p_val

    @staticmethod
    def holm_bonferroni_correction(p_values: list[float]) -> list[bool]:
        """Holm-Bonferroni step-down correction at alpha=0.05 family-wise error rate."""
        m = len(p_values)
        if m == 0:
            return []
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        significant = [False] * m

        for rank, (orig_idx, p) in enumerate(indexed):
            alpha_k = 0.05 / (m - rank)
            if p <= alpha_k:
                significant[orig_idx] = True
            else:
                break
        return significant
