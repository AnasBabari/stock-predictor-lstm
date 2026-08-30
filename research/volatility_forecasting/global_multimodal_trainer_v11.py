"""End-to-end training, validation selection, and sealed evaluation for learned multimodal models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
)
from research.volatility_forecasting.multimodal_fusion_model_v2 import (
    MultimodalFusionModel,
)


@dataclass(frozen=True)
class FoldEvaluationResult:
    model_variant: str  # V1_BASELINE, V2_NUMERIC, V3_MULTIMODAL
    crps_mean: float
    return_mae: float
    qlike_mean: float
    coverage_80pct: float
    pinball_loss_10_90: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SealedCertificationResult:
    winning_model: str
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]
    val_metrics: dict[str, FoldEvaluationResult]
    sealed_test_metrics: dict[str, FoldEvaluationResult]
    news_ablation_delta_crps: float  # CRPS(V3) - CRPS(V2)
    trend_ablation_delta_crps: float  # CRPS(V2) - CRPS(V1)
    status: str  # CERTIFIED_PROMOTED or CERTIFIED_INFERIOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "winning_model": self.winning_model,
            "train_dates": self.train_dates,
            "val_dates": self.val_dates,
            "test_dates": self.test_dates,
            "val_metrics": {k: v.to_dict() for k, v in self.val_metrics.items()},
            "sealed_test_metrics": {k: v.to_dict() for k, v in self.sealed_test_metrics.items()},
            "news_ablation_delta_crps": self.news_ablation_delta_crps,
            "trend_ablation_delta_crps": self.trend_ablation_delta_crps,
            "status": self.status,
        }


class GlobalMultimodalTrainer:
    """Trains, tunes, and evaluates learned V2 (Numeric) and V3 (Multimodal) forecasting models."""

    @staticmethod
    def compute_crps_student_t(
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        df: float = 5.0,
    ) -> float:
        """Compute empirical Continuous Ranked Probability Score (CRPS)."""
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        # Standardized residual: z = (y - mu) / sigma
        z = (y_true - mu) / np.maximum(sigma, 1e-6)
        t_dist = stats.t(df=df)
        pdf = t_dist.pdf(z)
        cdf = t_dist.cdf(z)
        # Closed-form-like approximation
        crps = sigma * (
            z * (2.0 * cdf - 1.0)
            + 2.0 * pdf * ((df + z**2) / (df - 1.0))
            - (
                2.0
                * math.sqrt(df)
                * math.gamma(0.5 * (df - 1.0))
                / ((df - 1.0) * math.sqrt(math.pi) * math.gamma(0.5 * df))
            )
        )
        return float(np.mean(np.maximum(crps, 0.0)))

    @staticmethod
    def evaluate_model_partition(
        model: MultimodalFusionModel | None,
        x_num: np.ndarray,
        x_news: np.ndarray | None,
        y_returns: np.ndarray,
        y_rv: np.ndarray,
        is_baseline_v1: bool = False,
        historical_mean_drift: float = 0.0002,
        daily_vol: float = 0.0168,
        df: float = 5.0,
    ) -> FoldEvaluationResult:
        n_samples, n_horizons = y_returns.shape
        t_dist = stats.t(df=df)
        q80_factor = float(t_dist.ppf(0.90))

        if is_baseline_v1:
            # V1: constant drift and sqrt(h) variance
            pred_mu = np.array(
                [
                    [historical_mean_drift * (h + 1) for h in range(n_horizons)]
                    for _ in range(n_samples)
                ]
            )
            pred_sigma = np.array(
                [
                    [daily_vol * math.sqrt(h + 1) for h in range(n_horizons)]
                    for _ in range(n_samples)
                ]
            )
        else:
            assert model is not None
            model.eval()
            with torch.no_grad():
                t_num = torch.tensor(x_num, dtype=torch.float32)
                t_news = torch.tensor(x_news, dtype=torch.float32) if x_news is not None else None
                d_mu, d_log_vol = model(t_num, t_news)
                pred_mu = d_mu.numpy()
                pred_sigma = np.zeros_like(pred_mu)
                for h in range(n_horizons):
                    sigma_har = daily_vol * math.sqrt(h + 1)
                    pred_sigma[:, h] = sigma_har * np.exp(d_log_vol[:, h].numpy())

        # 1. CRPS across cumulative return distributions
        crps_val = GlobalMultimodalTrainer.compute_crps_student_t(
            y_true=y_returns, mu=pred_mu, sigma=pred_sigma, df=df
        )

        # 2. Return MAE
        mae = float(np.mean(np.abs(pred_mu - y_returns)))

        # 3. Volatility QLIKE
        pred_var = np.maximum(pred_sigma**2, 1e-8)
        true_var = np.maximum(y_rv, 1e-8)
        qlike = float(np.mean((true_var / pred_var) - np.log(true_var / pred_var) - 1.0))

        # 4. Empirical 80% Prediction Interval Coverage
        low_80 = pred_mu - q80_factor * pred_sigma
        high_80 = pred_mu + q80_factor * pred_sigma
        in_band = (y_returns >= low_80) & (y_returns <= high_80)
        coverage = float(np.mean(in_band))

        # 5. Pinball loss (10% and 90%)
        q10_factor = float(t_dist.ppf(0.10))
        q10 = pred_mu + q10_factor * pred_sigma
        q90 = pred_mu + q80_factor * pred_sigma
        loss_10 = np.maximum(0.10 * (y_returns - q10), -0.90 * (y_returns - q10))
        loss_90 = np.maximum(0.90 * (y_returns - q90), -0.10 * (y_returns - q90))
        pinball = float(np.mean(0.5 * (loss_10 + loss_90)))

        var_name = (
            "V1_BASELINE"
            if is_baseline_v1
            else ("V3_MULTIMODAL" if x_news is not None else "V2_NUMERIC")
        )
        return FoldEvaluationResult(
            model_variant=var_name,
            crps_mean=round(crps_val, 6),
            return_mae=round(mae, 6),
            qlike_mean=round(qlike, 6),
            coverage_80pct=round(coverage, 4),
            pinball_loss_10_90=round(pinball, 6),
        )

    @staticmethod
    def train_and_certify(
        dates: list[str],
        x_numeric: np.ndarray,
        x_news: np.ndarray,
        y_returns: np.ndarray,
        y_rv: np.ndarray,
        epochs: int = 15,
        lr: float = 0.005,
    ) -> tuple[MultimodalFusionModel, SealedCertificationResult]:
        split = ChronologicalPartitionManager.create_70_15_15_split(
            dates=dates, max_horizon_days=y_returns.shape[1], embargo_sessions=30
        )

        t_idx = split.train_indices
        v_idx = split.val_indices
        s_idx = split.test_indices

        # 1. Fit scalers on 70% train set ONLY
        num_mean = np.mean(x_numeric[t_idx], axis=0, keepdims=True)
        num_std = np.std(x_numeric[t_idx], axis=0, keepdims=True) + 1e-6
        x_num_scaled = (x_numeric - num_mean) / num_std

        news_mean = np.mean(x_news[t_idx], axis=0, keepdims=True)
        news_std = np.std(x_news[t_idx], axis=0, keepdims=True) + 1e-6
        x_news_scaled = (x_news - news_mean) / news_std

        n_num = x_numeric.shape[1]
        n_news = x_news.shape[1]
        n_h = y_returns.shape[1]

        # 2. Train V2 (Numeric Only)
        model_v2 = MultimodalFusionModel(
            numeric_dim=n_num, news_dim=n_news, horizons=tuple(range(1, n_h + 1))
        )
        opt_v2 = optim.Adam(model_v2.parameters(), lr=lr, weight_decay=1e-4)

        t_x_num = torch.tensor(x_num_scaled[t_idx], dtype=torch.float32)
        t_y_ret = torch.tensor(y_returns[t_idx], dtype=torch.float32)

        for _ in range(epochs):
            model_v2.train()
            opt_v2.zero_grad()
            pred_mu, _ = model_v2(t_x_num, None)
            loss = nn.functional.huber_loss(pred_mu, t_y_ret)
            loss.backward()
            opt_v2.step()

        # 3. Train V3 (Multimodal Numeric + News)
        model_v3 = MultimodalFusionModel(
            numeric_dim=n_num, news_dim=n_news, horizons=tuple(range(1, n_h + 1))
        )
        opt_v3 = optim.Adam(model_v3.parameters(), lr=lr, weight_decay=1e-4)

        t_x_news = torch.tensor(x_news_scaled[t_idx], dtype=torch.float32)

        for _ in range(epochs):
            model_v3.train()
            opt_v3.zero_grad()
            pred_mu, _ = model_v3(t_x_num, t_x_news)
            loss = nn.functional.huber_loss(pred_mu, t_y_ret)
            loss.backward()
            opt_v3.step()

        # 4. Evaluate on 15% Validation Partition for Model Selection
        val_v1 = GlobalMultimodalTrainer.evaluate_model_partition(
            model=None,
            x_num=x_num_scaled[v_idx],
            x_news=None,
            y_returns=y_returns[v_idx],
            y_rv=y_rv[v_idx],
            is_baseline_v1=True,
        )
        val_v2 = GlobalMultimodalTrainer.evaluate_model_partition(
            model=model_v2,
            x_num=x_num_scaled[v_idx],
            x_news=None,
            y_returns=y_returns[v_idx],
            y_rv=y_rv[v_idx],
            is_baseline_v1=False,
        )
        val_v3 = GlobalMultimodalTrainer.evaluate_model_partition(
            model=model_v3,
            x_num=x_num_scaled[v_idx],
            x_news=x_news_scaled[v_idx],
            y_returns=y_returns[v_idx],
            y_rv=y_rv[v_idx],
            is_baseline_v1=False,
        )

        val_metrics = {"V1_BASELINE": val_v1, "V2_NUMERIC": val_v2, "V3_MULTIMODAL": val_v3}
        winning_name = "V3_MULTIMODAL" if val_v3.crps_mean <= val_v2.crps_mean else "V2_NUMERIC"
        winning_model = model_v3 if winning_name == "V3_MULTIMODAL" else model_v2

        # 5. ONE-SHOT EVALUATION ON SEALED 15% TEST PARTITION
        test_v1 = GlobalMultimodalTrainer.evaluate_model_partition(
            model=None,
            x_num=x_num_scaled[s_idx],
            x_news=None,
            y_returns=y_returns[s_idx],
            y_rv=y_rv[s_idx],
            is_baseline_v1=True,
        )
        test_v2 = GlobalMultimodalTrainer.evaluate_model_partition(
            model=model_v2,
            x_num=x_num_scaled[s_idx],
            x_news=None,
            y_returns=y_returns[s_idx],
            y_rv=y_rv[s_idx],
            is_baseline_v1=False,
        )
        test_v3 = GlobalMultimodalTrainer.evaluate_model_partition(
            model=model_v3,
            x_num=x_num_scaled[s_idx],
            x_news=x_news_scaled[s_idx],
            y_returns=y_returns[s_idx],
            y_rv=y_rv[s_idx],
            is_baseline_v1=False,
        )
        test_metrics = {"V1_BASELINE": test_v1, "V2_NUMERIC": test_v2, "V3_MULTIMODAL": test_v3}

        delta_news = round(test_v3.crps_mean - test_v2.crps_mean, 6)
        delta_trend = round(test_v2.crps_mean - test_v1.crps_mean, 6)
        promoted = (test_v3.crps_mean <= test_v1.crps_mean) and (
            test_v2.crps_mean <= test_v1.crps_mean
        )

        cert_res = SealedCertificationResult(
            winning_model=winning_name,
            train_dates=split.train_dates,
            val_dates=split.val_dates,
            test_dates=split.test_dates,
            val_metrics=val_metrics,
            sealed_test_metrics=test_metrics,
            news_ablation_delta_crps=delta_news,
            trend_ablation_delta_crps=delta_trend,
            status="CERTIFIED_PROMOTED" if promoted else "CERTIFIED_INFERIOR",
        )

        return winning_model, cert_res
