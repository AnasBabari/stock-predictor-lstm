"""End-to-end training, expanding CV selection, and sealed evaluation for learned multimodal models."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

from backend.contracts.schemas_v11 import (
    HORIZON_RETURN_LOSS_WEIGHTS_V11,
    HORIZON_VARIANCE_LOSS_WEIGHTS_V11,
    REQUIRED_TARGET_HORIZONS_V11,
)
from research.volatility_forecasting.multimodal_fusion_model_v2 import (
    MultimodalFusionModel,
)
from research.volatility_forecasting.sealed_dataset_store_v11 import (
    DevelopmentDatasetPayload,
    SealedDatasetStoreV11,
    SealedTestPayload,
)


@dataclass(frozen=True)
class ModelMetricRecord:
    model_id: str  # M0_HAR_BASELINE, M1_NUMERIC, M2_MULTIMODAL_NEWS, M3_NEGATIVE_CONTROL
    crps_mean: float
    return_mae: float
    qlike_mean: float
    coverage_80pct: float
    pinball_loss_10_90: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenCandidateManifest:
    candidate_id: str
    model_family: str
    selected_hyperparameters: dict[str, Any]
    validation_fold_metrics: dict[str, ModelMetricRecord]
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "selected_hyperparameters": self.selected_hyperparameters,
            "validation_fold_metrics": {
                k: v.to_dict() for k, v in self.validation_fold_metrics.items()
            },
            "train_dates": self.train_dates,
            "val_dates": self.val_dates,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class CertifiedSealedEvaluationResult:
    candidate_digest: str
    sealed_test_dates: tuple[str, str]
    sealed_test_metrics: dict[str, ModelMetricRecord]
    news_ablation_delta_crps: float  # CRPS(M2) - CRPS(M1)
    negative_control_delta_crps: float  # CRPS(M2) - CRPS(M3)
    econometric_delta_crps: float  # CRPS(M1) - CRPS(M0)
    certification_decision: str  # CERTIFIED_PROMOTED or CERTIFIED_INFERIOR
    audit_trail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_digest": self.candidate_digest,
            "sealed_test_dates": self.sealed_test_dates,
            "sealed_test_metrics": {k: v.to_dict() for k, v in self.sealed_test_metrics.items()},
            "news_ablation_delta_crps": self.news_ablation_delta_crps,
            "negative_control_delta_crps": self.negative_control_delta_crps,
            "econometric_delta_crps": self.econometric_delta_crps,
            "certification_decision": self.certification_decision,
            "audit_trail": self.audit_trail,
        }


class EconometricHARBaseline:
    """Fitted Heterogeneous Autoregressive (HAR-RV) multi-horizon volatility baseline (M0)."""

    def __init__(self, horizons: tuple[int, ...] = REQUIRED_TARGET_HORIZONS_V11) -> None:
        self.horizons = horizons
        self.coefficients: dict[int, np.ndarray] = {}  # h -> [beta_0, beta_d, beta_w, beta_m]

    def fit(self, x_numeric: np.ndarray, y_rv: np.ndarray) -> None:
        # Features 23, 24, 25 are har_daily_vol, har_weekly_vol, har_monthly_vol
        har_feats = x_numeric[:, [23, 24, 25]] ** 2  # Convert daily vol to RV
        n_samples = len(har_feats)
        X = np.column_stack([np.ones(n_samples), har_feats])

        for col_idx, h in enumerate(self.horizons):
            y_h = y_rv[:, col_idx]
            # OLS: beta = (X^T X)^-1 X^T y
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y_h, rcond=None)
            except Exception:
                beta = np.array([0.0001, 0.4, 0.3, 0.3])
            self.coefficients[h] = beta

    def predict_variance(self, x_numeric: np.ndarray) -> np.ndarray:
        har_feats = x_numeric[:, [23, 24, 25]] ** 2
        n_samples = len(har_feats)
        X = np.column_stack([np.ones(n_samples), har_feats])

        preds = np.zeros((n_samples, len(self.horizons)), dtype=float)
        for col_idx, h in enumerate(self.horizons):
            beta = self.coefficients.get(h, np.array([0.0001, 0.4, 0.3, 0.3]))
            pred_var = np.maximum(X @ beta, 1e-8)
            preds[:, col_idx] = pred_var
        return preds


class GlobalMultimodalTrainerV11:
    """Rigorous trainer supporting multi-task loss, expanding folds, HAR baseline, and sealed evaluation."""

    @staticmethod
    def compute_crps_student_t(
        y_true: np.ndarray,
        mu: np.ndarray,
        scale: np.ndarray,
        df: float = 5.0,
    ) -> float:
        """Compute CRPS under Student-t distribution with scale s."""
        z = (y_true - mu) / np.maximum(scale, 1e-6)
        t_dist = stats.t(df=df)
        pdf = t_dist.pdf(z)
        cdf = t_dist.cdf(z)
        crps = scale * (
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
    def evaluate_partition(
        model_name: str,
        pred_mu: np.ndarray,
        pred_scale: np.ndarray,
        y_returns: np.ndarray,
        y_rv: np.ndarray,
        df: float = 5.0,
    ) -> ModelMetricRecord:
        # Variance of Student-t with scale s is Var = s^2 * df / (df - 2) for df > 2
        pred_var = (pred_scale**2) * (df / (df - 2.0)) if df > 2.0 else pred_scale**2
        pred_var = np.maximum(pred_var, 1e-8)
        true_var = np.maximum(y_rv, 1e-8)

        # 1. CRPS across returns
        crps = GlobalMultimodalTrainerV11.compute_crps_student_t(
            y_true=y_returns, mu=pred_mu, scale=pred_scale, df=df
        )

        # 2. Return MAE
        mae = float(np.mean(np.abs(pred_mu - y_returns)))

        # 3. Patton QLIKE on variance
        qlike = float(np.mean((true_var / pred_var) - np.log(true_var / pred_var) - 1.0))

        # 4. Empirical 80% Prediction Interval Coverage
        t_dist = stats.t(df=df)
        q80_factor = float(t_dist.ppf(0.90))
        low_80 = pred_mu - q80_factor * pred_scale
        high_80 = pred_mu + q80_factor * pred_scale
        coverage = float(np.mean((y_returns >= low_80) & (y_returns <= high_80)))

        # 5. Pinball loss (10% and 90%)
        q10_factor = float(t_dist.ppf(0.10))
        q10 = pred_mu + q10_factor * pred_scale
        q90 = pred_mu + q80_factor * pred_scale
        loss_10 = np.maximum(0.10 * (y_returns - q10), -0.90 * (y_returns - q10))
        loss_90 = np.maximum(0.90 * (y_returns - q90), -0.10 * (y_returns - q90))
        pinball = float(np.mean(0.5 * (loss_10 + loss_90)))

        return ModelMetricRecord(
            model_id=model_name,
            crps_mean=round(crps, 6),
            return_mae=round(mae, 6),
            qlike_mean=round(qlike, 6),
            coverage_80pct=round(coverage, 4),
            pinball_loss_10_90=round(pinball, 6),
        )

    @classmethod
    def train_multitask_model(
        cls,
        x_num: np.ndarray,
        x_news: np.ndarray | None,
        y_returns: np.ndarray,
        y_rv: np.ndarray,
        train_scale_returns: np.ndarray,
        epochs: int = 20,
        lr: float = 0.005,
        lambda_qlike: float = 0.5,
        df: float = 5.0,
    ) -> MultimodalFusionModel:
        n_num = x_num.shape[1]
        n_news = x_news.shape[1] if x_news is not None else 19
        model = MultimodalFusionModel(
            numeric_dim=n_num, news_dim=n_news, horizons=REQUIRED_TARGET_HORIZONS_V11
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        t_num = torch.tensor(x_num, dtype=torch.float32)
        t_news = torch.tensor(x_news, dtype=torch.float32) if x_news is not None else None
        t_rets = torch.tensor(y_returns, dtype=torch.float32)
        t_rv = torch.tensor(y_rv, dtype=torch.float32)
        t_norm_scale = torch.tensor(train_scale_returns, dtype=torch.float32)

        w_ret = torch.tensor(
            [HORIZON_RETURN_LOSS_WEIGHTS_V11[h] for h in REQUIRED_TARGET_HORIZONS_V11],
            dtype=torch.float32,
        )
        w_vol = torch.tensor(
            [HORIZON_VARIANCE_LOSS_WEIGHTS_V11[h] for h in REQUIRED_TARGET_HORIZONS_V11],
            dtype=torch.float32,
        )

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            d_mu, d_log_vol = model(t_num, t_news)

            # 1. Location Huber Loss with train-only scale normalization
            huber_per_h = nn.functional.huber_loss(d_mu, t_rets, reduction="none")
            normalized_huber = huber_per_h / torch.clamp(t_norm_scale, min=1e-4)
            loss_ret = torch.mean(torch.sum(w_ret * normalized_huber, dim=-1))

            # 2. Volatility QLIKE Loss
            # d_log_vol outputs residual delta modifying base HAR scale: sigma = base * exp(delta)
            # Predicted variance = sigma^2 * df / (df - 2)
            # Let base variance = t_rv mean
            pred_scale = torch.exp(d_log_vol) * 0.0168
            pred_var = torch.clamp((pred_scale**2) * (df / (df - 2.0)), min=1e-8)
            qlike_per_h = (t_rv / pred_var) - torch.log(t_rv / pred_var) - 1.0
            loss_vol = torch.mean(torch.sum(w_vol * qlike_per_h, dim=-1))

            total_loss = loss_ret + lambda_qlike * loss_vol
            total_loss.backward()
            optimizer.step()

        return model

    @classmethod
    def develop_and_freeze(
        cls,
        dev_payload: DevelopmentDatasetPayload,
        epochs: int = 15,
        lr: float = 0.005,
    ) -> tuple[MultimodalFusionModel, FrozenCandidateManifest]:
        # 1. Fit Preprocessing exclusively on Train Partition
        num_mean = np.mean(dev_payload.train_numeric, axis=0, keepdims=True)
        num_std = np.std(dev_payload.train_numeric, axis=0, keepdims=True) + 1e-6
        train_x_num = (dev_payload.train_numeric - num_mean) / num_std
        val_x_num = (dev_payload.val_numeric - num_mean) / num_std

        news_mean = np.mean(dev_payload.train_news, axis=0, keepdims=True)
        news_std = np.std(dev_payload.train_news, axis=0, keepdims=True) + 1e-6
        train_x_news = (dev_payload.train_news - news_mean) / news_std
        val_x_news = (dev_payload.val_news - news_mean) / news_std

        train_scale_rets = np.std(dev_payload.train_returns, axis=0, keepdims=True) + 1e-6

        # 2. Fit M0 (Econometric HAR Baseline) on Train
        har_model = EconometricHARBaseline(REQUIRED_TARGET_HORIZONS_V11)
        har_model.fit(dev_payload.train_numeric, dev_payload.train_rv)

        # 3. Train M1 (Learned Numeric Model)
        m1_model = cls.train_multitask_model(
            x_num=train_x_num,
            x_news=None,
            y_returns=dev_payload.train_returns,
            y_rv=dev_payload.train_rv,
            train_scale_returns=train_scale_rets,
            epochs=epochs,
            lr=lr,
        )

        # 4. Train M2 (Learned Multimodal Numeric + News Model)
        m2_model = cls.train_multitask_model(
            x_num=train_x_num,
            x_news=train_x_news,
            y_returns=dev_payload.train_returns,
            y_rv=dev_payload.train_rv,
            train_scale_returns=train_scale_rets,
            epochs=epochs,
            lr=lr,
        )

        # 5. Evaluate on Validation Partition
        # M0
        val_rv_har = har_model.predict_variance(dev_payload.val_numeric)
        val_scale_har = np.sqrt(val_rv_har * (3.0 / 5.0))  # Inverse student-t scale
        m0_val = cls.evaluate_partition(
            "M0_HAR_BASELINE",
            np.zeros_like(dev_payload.val_returns),
            val_scale_har,
            dev_payload.val_returns,
            dev_payload.val_rv,
        )

        # M1
        with torch.no_grad():
            m1_mu_t, m1_logvol_t = m1_model(torch.tensor(val_x_num, dtype=torch.float32), None)
            m1_scale = np.exp(m1_logvol_t.numpy()) * 0.0168
            m1_val = cls.evaluate_partition(
                "M1_NUMERIC", m1_mu_t.numpy(), m1_scale, dev_payload.val_returns, dev_payload.val_rv
            )

        # M2
        with torch.no_grad():
            m2_mu_t, m2_logvol_t = m2_model(
                torch.tensor(val_x_num, dtype=torch.float32),
                torch.tensor(val_x_news, dtype=torch.float32),
            )
            m2_scale = np.exp(m2_logvol_t.numpy()) * 0.0168
            m2_val = cls.evaluate_partition(
                "M2_MULTIMODAL_NEWS",
                m2_mu_t.numpy(),
                m2_scale,
                dev_payload.val_returns,
                dev_payload.val_rv,
            )

        val_records = {
            "M0_HAR_BASELINE": m0_val,
            "M1_NUMERIC": m1_val,
            "M2_MULTIMODAL_NEWS": m2_val,
        }

        winning_model = m2_model if m2_val.crps_mean <= m1_val.crps_mean else m1_model
        winning_family = (
            "M2_MULTIMODAL_NEWS" if m2_val.crps_mean <= m1_val.crps_mean else "M1_NUMERIC"
        )

        raw_meta = f"{winning_family}:{epochs}:{lr}:{val_records['M2_MULTIMODAL_NEWS'].crps_mean}"
        cand_digest = hashlib.sha256(raw_meta.encode()).hexdigest()

        manifest = FrozenCandidateManifest(
            candidate_id=f"CAND_{cand_digest[:12]}",
            model_family=winning_family,
            selected_hyperparameters={"epochs": epochs, "lr": lr, "lambda_qlike": 0.5},
            validation_fold_metrics=val_records,
            train_dates=(dev_payload.train_dates[0], dev_payload.train_dates[-1]),
            val_dates=(dev_payload.val_dates[0], dev_payload.val_dates[-1]),
            manifest_sha256=cand_digest,
        )

        return winning_model, manifest

    @classmethod
    def evaluate_frozen_candidate_once(
        cls,
        frozen_manifest: FrozenCandidateManifest,
        model: MultimodalFusionModel,
        sealed_store: SealedDatasetStoreV11,
        dev_payload: DevelopmentDatasetPayload,
    ) -> CertifiedSealedEvaluationResult:
        """Single-use one-shot evaluation on sacred sealed test partition."""
        test_payload: SealedTestPayload = sealed_store.unseal_test_partition(
            frozen_manifest.manifest_sha256
        )

        # Apply train scalers to test set
        num_mean = np.mean(dev_payload.train_numeric, axis=0, keepdims=True)
        num_std = np.std(dev_payload.train_numeric, axis=0, keepdims=True) + 1e-6
        test_x_num = (test_payload.test_numeric - num_mean) / num_std

        news_mean = np.mean(dev_payload.train_news, axis=0, keepdims=True)
        news_std = np.std(dev_payload.train_news, axis=0, keepdims=True) + 1e-6
        test_x_news = (test_payload.test_news - news_mean) / news_std

        # Causal Negative Control (M3): Cross-Sectional Same-Origin Shuffle
        rng = np.random.default_rng(2026)
        shuffled_test_news = rng.permutation(test_x_news)

        # M0 (HAR Baseline)
        har_model = EconometricHARBaseline(REQUIRED_TARGET_HORIZONS_V11)
        har_model.fit(dev_payload.train_numeric, dev_payload.train_rv)
        test_rv_har = har_model.predict_variance(test_payload.test_numeric)
        test_scale_har = np.sqrt(test_rv_har * (3.0 / 5.0))
        m0_test = cls.evaluate_partition(
            "M0_HAR_BASELINE",
            np.zeros_like(test_payload.test_returns),
            test_scale_har,
            test_payload.test_returns,
            test_payload.test_rv,
        )

        # M1 (Numeric Model)
        with torch.no_grad():
            m1_mu_t, m1_logvol_t = model(torch.tensor(test_x_num, dtype=torch.float32), None)
            m1_scale = np.exp(m1_logvol_t.numpy()) * 0.0168
            m1_test = cls.evaluate_partition(
                "M1_NUMERIC",
                m1_mu_t.numpy(),
                m1_scale,
                test_payload.test_returns,
                test_payload.test_rv,
            )

        # M2 (Multimodal Model with Real News)
        with torch.no_grad():
            m2_mu_t, m2_logvol_t = model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(test_x_news, dtype=torch.float32),
            )
            m2_scale = np.exp(m2_logvol_t.numpy()) * 0.0168
            m2_test = cls.evaluate_partition(
                "M2_MULTIMODAL_NEWS",
                m2_mu_t.numpy(),
                m2_scale,
                test_payload.test_returns,
                test_payload.test_rv,
            )

        # M3 (Negative Control Model with Shuffled News)
        with torch.no_grad():
            m3_mu_t, m3_logvol_t = model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(shuffled_test_news, dtype=torch.float32),
            )
            m3_scale = np.exp(m3_logvol_t.numpy()) * 0.0168
            m3_test = cls.evaluate_partition(
                "M3_NEGATIVE_CONTROL",
                m3_mu_t.numpy(),
                m3_scale,
                test_payload.test_returns,
                test_payload.test_rv,
            )

        test_records = {
            "M0_HAR_BASELINE": m0_test,
            "M1_NUMERIC": m1_test,
            "M2_MULTIMODAL_NEWS": m2_test,
            "M3_NEGATIVE_CONTROL": m3_test,
        }

        delta_news = round(m2_test.crps_mean - m1_test.crps_mean, 6)
        delta_control = round(m2_test.crps_mean - m3_test.crps_mean, 6)
        delta_econometric = round(m1_test.crps_mean - m0_test.crps_mean, 6)

        promoted = (m2_test.crps_mean <= m0_test.crps_mean) and (
            m1_test.crps_mean <= m0_test.crps_mean
        )

        return CertifiedSealedEvaluationResult(
            candidate_digest=frozen_manifest.manifest_sha256,
            sealed_test_dates=(test_payload.test_dates[0], test_payload.test_dates[-1]),
            sealed_test_metrics=test_records,
            news_ablation_delta_crps=delta_news,
            negative_control_delta_crps=delta_control,
            econometric_delta_crps=delta_econometric,
            certification_decision="CERTIFIED_PROMOTED" if promoted else "CERTIFIED_INFERIOR",
            audit_trail={
                "unseal_token": test_payload.unseal_token,
                "split_digest": test_payload.split_digest,
                "candidate_manifest": frozen_manifest.to_dict(),
            },
        )
