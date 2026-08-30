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
from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
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
    model_id: (
        str  # M0_HAR_BASELINE, M1_NUMERIC, M2_MULTIMODAL_NEWS, M3_SHUFFLE_CONTROL, M3_DELAY_CONTROL
    )
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
    winning_model_family: str  # M2_MULTIMODAL_NEWS or M1_NUMERIC
    selected_hyperparameters: dict[str, Any]
    validation_oof_metrics: dict[str, ModelMetricRecord]
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "winning_model_family": self.winning_model_family,
            "selected_hyperparameters": self.selected_hyperparameters,
            "validation_oof_metrics": {
                k: v.to_dict() for k, v in self.validation_oof_metrics.items()
            },
            "train_dates": self.train_dates,
            "val_dates": self.val_dates,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass
class FrozenCandidateBundle:
    m0_har_baseline: EconometricHARBaseline
    m1_numeric_model: MultimodalFusionModel
    m2_multimodal_model: MultimodalFusionModel
    num_scaler_mean: np.ndarray
    num_scaler_std: np.ndarray
    news_scaler_mean: np.ndarray
    news_scaler_std: np.ndarray
    train_scale_returns: np.ndarray
    manifest: FrozenCandidateManifest


@dataclass(frozen=True)
class CertifiedSealedEvaluationResult:
    candidate_digest: str
    sealed_test_dates: tuple[str, str]
    sealed_test_metrics: dict[str, ModelMetricRecord]
    news_ablation_delta_crps: float  # CRPS(M2) - CRPS(M1)
    shuffle_control_delta_crps: float  # CRPS(M2) - CRPS(M3_shuffle)
    delay_control_delta_crps: float  # CRPS(M2) - CRPS(M3_delay)
    econometric_delta_crps: float  # CRPS(M1) - CRPS(M0)
    certification_decision: (
        str  # CERTIFIED_M2_PROMOTED, CERTIFIED_M1_NUMERIC_CHAMPION, CERTIFIED_INFERIOR
    )
    audit_trail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_digest": self.candidate_digest,
            "sealed_test_dates": self.sealed_test_dates,
            "sealed_test_metrics": {k: v.to_dict() for k, v in self.sealed_test_metrics.items()},
            "news_ablation_delta_crps": self.news_ablation_delta_crps,
            "shuffle_control_delta_crps": self.shuffle_control_delta_crps,
            "delay_control_delta_crps": self.delay_control_delta_crps,
            "econometric_delta_crps": self.econometric_delta_crps,
            "certification_decision": self.certification_decision,
            "audit_trail": self.audit_trail,
        }


class EconometricHARBaseline:
    """Fitted Heterogeneous Autoregressive (HAR-RV) multi-horizon volatility baseline (M0)."""

    def __init__(self, horizons: tuple[int, ...] = REQUIRED_TARGET_HORIZONS_V11) -> None:
        self.horizons = horizons
        self.coefficients: dict[int, np.ndarray] = {}

    def fit(self, x_numeric: np.ndarray, y_rv: np.ndarray) -> None:
        # Features 23, 24, 25 are har_daily_vol, har_weekly_vol, har_monthly_vol
        har_feats = x_numeric[:, [23, 24, 25]] ** 2  # Convert daily vol to RV
        n_samples = len(har_feats)
        X = np.column_stack([np.ones(n_samples), har_feats])

        for col_idx, h in enumerate(self.horizons):
            y_h = y_rv[:, col_idx]
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
    """Rigorous trainer supporting multi-task loss, expanding folds, HAR-residual scaling, and sealed evaluation."""

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
    def train_har_residual_model(
        cls,
        x_num: np.ndarray,
        x_news: np.ndarray | None,
        base_har_variance: np.ndarray,
        y_returns: np.ndarray,
        y_rv: np.ndarray,
        train_scale_returns: np.ndarray,
        epochs: int = 15,
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
        t_har_var = torch.tensor(base_har_variance, dtype=torch.float32)
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

            # 1. Normalized Location Huber Loss
            huber_per_h = nn.functional.huber_loss(d_mu, t_rets, reduction="none")
            normalized_huber = huber_per_h / torch.clamp(t_norm_scale, min=1e-4)
            loss_ret = torch.mean(torch.sum(w_ret * normalized_huber, dim=-1))

            # 2. Genuine Observation-Specific HAR-Residual Variance & QLIKE Loss
            # Pred_Var_{t,h} = Var_{HAR, t, h} * exp(2 * d_log_vol)
            pred_var = torch.clamp(t_har_var * torch.exp(2.0 * d_log_vol), min=1e-8)
            qlike_per_h = (t_rv / pred_var) - torch.log(t_rv / pred_var) - 1.0
            loss_vol = torch.mean(torch.sum(w_vol * qlike_per_h, dim=-1))

            total_loss = loss_ret + lambda_qlike * loss_vol
            total_loss.backward()
            optimizer.step()

        return model

    @classmethod
    def develop_and_freeze_bundle(
        cls,
        dev_payload: DevelopmentDatasetPayload,
        max_epochs: int = 15,
        lr: float = 0.005,
        n_expanding_folds: int = 4,
        df: float = 5.0,
    ) -> FrozenCandidateBundle:
        """Executes expanding cross-validation folds, early stopping, and returns frozen candidate bundle."""
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

        train_har_var = har_model.predict_variance(dev_payload.train_numeric)
        val_har_var = har_model.predict_variance(dev_payload.val_numeric)

        # 3. Expanding Folds Cross-Validation on Development Set
        all_dev_dates = dev_payload.train_dates + dev_payload.val_dates
        all_dev_num = np.vstack([train_x_num, val_x_num])
        all_dev_news = np.vstack([train_x_news, val_x_news])
        all_dev_rets = np.vstack([dev_payload.train_returns, dev_payload.val_returns])
        all_dev_rv = np.vstack([dev_payload.train_rv, dev_payload.val_rv])
        all_dev_har_var = np.vstack([train_har_var, val_har_var])

        folds = ChronologicalPartitionManager.create_expanding_folds(
            all_dev_dates, n_folds=n_expanding_folds, max_horizon_days=7, embargo_sessions=15
        )

        # Evaluate OOF performance across expanding folds for hyperparameter tuning & early stopping
        best_epoch = max_epochs
        best_oof_crps = float("inf")

        for trial_epoch in [5, max_epochs]:
            oof_m2_crps_list = []
            for t_f_idx, v_f_idx in folds:
                fold_model = cls.train_har_residual_model(
                    x_num=all_dev_num[t_f_idx],
                    x_news=all_dev_news[t_f_idx],
                    base_har_variance=all_dev_har_var[t_f_idx],
                    y_returns=all_dev_rets[t_f_idx],
                    y_rv=all_dev_rv[t_f_idx],
                    train_scale_returns=train_scale_rets,
                    epochs=trial_epoch,
                    lr=lr,
                    df=df,
                )
                with torch.no_grad():
                    mu_t, logvol_t = fold_model(
                        torch.tensor(all_dev_num[v_f_idx], dtype=torch.float32),
                        torch.tensor(all_dev_news[v_f_idx], dtype=torch.float32),
                    )
                    fold_scale = np.sqrt(all_dev_har_var[v_f_idx] * (df - 2.0) / df) * np.exp(
                        logvol_t.numpy()
                    )
                    f_crps = cls.compute_crps_student_t(
                        y_true=all_dev_rets[v_f_idx], mu=mu_t.numpy(), scale=fold_scale, df=df
                    )
                    oof_m2_crps_list.append(f_crps)
            mean_oof_crps = float(np.mean(oof_m2_crps_list))
            if mean_oof_crps < best_oof_crps:
                best_oof_crps = mean_oof_crps
                best_epoch = trial_epoch

        # 4. Train Final M1 (Numeric Only) on Train Partition
        m1_final = cls.train_har_residual_model(
            x_num=train_x_num,
            x_news=None,
            base_har_variance=train_har_var,
            y_returns=dev_payload.train_returns,
            y_rv=dev_payload.train_rv,
            train_scale_returns=train_scale_rets,
            epochs=best_epoch,
            lr=lr,
            df=df,
        )

        # 5. Train Final M2 (Multimodal Numeric + News) on Train Partition
        m2_final = cls.train_har_residual_model(
            x_num=train_x_num,
            x_news=train_x_news,
            base_har_variance=train_har_var,
            y_returns=dev_payload.train_returns,
            y_rv=dev_payload.train_rv,
            train_scale_returns=train_scale_rets,
            epochs=best_epoch,
            lr=lr,
            df=df,
        )

        # 6. Evaluate Validation Metrics
        # M0
        val_scale_har = np.sqrt(val_har_var * (df - 2.0) / df)
        m0_val = cls.evaluate_partition(
            "M0_HAR_BASELINE",
            np.zeros_like(dev_payload.val_returns),
            val_scale_har,
            dev_payload.val_returns,
            dev_payload.val_rv,
            df=df,
        )

        # M1
        with torch.no_grad():
            m1_mu_t, m1_logvol_t = m1_final(torch.tensor(val_x_num, dtype=torch.float32), None)
            m1_scale = val_scale_har * np.exp(m1_logvol_t.numpy())
            m1_val = cls.evaluate_partition(
                "M1_NUMERIC",
                m1_mu_t.numpy(),
                m1_scale,
                dev_payload.val_returns,
                dev_payload.val_rv,
                df=df,
            )

        # M2
        with torch.no_grad():
            m2_mu_t, m2_logvol_t = m2_final(
                torch.tensor(val_x_num, dtype=torch.float32),
                torch.tensor(val_x_news, dtype=torch.float32),
            )
            m2_scale = val_scale_har * np.exp(m2_logvol_t.numpy())
            m2_val = cls.evaluate_partition(
                "M2_MULTIMODAL_NEWS",
                m2_mu_t.numpy(),
                m2_scale,
                dev_payload.val_returns,
                dev_payload.val_rv,
                df=df,
            )

        val_records = {
            "M0_HAR_BASELINE": m0_val,
            "M1_NUMERIC": m1_val,
            "M2_MULTIMODAL_NEWS": m2_val,
        }

        winning_family = (
            "M2_MULTIMODAL_NEWS" if m2_val.crps_mean <= m1_val.crps_mean else "M1_NUMERIC"
        )

        raw_meta = f"{winning_family}:{best_epoch}:{lr}:{m2_val.crps_mean}:{m1_val.crps_mean}"
        cand_digest = hashlib.sha256(raw_meta.encode()).hexdigest()

        manifest = FrozenCandidateManifest(
            candidate_id=f"CAND_{cand_digest[:12]}",
            winning_model_family=winning_family,
            selected_hyperparameters={
                "epochs": best_epoch,
                "lr": lr,
                "lambda_qlike": 0.5,
                "df": df,
            },
            validation_oof_metrics=val_records,
            train_dates=(dev_payload.train_dates[0], dev_payload.train_dates[-1]),
            val_dates=(dev_payload.val_dates[0], dev_payload.val_dates[-1]),
            manifest_sha256=cand_digest,
        )

        return FrozenCandidateBundle(
            m0_har_baseline=har_model,
            m1_numeric_model=m1_final,
            m2_multimodal_model=m2_final,
            num_scaler_mean=num_mean,
            num_scaler_std=num_std,
            news_scaler_mean=news_mean,
            news_scaler_std=news_std,
            train_scale_returns=train_scale_rets,
            manifest=manifest,
        )

    @classmethod
    def evaluate_frozen_bundle_once(
        cls,
        bundle: FrozenCandidateBundle,
        sealed_store: SealedDatasetStoreV11,
        test_same_origin_shuffled_news: np.ndarray | None = None,
        test_causal_delayed_news: np.ndarray | None = None,
    ) -> CertifiedSealedEvaluationResult:
        """Single-use one-shot evaluation on sacred sealed test partition using the exact frozen bundle."""
        test_payload: SealedTestPayload = sealed_store.unseal_test_partition(
            bundle.manifest.manifest_sha256
        )

        # Apply train scalers
        test_x_num = (test_payload.test_numeric - bundle.num_scaler_mean) / bundle.num_scaler_std
        test_x_news = (test_payload.test_news - bundle.news_scaler_mean) / bundle.news_scaler_std

        # Causal Negative Controls
        if test_same_origin_shuffled_news is not None:
            shuffled_test_news = (
                test_same_origin_shuffled_news - bundle.news_scaler_mean
            ) / bundle.news_scaler_std
        else:
            rng = np.random.default_rng(2026)
            shuffled_test_news = rng.permutation(test_x_news)

        if test_causal_delayed_news is not None:
            delayed_test_news = (
                test_causal_delayed_news - bundle.news_scaler_mean
            ) / bundle.news_scaler_std
        else:
            delayed_test_news = np.roll(test_x_news, shift=10, axis=0)

        df = bundle.manifest.selected_hyperparameters.get("df", 5.0)

        # M0 (HAR Baseline)
        test_rv_har = bundle.m0_har_baseline.predict_variance(test_payload.test_numeric)
        test_scale_har = np.sqrt(test_rv_har * (df - 2.0) / df)
        m0_test = cls.evaluate_partition(
            "M0_HAR_BASELINE",
            np.zeros_like(test_payload.test_returns),
            test_scale_har,
            test_payload.test_returns,
            test_payload.test_rv,
            df=df,
        )

        # M1 (Separately Trained Frozen Numeric Model)
        with torch.no_grad():
            m1_mu_t, m1_logvol_t = bundle.m1_numeric_model(
                torch.tensor(test_x_num, dtype=torch.float32), None
            )
            m1_scale = test_scale_har * np.exp(m1_logvol_t.numpy())
            m1_test = cls.evaluate_partition(
                "M1_NUMERIC",
                m1_mu_t.numpy(),
                m1_scale,
                test_payload.test_returns,
                test_payload.test_rv,
                df=df,
            )

        # M2 (Separately Trained Frozen Multimodal Model with Real News)
        with torch.no_grad():
            m2_mu_t, m2_logvol_t = bundle.m2_multimodal_model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(test_x_news, dtype=torch.float32),
            )
            m2_scale = test_scale_har * np.exp(m2_logvol_t.numpy())
            m2_test = cls.evaluate_partition(
                "M2_MULTIMODAL_NEWS",
                m2_mu_t.numpy(),
                m2_scale,
                test_payload.test_returns,
                test_payload.test_rv,
                df=df,
            )

        # M3_shuffle (Negative Control with Same-Origin Shuffled News)
        with torch.no_grad():
            m3s_mu_t, m3s_logvol_t = bundle.m2_multimodal_model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(shuffled_test_news, dtype=torch.float32),
            )
            m3s_scale = test_scale_har * np.exp(m3s_logvol_t.numpy())
            m3s_test = cls.evaluate_partition(
                "M3_SHUFFLE_CONTROL",
                m3s_mu_t.numpy(),
                m3s_scale,
                test_payload.test_returns,
                test_payload.test_rv,
                df=df,
            )

        # M3_delay (Negative Control with Causal Delayed News)
        with torch.no_grad():
            m3d_mu_t, m3d_logvol_t = bundle.m2_multimodal_model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(delayed_test_news, dtype=torch.float32),
            )
            m3d_scale = test_scale_har * np.exp(m3d_logvol_t.numpy())
            m3d_test = cls.evaluate_partition(
                "M3_DELAY_CONTROL",
                m3d_mu_t.numpy(),
                m3d_scale,
                test_payload.test_returns,
                test_payload.test_rv,
                df=df,
            )

        test_records = {
            "M0_HAR_BASELINE": m0_test,
            "M1_NUMERIC": m1_test,
            "M2_MULTIMODAL_NEWS": m2_test,
            "M3_SHUFFLE_CONTROL": m3s_test,
            "M3_DELAY_CONTROL": m3d_test,
        }

        delta_news = round(m2_test.crps_mean - m1_test.crps_mean, 6)
        delta_shuffle = round(m2_test.crps_mean - m3s_test.crps_mean, 6)
        delta_delay = round(m2_test.crps_mean - m3d_test.crps_mean, 6)
        delta_econometric = round(m1_test.crps_mean - m0_test.crps_mean, 6)

        # Strict Promotion Hierarchy:
        # M2 is promoted only if M2 < M0, M2 < M1, M2 < M3_shuffle, and M2 < M3_delay
        if (
            m2_test.crps_mean < m0_test.crps_mean
            and m2_test.crps_mean < m1_test.crps_mean
            and m2_test.crps_mean < m3s_test.crps_mean
            and m2_test.crps_mean < m3d_test.crps_mean
        ):
            decision = "CERTIFIED_M2_PROMOTED"
        elif m1_test.crps_mean < m0_test.crps_mean:
            decision = "CERTIFIED_M1_NUMERIC_CHAMPION"
        else:
            decision = "CERTIFIED_INFERIOR"

        return CertifiedSealedEvaluationResult(
            candidate_digest=bundle.manifest.manifest_sha256,
            sealed_test_dates=(test_payload.test_dates[0], test_payload.test_dates[-1]),
            sealed_test_metrics=test_records,
            news_ablation_delta_crps=delta_news,
            shuffle_control_delta_crps=delta_shuffle,
            delay_control_delta_crps=delta_delay,
            econometric_delta_crps=delta_econometric,
            certification_decision=decision,
            audit_trail={
                "unseal_token": test_payload.unseal_token,
                "split_digest": test_payload.split_digest,
                "candidate_manifest": bundle.manifest.to_dict(),
            },
        )
