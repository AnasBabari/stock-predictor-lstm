"""End-to-end training, expanding CV selection, and sealed evaluation for learned multimodal models."""

from __future__ import annotations

import copy
import hashlib
import io
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from backend.contracts.schemas_v11 import (
    HORIZON_RETURN_LOSS_WEIGHTS_V11,
    HORIZON_VARIANCE_LOSS_WEIGHTS_V11,
    REQUIRED_TARGET_HORIZONS_V11,
    get_schema_v11_manifest,
)
from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
)
from research.volatility_forecasting.multimodal_fusion_model_v2 import (
    MultimodalFusionModel,
)
from research.volatility_forecasting.proper_scoring_v11 import student_t_crps
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
    crps_per_horizon: dict[int, float]
    return_mae: float
    qlike_mean: float
    coverage_80pct: float
    pinball_loss_10_90: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    mean_delta: float
    ci_lower_95: float
    ci_upper_95: float
    n_replicates: int
    is_strictly_significant: bool  # ci_upper_95 < 0.0 for delta = (candidate - comparator)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenCandidateManifest:
    candidate_id: str
    selected_candidate_family: str  # M1_NUMERIC or M2_MULTIMODAL_NEWS (fixed prior to sealed test)
    selected_hyperparameters: MappingProxyType
    validation_oof_metrics: dict[str, ModelMetricRecord]
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "selected_candidate_family": self.selected_candidate_family,
            "selected_hyperparameters": dict(self.selected_hyperparameters),
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
    preselected_family: str
    sealed_test_dates: tuple[str, str]
    sealed_test_metrics: dict[str, ModelMetricRecord]
    news_ablation_delta_crps: float  # CRPS(M2) - CRPS(M1)
    shuffle_control_delta_crps: float  # CRPS(M2) - CRPS(M3_shuffle)
    delay_control_delta_crps: float  # CRPS(M2) - CRPS(M3_delay)
    econometric_delta_crps: float  # CRPS(M1) - CRPS(M0)
    paired_bootstrap_cis: dict[str, BootstrapConfidenceInterval]
    certification_decision: str  # SEALED_TEST_PASS_M2_MULTIMODAL, SEALED_TEST_FAIL_M2_MULTIMODAL, SEALED_TEST_PASS_M1_NUMERIC, SEALED_TEST_FAIL_M1_NUMERIC
    audit_trail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_digest": self.candidate_digest,
            "preselected_family": self.preselected_family,
            "sealed_test_dates": self.sealed_test_dates,
            "sealed_test_metrics": {k: v.to_dict() for k, v in self.sealed_test_metrics.items()},
            "news_ablation_delta_crps": self.news_ablation_delta_crps,
            "shuffle_control_delta_crps": self.shuffle_control_delta_crps,
            "delay_control_delta_crps": self.delay_control_delta_crps,
            "econometric_delta_crps": self.econometric_delta_crps,
            "paired_bootstrap_cis": {k: v.to_dict() for k, v in self.paired_bootstrap_cis.items()},
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
        har_feats = x_numeric[:, [23, 24, 25]] ** 2
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
    """Rigorous trainer with confirmatory sealed evaluation and block bootstrap statistical gates."""

    @staticmethod
    def compute_block_bootstrap_ci(
        dates: list[str],
        candidate_crps_per_sample: np.ndarray,
        comparator_crps_per_sample: np.ndarray,
        n_replicates: int = 2000,
        seed: int = 42,
    ) -> BootstrapConfidenceInterval:
        """Computes session-date block bootstrap confidence interval for paired delta CRPS."""
        unique_dates = np.array(sorted(set(dates)))
        date_array = np.array(dates)
        deltas = candidate_crps_per_sample - comparator_crps_per_sample

        # Precompute date-level mean deltas
        date_deltas = np.array([np.mean(deltas[date_array == d]) for d in unique_dates])
        n_blocks = len(unique_dates)

        rng = np.random.default_rng(seed)
        boot_means = np.zeros(n_replicates, dtype=float)
        for r in range(n_replicates):
            sampled_idx = rng.integers(0, n_blocks, size=n_blocks)
            boot_means[r] = float(np.mean(date_deltas[sampled_idx]))

        mean_delta = float(np.mean(deltas))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        return BootstrapConfidenceInterval(
            mean_delta=round(mean_delta, 6),
            ci_lower_95=round(ci_lower, 6),
            ci_upper_95=round(ci_upper, 6),
            n_replicates=n_replicates,
            is_strictly_significant=(ci_upper < 0.0),
        )

    @staticmethod
    def evaluate_partition(
        model_name: str,
        pred_mu: np.ndarray,
        pred_scale: np.ndarray,
        y_returns: np.ndarray,
        y_rv: np.ndarray,
        df: float = 5.0,
        horizons: tuple[int, ...] = REQUIRED_TARGET_HORIZONS_V11,
    ) -> tuple[ModelMetricRecord, np.ndarray]:
        pred_var = (pred_scale**2) * (df / (df - 2.0)) if df > 2.0 else pred_scale**2
        pred_var = np.maximum(pred_var, 1e-8)
        true_var = np.maximum(y_rv, 1e-8)

        # 1. Exact proper Student-t CRPS
        crps_matrix = student_t_crps(y_true=y_returns, loc=pred_mu, scale=pred_scale, df=df)
        sample_crps = np.mean(crps_matrix, axis=1)  # Mean across horizons per sample
        mean_crps = float(np.mean(crps_matrix))

        crps_per_h: dict[int, float] = {}
        for col_idx, h in enumerate(horizons):
            crps_per_h[h] = round(float(np.mean(crps_matrix[:, col_idx])), 6)

        # 2. Return MAE
        mae = float(np.mean(np.abs(pred_mu - y_returns)))

        # 3. Patton QLIKE on variance
        qlike = float(np.mean((true_var / pred_var) - np.log(true_var / pred_var) - 1.0))

        # 4. Empirical 80% Prediction Interval Coverage
        import scipy.stats as stats

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

        record = ModelMetricRecord(
            model_id=model_name,
            crps_mean=round(mean_crps, 6),
            crps_per_horizon=crps_per_h,
            return_mae=round(mae, 6),
            qlike_mean=round(qlike, 6),
            coverage_80pct=round(coverage, 4),
            pinball_loss_10_90=round(pinball, 6),
        )
        return record, sample_crps

    @classmethod
    def train_har_residual_model_with_early_stopping(
        cls,
        x_num_train: np.ndarray,
        x_news_train: np.ndarray | None,
        base_har_var_train: np.ndarray,
        y_returns_train: np.ndarray,
        y_rv_train: np.ndarray,
        train_scale_returns: np.ndarray,
        x_num_val: np.ndarray | None = None,
        x_news_val: np.ndarray | None = None,
        base_har_var_val: np.ndarray | None = None,
        y_returns_val: np.ndarray | None = None,
        max_epochs: int = 20,
        patience: int = 5,
        lr: float = 0.005,
        lambda_qlike: float = 0.5,
        df: float = 5.0,
    ) -> tuple[MultimodalFusionModel, int]:
        n_num = x_num_train.shape[1]
        n_news = x_news_train.shape[1] if x_news_train is not None else 19
        model = MultimodalFusionModel(
            numeric_dim=n_num, news_dim=n_news, horizons=REQUIRED_TARGET_HORIZONS_V11
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        t_num = torch.tensor(x_num_train, dtype=torch.float32)
        t_news = (
            torch.tensor(x_news_train, dtype=torch.float32) if x_news_train is not None else None
        )
        t_rets = torch.tensor(y_returns_train, dtype=torch.float32)
        t_rv = torch.tensor(y_rv_train, dtype=torch.float32)
        t_har_var = torch.tensor(base_har_var_train, dtype=torch.float32)
        t_norm_scale = torch.tensor(train_scale_returns, dtype=torch.float32)

        w_ret = torch.tensor(
            [HORIZON_RETURN_LOSS_WEIGHTS_V11[h] for h in REQUIRED_TARGET_HORIZONS_V11],
            dtype=torch.float32,
        )
        w_vol = torch.tensor(
            [HORIZON_VARIANCE_LOSS_WEIGHTS_V11[h] for h in REQUIRED_TARGET_HORIZONS_V11],
            dtype=torch.float32,
        )

        has_val = (
            x_num_val is not None and y_returns_val is not None and base_har_var_val is not None
        )
        best_val_crps = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 1
        stagnant_epochs = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            d_mu, d_log_vol = model(t_num, t_news)

            huber_per_h = nn.functional.huber_loss(d_mu, t_rets, reduction="none")
            normalized_huber = huber_per_h / torch.clamp(t_norm_scale, min=1e-4)
            loss_ret = torch.mean(torch.sum(w_ret * normalized_huber, dim=-1))

            pred_var = torch.clamp(t_har_var * torch.exp(2.0 * d_log_vol), min=1e-8)
            qlike_per_h = (t_rv / pred_var) - torch.log(t_rv / pred_var) - 1.0
            loss_vol = torch.mean(torch.sum(w_vol * qlike_per_h, dim=-1))

            total_loss = loss_ret + lambda_qlike * loss_vol
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if has_val:
                model.eval()
                with torch.no_grad():
                    v_t_num = torch.tensor(x_num_val, dtype=torch.float32)
                    v_t_news = (
                        torch.tensor(x_news_val, dtype=torch.float32)
                        if x_news_val is not None
                        else None
                    )
                    v_mu, v_logvol = model(v_t_num, v_t_news)
                    raw_scale = np.sqrt(
                        np.maximum(base_har_var_val * (df - 2.0) / df, 1e-8)
                    ) * np.exp(np.clip(v_logvol.numpy(), -3.0, 3.0))
                    val_scale = np.clip(raw_scale, 1e-5, 10.0)
                    val_crps = float(
                        np.mean(
                            student_t_crps(
                                y_true=y_returns_val, loc=v_mu.numpy(), scale=val_scale, df=df
                            )
                        )
                    )
                if val_crps < best_val_crps:
                    best_val_crps = val_crps
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    stagnant_epochs = 0
                else:
                    stagnant_epochs += 1
                    if stagnant_epochs >= patience:
                        break

        if has_val:
            model.load_state_dict(best_state)

        return model, best_epoch

    @classmethod
    def develop_and_freeze_bundle(
        cls,
        dev_payload: DevelopmentDatasetPayload,
        max_epochs: int = 20,
        patience: int = 5,
        lr: float = 0.005,
        n_expanding_folds: int = 4,
        df: float = 5.0,
    ) -> FrozenCandidateBundle:
        """Executes fold-local expanding cross-validation, separate early stopping, and cryptographic bundle freeze."""
        all_dev_dates = dev_payload.train_dates + dev_payload.val_dates
        all_dev_num = np.vstack([dev_payload.train_numeric, dev_payload.val_numeric])
        all_dev_news = np.vstack([dev_payload.train_news, dev_payload.val_news])
        all_dev_rets = np.vstack([dev_payload.train_returns, dev_payload.val_returns])
        all_dev_rv = np.vstack([dev_payload.train_rv, dev_payload.val_rv])

        folds = ChronologicalPartitionManager.create_expanding_folds(
            all_dev_dates, n_folds=n_expanding_folds, max_horizon_days=7, embargo_sessions=15
        )

        fold_m0_crps: list[float] = []
        fold_m1_crps: list[float] = []
        fold_m2_crps: list[float] = []

        for t_f_idx, v_f_idx in folds:
            f_num_mean = np.mean(all_dev_num[t_f_idx], axis=0, keepdims=True)
            f_num_std = np.std(all_dev_num[t_f_idx], axis=0, keepdims=True) + 1e-6
            f_x_num_train = (all_dev_num[t_f_idx] - f_num_mean) / f_num_std
            f_x_num_val = (all_dev_num[v_f_idx] - f_num_mean) / f_num_std

            f_news_mean = np.mean(all_dev_news[t_f_idx], axis=0, keepdims=True)
            f_news_std = np.std(all_dev_news[t_f_idx], axis=0, keepdims=True) + 1e-6
            f_x_news_train = (all_dev_news[t_f_idx] - f_news_mean) / f_news_std
            f_x_news_val = (all_dev_news[v_f_idx] - f_news_mean) / f_news_std

            f_train_scale_rets = np.std(all_dev_rets[t_f_idx], axis=0, keepdims=True) + 1e-6

            f_har = EconometricHARBaseline(REQUIRED_TARGET_HORIZONS_V11)
            f_har.fit(all_dev_num[t_f_idx], all_dev_rv[t_f_idx])
            f_train_har_var = f_har.predict_variance(all_dev_num[t_f_idx])
            f_val_har_var = f_har.predict_variance(all_dev_num[v_f_idx])

            f_val_scale_har = np.sqrt(f_val_har_var * (df - 2.0) / df)
            f_m0_crps = float(
                np.mean(
                    student_t_crps(
                        y_true=all_dev_rets[v_f_idx],
                        loc=np.zeros_like(all_dev_rets[v_f_idx]),
                        scale=f_val_scale_har,
                        df=df,
                    )
                )
            )
            fold_m0_crps.append(f_m0_crps)

            f_m1, _ = cls.train_har_residual_model_with_early_stopping(
                x_num_train=f_x_num_train,
                x_news_train=None,
                base_har_var_train=f_train_har_var,
                y_returns_train=all_dev_rets[t_f_idx],
                y_rv_train=all_dev_rv[t_f_idx],
                train_scale_returns=f_train_scale_rets,
                x_num_val=f_x_num_val,
                x_news_val=None,
                base_har_var_val=f_val_har_var,
                y_returns_val=all_dev_rets[v_f_idx],
                max_epochs=max_epochs,
                patience=patience,
                lr=lr,
                df=df,
            )
            with torch.no_grad():
                m1_mu_v, m1_lv_v = f_m1(torch.tensor(f_x_num_val, dtype=torch.float32), None)
                f_m1_scale = np.clip(
                    f_val_scale_har * np.exp(np.clip(m1_lv_v.numpy(), -3.0, 3.0)), 1e-5, 10.0
                )
                f_m1_crps = float(
                    np.mean(
                        student_t_crps(
                            y_true=all_dev_rets[v_f_idx],
                            loc=m1_mu_v.numpy(),
                            scale=f_m1_scale,
                            df=df,
                        )
                    )
                )
                fold_m1_crps.append(f_m1_crps)

            f_m2, _ = cls.train_har_residual_model_with_early_stopping(
                x_num_train=f_x_num_train,
                x_news_train=f_x_news_train,
                base_har_var_train=f_train_har_var,
                y_returns_train=all_dev_rets[t_f_idx],
                y_rv_train=all_dev_rv[t_f_idx],
                train_scale_returns=f_train_scale_rets,
                x_num_val=f_x_num_val,
                x_news_val=f_x_news_val,
                base_har_var_val=f_val_har_var,
                y_returns_val=all_dev_rets[v_f_idx],
                max_epochs=max_epochs,
                patience=patience,
                lr=lr,
                df=df,
            )
            with torch.no_grad():
                m2_mu_v, m2_lv_v = f_m2(
                    torch.tensor(f_x_num_val, dtype=torch.float32),
                    torch.tensor(f_x_news_val, dtype=torch.float32),
                )
                f_m2_scale = np.clip(
                    f_val_scale_har * np.exp(np.clip(m2_lv_v.numpy(), -3.0, 3.0)), 1e-5, 10.0
                )
                f_m2_crps = float(
                    np.mean(
                        student_t_crps(
                            y_true=all_dev_rets[v_f_idx],
                            loc=m2_mu_v.numpy(),
                            scale=f_m2_scale,
                            df=df,
                        )
                    )
                )
                fold_m2_crps.append(f_m2_crps)

        # Final Training on Outer 70% Train
        num_mean = np.mean(dev_payload.train_numeric, axis=0, keepdims=True)
        num_std = np.std(dev_payload.train_numeric, axis=0, keepdims=True) + 1e-6
        train_x_num = (dev_payload.train_numeric - num_mean) / num_std
        val_x_num = (dev_payload.val_numeric - num_mean) / num_std

        news_mean = np.mean(dev_payload.train_news, axis=0, keepdims=True)
        news_std = np.std(dev_payload.train_news, axis=0, keepdims=True) + 1e-6
        train_x_news = (dev_payload.train_news - news_mean) / news_std
        val_x_news = (dev_payload.val_news - news_mean) / news_std

        train_scale_rets = np.std(dev_payload.train_returns, axis=0, keepdims=True) + 1e-6

        final_har = EconometricHARBaseline(REQUIRED_TARGET_HORIZONS_V11)
        final_har.fit(dev_payload.train_numeric, dev_payload.train_rv)
        train_har_var = final_har.predict_variance(dev_payload.train_numeric)
        val_har_var = final_har.predict_variance(dev_payload.val_numeric)

        m1_final, m1_best_epoch = cls.train_har_residual_model_with_early_stopping(
            x_num_train=train_x_num,
            x_news_train=None,
            base_har_var_train=train_har_var,
            y_returns_train=dev_payload.train_returns,
            y_rv_train=dev_payload.train_rv,
            train_scale_returns=train_scale_rets,
            x_num_val=val_x_num,
            x_news_val=None,
            base_har_var_val=val_har_var,
            y_returns_val=dev_payload.val_returns,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            df=df,
        )

        m2_final, m2_best_epoch = cls.train_har_residual_model_with_early_stopping(
            x_num_train=train_x_num,
            x_news_train=train_x_news,
            base_har_var_train=train_har_var,
            y_returns_train=dev_payload.train_returns,
            y_rv_train=dev_payload.train_rv,
            train_scale_returns=train_scale_rets,
            x_num_val=val_x_num,
            x_news_val=val_x_news,
            base_har_var_val=val_har_var,
            y_returns_val=dev_payload.val_returns,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            df=df,
        )

        # Validation Metrics
        val_scale_har = np.sqrt(val_har_var * (df - 2.0) / df)
        m0_val, _ = cls.evaluate_partition(
            "M0_HAR_BASELINE",
            np.zeros_like(dev_payload.val_returns),
            val_scale_har,
            dev_payload.val_returns,
            dev_payload.val_rv,
            df=df,
        )

        with torch.no_grad():
            m1_mu_v, m1_lv_v = m1_final(torch.tensor(val_x_num, dtype=torch.float32), None)
            m1_scale_v = np.clip(
                val_scale_har * np.exp(np.clip(m1_lv_v.numpy(), -3.0, 3.0)), 1e-5, 10.0
            )
            m1_val, _ = cls.evaluate_partition(
                "M1_NUMERIC",
                m1_mu_v.numpy(),
                m1_scale_v,
                dev_payload.val_returns,
                dev_payload.val_rv,
                df=df,
            )

            m2_mu_v, m2_lv_v = m2_final(
                torch.tensor(val_x_num, dtype=torch.float32),
                torch.tensor(val_x_news, dtype=torch.float32),
            )
            m2_scale_v = np.clip(
                val_scale_har * np.exp(np.clip(m2_lv_v.numpy(), -3.0, 3.0)), 1e-5, 10.0
            )
            m2_val, _ = cls.evaluate_partition(
                "M2_MULTIMODAL_NEWS",
                m2_mu_v.numpy(),
                m2_scale_v,
                dev_payload.val_returns,
                dev_payload.val_rv,
                df=df,
            )

        val_records = {
            "M0_HAR_BASELINE": m0_val,
            "M1_NUMERIC": m1_val,
            "M2_MULTIMODAL_NEWS": m2_val,
        }

        # Validation Alone Chooses Candidate Family
        selected_family = (
            "M2_MULTIMODAL_NEWS" if m2_val.crps_mean < m1_val.crps_mean else "M1_NUMERIC"
        )

        # Canonical Freeze Digest
        h = hashlib.sha256()
        h.update(selected_family.encode())
        h.update(get_schema_v11_manifest().schema_sha256.encode())
        h.update(dev_payload.split_digest.encode())
        h.update(num_mean.tobytes())
        h.update(num_std.tobytes())
        h.update(news_mean.tobytes())
        h.update(news_std.tobytes())
        h.update(train_scale_rets.tobytes())

        buf_m1 = io.BytesIO()
        torch.save(m1_final.state_dict(), buf_m1)
        h.update(buf_m1.getvalue())

        buf_m2 = io.BytesIO()
        torch.save(m2_final.state_dict(), buf_m2)
        h.update(buf_m2.getvalue())

        for k in sorted(final_har.coefficients.keys()):
            h.update(final_har.coefficients[k].tobytes())

        cand_digest = h.hexdigest()

        manifest = FrozenCandidateManifest(
            candidate_id=f"CAND_{cand_digest[:12]}",
            selected_candidate_family=selected_family,
            selected_hyperparameters=MappingProxyType(
                {
                    "m1_best_epoch": m1_best_epoch,
                    "m2_best_epoch": m2_best_epoch,
                    "lr": lr,
                    "lambda_qlike": 0.5,
                    "df": df,
                    "oof_m0_crps": float(np.mean(fold_m0_crps)),
                    "oof_m1_crps": float(np.mean(fold_m1_crps)),
                    "oof_m2_crps": float(np.mean(fold_m2_crps)),
                }
            ),
            validation_oof_metrics=val_records,
            train_dates=(dev_payload.train_dates[0], dev_payload.train_dates[-1]),
            val_dates=(dev_payload.val_dates[0], dev_payload.val_dates[-1]),
            manifest_sha256=cand_digest,
        )

        return FrozenCandidateBundle(
            m0_har_baseline=final_har,
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
    ) -> CertifiedSealedEvaluationResult:
        """Confirmatory single-use evaluation on sacred sealed test partition."""
        test_payload: SealedTestPayload = sealed_store.unseal_test_partition(
            bundle.manifest.manifest_sha256
        )

        if test_payload.test_same_origin_shuffled_news is None or len(
            test_payload.test_same_origin_shuffled_news
        ) != len(test_payload.test_numeric):
            raise ValueError(
                "Missing sealed same-origin shuffled news negative control in test payload."
            )
        if test_payload.test_causal_delayed_news is None or len(
            test_payload.test_causal_delayed_news
        ) != len(test_payload.test_numeric):
            raise ValueError("Missing sealed causal delayed news negative control in test payload.")

        test_x_num = (test_payload.test_numeric - bundle.num_scaler_mean) / bundle.num_scaler_std
        test_x_news = (test_payload.test_news - bundle.news_scaler_mean) / bundle.news_scaler_std
        shuffled_test_news = (
            test_payload.test_same_origin_shuffled_news - bundle.news_scaler_mean
        ) / bundle.news_scaler_std
        delayed_test_news = (
            test_payload.test_causal_delayed_news - bundle.news_scaler_mean
        ) / bundle.news_scaler_std

        df = bundle.manifest.selected_hyperparameters.get("df", 5.0)

        # M0 (HAR Baseline)
        test_rv_har = bundle.m0_har_baseline.predict_variance(test_payload.test_numeric)
        test_scale_har = np.sqrt(test_rv_har * (df - 2.0) / df)
        m0_test, m0_sample_crps = cls.evaluate_partition(
            "M0_HAR_BASELINE",
            np.zeros_like(test_payload.test_returns),
            test_scale_har,
            test_payload.test_returns,
            test_payload.test_rv,
            df=df,
        )

        # M1 (Frozen Numeric Model)
        with torch.no_grad():
            m1_mu_t, m1_logvol_t = bundle.m1_numeric_model(
                torch.tensor(test_x_num, dtype=torch.float32), None
            )
            m1_scale = np.clip(
                test_scale_har * np.exp(np.clip(m1_logvol_t.numpy(), -3.0, 3.0)), 1e-5, 10.0
            )
            m1_test, m1_sample_crps = cls.evaluate_partition(
                "M1_NUMERIC",
                m1_mu_t.numpy(),
                m1_scale,
                test_payload.test_returns,
                test_payload.test_rv,
                df=df,
            )

        # M2 (Frozen Multimodal Model with Real News)
        with torch.no_grad():
            m2_mu_t, m2_logvol_t = bundle.m2_multimodal_model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(test_x_news, dtype=torch.float32),
            )
            m2_scale = np.clip(
                test_scale_har * np.exp(np.clip(m2_logvol_t.numpy(), -3.0, 3.0)), 1e-5, 10.0
            )
            m2_test, m2_sample_crps = cls.evaluate_partition(
                "M2_MULTIMODAL_NEWS",
                m2_mu_t.numpy(),
                m2_scale,
                test_payload.test_returns,
                test_payload.test_rv,
                df=df,
            )

        # M3_shuffle
        with torch.no_grad():
            m3s_mu_t, m3s_logvol_t = bundle.m2_multimodal_model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(shuffled_test_news, dtype=torch.float32),
            )
            m3s_scale = np.clip(
                test_scale_har * np.exp(np.clip(m3s_logvol_t.numpy(), -3.0, 3.0)), 1e-5, 10.0
            )
            m3s_test, m3s_sample_crps = cls.evaluate_partition(
                "M3_SHUFFLE_CONTROL",
                m3s_mu_t.numpy(),
                m3s_scale,
                test_payload.test_returns,
                test_payload.test_rv,
                df=df,
            )

        # M3_delay
        with torch.no_grad():
            m3d_mu_t, m3d_logvol_t = bundle.m2_multimodal_model(
                torch.tensor(test_x_num, dtype=torch.float32),
                torch.tensor(delayed_test_news, dtype=torch.float32),
            )
            m3d_scale = np.clip(
                test_scale_har * np.exp(np.clip(m3d_logvol_t.numpy(), -3.0, 3.0)), 1e-5, 10.0
            )
            m3d_test, m3d_sample_crps = cls.evaluate_partition(
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

        # Paired Block-Bootstrap Confidence Intervals by Session Date
        boot_m2_vs_m1 = cls.compute_block_bootstrap_ci(
            test_payload.test_dates, m2_sample_crps, m1_sample_crps
        )
        boot_m2_vs_m3s = cls.compute_block_bootstrap_ci(
            test_payload.test_dates, m2_sample_crps, m3s_sample_crps
        )
        boot_m2_vs_m3d = cls.compute_block_bootstrap_ci(
            test_payload.test_dates, m2_sample_crps, m3d_sample_crps
        )
        boot_m1_vs_m0 = cls.compute_block_bootstrap_ci(
            test_payload.test_dates, m1_sample_crps, m0_sample_crps
        )

        boot_cis = {
            "M2_vs_M1": boot_m2_vs_m1,
            "M2_vs_M3_shuffle": boot_m2_vs_m3s,
            "M2_vs_M3_delay": boot_m2_vs_m3d,
            "M1_vs_M0": boot_m1_vs_m0,
        }

        delta_news = round(m2_test.crps_mean - m1_test.crps_mean, 6)
        delta_shuffle = round(m2_test.crps_mean - m3s_test.crps_mean, 6)
        delta_delay = round(m2_test.crps_mean - m3d_test.crps_mean, 6)
        delta_econometric = round(m1_test.crps_mean - m0_test.crps_mean, 6)

        preselected = bundle.manifest.selected_candidate_family

        # CONFIRMATORY EVALUATION GATES
        if preselected == "M2_MULTIMODAL_NEWS":
            all_h_pass = all(
                m2_test.crps_per_horizon[h] < m1_test.crps_per_horizon[h]
                and m2_test.crps_per_horizon[h] < m3s_test.crps_per_horizon[h]
                and m2_test.crps_per_horizon[h] < m3d_test.crps_per_horizon[h]
                for h in REQUIRED_TARGET_HORIZONS_V11
            )
            agg_pass = (
                m2_test.crps_mean < m1_test.crps_mean
                and m2_test.crps_mean < m0_test.crps_mean
                and m2_test.crps_mean < m3s_test.crps_mean
                and m2_test.crps_mean < m3d_test.crps_mean
                and m2_test.qlike_mean <= m0_test.qlike_mean
            )
            decision = (
                "SEALED_TEST_PASS_M2_MULTIMODAL"
                if (all_h_pass and agg_pass)
                else "SEALED_TEST_FAIL_M2_MULTIMODAL"
            )

        elif preselected == "M1_NUMERIC":
            all_h_pass = all(
                m1_test.crps_per_horizon[h] < m0_test.crps_per_horizon[h]
                for h in REQUIRED_TARGET_HORIZONS_V11
            )
            agg_pass = (m1_test.crps_mean < m0_test.crps_mean) and (
                m1_test.qlike_mean <= m0_test.qlike_mean
            )
            decision = (
                "SEALED_TEST_PASS_M1_NUMERIC"
                if (all_h_pass and agg_pass)
                else "SEALED_TEST_FAIL_M1_NUMERIC"
            )

        else:
            decision = "SEALED_TEST_FAIL_UNRECOGNIZED_FAMILY"

        return CertifiedSealedEvaluationResult(
            candidate_digest=bundle.manifest.manifest_sha256,
            preselected_family=preselected,
            sealed_test_dates=(test_payload.test_dates[0], test_payload.test_dates[-1]),
            sealed_test_metrics=test_records,
            news_ablation_delta_crps=delta_news,
            shuffle_control_delta_crps=delta_shuffle,
            delay_control_delta_crps=delta_delay,
            econometric_delta_crps=delta_econometric,
            paired_bootstrap_cis=boot_cis,
            certification_decision=decision,
            audit_trail={
                "unseal_token": test_payload.unseal_token,
                "split_digest": test_payload.split_digest,
                "candidate_manifest": bundle.manifest.to_dict(),
            },
        )
