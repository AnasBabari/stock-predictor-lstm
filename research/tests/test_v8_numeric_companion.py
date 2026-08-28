from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.volatility_forecasting.baselines import (
    AdaptiveBaselineHorizon,
    AdaptiveBaselineSelection,
)
from research.volatility_forecasting.candidate_v8 import (
    V8MemberEvidence,
    save_v8_development_candidate,
    v8_ensemble_identity,
)
from research.volatility_forecasting.export import (
    load_prospective_v8_candidate_member,
    load_prospective_v8_numeric_companion_member,
)
from research.volatility_forecasting.folds import InnerTrainingSplit
from research.volatility_forecasting.model import (
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
    RobustSequenceScaler,
    TrainingResult,
    VolatilityLossWeights,
)
from research.volatility_forecasting.refit import (
    FrozenCandidate,
    FrozenEnsemble,
    candidate_identity,
)
from research.volatility_forecasting.v8_protocol import v8_manifest

PROTOCOL_HORIZONS = (1, 3, 5, 7, 14, 30)
SEED = 41


def _member(*, seed: int, news_feature_count: int) -> FrozenCandidate:
    """Small real CPU candidate whose weights, scalers, and identity are genuine."""
    config = BaselineResidualTCNConfig(
        feature_count=4,
        horizon_count=2,
        window_size=60,
        channels=8,
        dilations=(1, 2),
        news_feature_count=news_feature_count,
        news_channels=4,
        dropout=0.0,
    )
    model = BaselineResidualTCN(config).eval()
    scaler = RobustSequenceScaler(median=np.zeros(4), iqr=np.ones(4))
    news_scaler = (
        RobustSequenceScaler(
            median=np.zeros(news_feature_count),
            iqr=np.ones(news_feature_count),
        )
        if news_feature_count
        else None
    )
    training = TrainingResult(
        model=model,
        scaler=scaler,
        news_scaler=news_scaler,
        best_epoch=1,
        history=(),
        device="cpu",
        duration_seconds=0.0,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    comparison = AdaptiveBaselineSelection(
        horizons=tuple(
            AdaptiveBaselineHorizon(
                horizon=horizon,
                family="causal_log_har",
                blend_alpha=0.0,
                multiplicative_scale=1.0,
                calibration_qlike=0.1,
                har_calibration_qlike=0.1,
            )
            for horizon in PROTOCOL_HORIZONS
        )
    )
    fit_split = InnerTrainingSplit(
        fit_indices=np.arange(0, 10),
        early_stopping_indices=np.arange(10, 20),
        fit_end=np.datetime64("2024-01-10"),
        early_stopping_start=np.datetime64("2024-01-11"),
        early_stopping_end=np.datetime64("2024-01-20"),
    )
    variance_scale = np.asarray((2.0, 3.0))
    return_variance_scale = np.asarray((0.5, 0.25))
    baseline_return_scale = np.ones(2)
    loss_weights = VolatilityLossWeights()
    candidate = FrozenCandidate(
        training=training,
        architecture=config,
        fit_split=fit_split,
        seed=seed,
        epoch_budget=1,
        variance_scale=variance_scale,
        return_variance_scale=return_variance_scale,
        comparison_baseline=comparison,
        baseline_return_variance_scale=baseline_return_scale,
        model_identity="fixture",
        loss_weights=loss_weights,
    )
    identity = candidate_identity(
        candidate.training,
        architecture=config,
        seed=seed,
        epoch_budget=1,
        variance_scale=variance_scale,
        return_variance_scale=return_variance_scale,
        comparison_baseline=comparison,
        baseline_return_variance_scale=baseline_return_scale,
        loss_weights=loss_weights,
    )
    return FrozenCandidate(**{**candidate.__dict__, "model_identity": identity})


def _ensemble(*, news_feature_count: int, seed: int = SEED) -> FrozenEnsemble:
    member = _member(seed=seed, news_feature_count=news_feature_count)
    return FrozenEnsemble(members=(member,), model_identity=v8_ensemble_identity((member,)))


def _evidence(seed: int = SEED) -> tuple[V8MemberEvidence, ...]:
    return (
        V8MemberEvidence(
            seed=seed,
            eligible=True,
            best_epoch=1,
            duration_seconds=0.0,
            metrics=(),
            ratio_upper_95=(),
            reasons=(),
        ),
    )


def _save_kwargs() -> dict[str, object]:
    return {
        "protocol": v8_manifest(news_enabled=True),
        "split_manifest": {"split": "fixture"},
        "split_manifest_sha256": "0" * 64,
        "panel_checksum": "sha256:" + "0" * 64,
        "universe_manifest_sha256": "sha256:" + "0" * 64,
        "news_snapshot_checksum": "sha256:" + "0" * 64,
        "universe_certifiable": True,
    }


def _news_extra_kwargs() -> dict[str, object]:
    return {
        "news_matrix_sha256": "sha256:" + "1" * 64,
        "news_feature_names": ("n1", "n2", "n3"),
        "news_ablation_evidence": ({"horizon": 1, "promoted": True},),
    }


def test_news_candidate_requires_frozen_numeric_companion(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="frozen numeric companion"):
        save_v8_development_candidate(
            output,
            ensemble=_ensemble(news_feature_count=3),
            evidence=_evidence(),
            **_save_kwargs(),  # type: ignore[arg-type]
            **_news_extra_kwargs(),  # type: ignore[arg-type]
        )
    assert not output.exists()


def test_news_candidate_persists_frozen_numeric_companion(tmp_path: Path) -> None:
    manifest = save_v8_development_candidate(
        tmp_path / "candidate",
        ensemble=_ensemble(news_feature_count=3),
        evidence=_evidence(),
        numeric_companion_ensemble=_ensemble(news_feature_count=0),
        numeric_companion_evidence=_evidence(),
        **_save_kwargs(),  # type: ignore[arg-type]
        **_news_extra_kwargs(),  # type: ignore[arg-type]
    )

    files = sorted(path.name for path in (tmp_path / "candidate").iterdir())
    assert "seed-41.pt" in files
    assert "numeric-seed-41.pt" in files
    assert manifest["artifact_role"] == "prospective_v8_development_candidate"
    assert manifest["validation_selected"] is True

    companion = manifest["numeric_companion"]
    assert companion["role"] == "predeclared_numeric_fallback_companion"
    assert companion["architecture"]["news_feature_count"] == 0
    news_seeds = {row["seed"] for row in manifest["members"]}
    companion_seeds = {row["seed"] for row in companion["members"]}
    assert news_seeds == companion_seeds == {SEED}
    assert {row["weights_file"] for row in companion["members"]} == {"numeric-seed-41.pt"}


def test_numeric_companion_round_trip_and_missing_section(tmp_path: Path) -> None:
    save_v8_development_candidate(
        tmp_path / "news-candidate",
        ensemble=_ensemble(news_feature_count=3),
        evidence=_evidence(),
        numeric_companion_ensemble=_ensemble(news_feature_count=0),
        numeric_companion_evidence=_evidence(),
        **_save_kwargs(),  # type: ignore[arg-type]
        **_news_extra_kwargs(),  # type: ignore[arg-type]
    )
    companion = load_prospective_v8_numeric_companion_member(
        tmp_path / "news-candidate", SEED
    )
    assert companion.seed == SEED
    assert companion.architecture.news_feature_count == 0
    news_member = load_prospective_v8_candidate_member(tmp_path / "news-candidate", SEED)
    assert news_member.architecture.news_feature_count == 3

    save_v8_development_candidate(
        tmp_path / "numeric-candidate",
        ensemble=_ensemble(news_feature_count=0),
        evidence=_evidence(),
        **_save_kwargs(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="section is missing"):
        load_prospective_v8_numeric_companion_member(tmp_path / "numeric-candidate", SEED)


def test_news_candidate_rejects_mismatched_or_news_companions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seeds must match"):
        save_v8_development_candidate(
            tmp_path / "mismatched",
            ensemble=_ensemble(news_feature_count=3),
            evidence=_evidence(),
            numeric_companion_ensemble=_ensemble(news_feature_count=0, seed=42),
            numeric_companion_evidence=_evidence(seed=42),
            **_save_kwargs(),  # type: ignore[arg-type]
            **_news_extra_kwargs(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="cannot contain news features"):
        save_v8_development_candidate(
            tmp_path / "news-companion",
            ensemble=_ensemble(news_feature_count=3),
            evidence=_evidence(),
            numeric_companion_ensemble=_ensemble(news_feature_count=3),
            numeric_companion_evidence=_evidence(),
            **_save_kwargs(),  # type: ignore[arg-type]
            **_news_extra_kwargs(),  # type: ignore[arg-type]
        )


def test_market_only_candidate_rejects_companion_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot persist news evidence"):
        save_v8_development_candidate(
            tmp_path / "market-only",
            ensemble=_ensemble(news_feature_count=0),
            evidence=_evidence(),
            numeric_companion_ensemble=_ensemble(news_feature_count=0),
            numeric_companion_evidence=_evidence(),
            **_save_kwargs(),  # type: ignore[arg-type]
        )
    assert not (tmp_path / "market-only").exists()
