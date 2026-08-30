"""Synthetic Integration Smoke Runner for StockLSTM V11.

Used exclusively for end-to-end integration and software verification.
DOES NOT use real historical market or news data.
Writes experiment_type = 'SYNTHETIC_INTEGRATION_SMOKE' with certification_eligible = False.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from backend.contracts.schemas_v11 import (
    REQUIRED_TARGET_HORIZONS_V11,
)
from research.volatility_forecasting.candidate_freezer_v11 import (
    CandidateFreezerV11,
)
from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
)
from research.volatility_forecasting.exchange_calendar_v11 import (
    get_session_close_utc,
)
from research.volatility_forecasting.global_multimodal_trainer_v11 import (
    GlobalMultimodalTrainerV11,
)
from research.volatility_forecasting.historical_pit_dataset_builder_v11 import (
    HistoricalPITDatasetBuilderV11,
)
from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
)
from research.volatility_forecasting.news_coverage_auditor_v11 import (
    NewsCoverageAuditorV11,
    NewsCoverageGateV11,
)
from research.volatility_forecasting.sealed_dataset_store_v11 import (
    SealedDatasetStoreV11,
)
from research.volatility_forecasting.split_manifest_v11 import (
    SplitManifestBuilderV11,
)
from research.volatility_forecasting.stable_security_identity_v11 import (
    SecurityIdentityResolver,
    StableSecurityIdentity,
)


def run_synthetic_smoke() -> None:
    repo_root = Path(r"c:\Users\Babar\stock-predictor-lstm")
    artifacts_dir = repo_root / "artifacts"
    training_dir = artifacts_dir / "v11_training"
    ckpt_dir = training_dir / "checkpoints"
    metrics_dir = training_dir / "metrics"
    logs_dir = training_dir / "logs"

    for d in [artifacts_dir, training_dir, ckpt_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STOCKLSTM V11 SYNTHETIC INTEGRATION SMOKE RUNNER")
    print("=" * 70)

    # 1. PIT Universe from mock records
    print("\n[1/7] Building PIT Universe from synthetic records...")
    universe_identities: list[StableSecurityIdentity] = [
        StableSecurityIdentity(
            security_id="US.AMGN",
            exchange_mic="XNAS",
            provider_aliases=("AMGN",),
            ticker_intervals=(("AMGN", "2021-01-01", "2026-08-28"),),
            active_membership_intervals=(("2021-01-01", "2026-08-28"),),
        ),
        StableSecurityIdentity(
            security_id="US.AAPL",
            exchange_mic="XNAS",
            provider_aliases=("AAPL",),
            ticker_intervals=(("AAPL", "2021-01-01", "2026-08-28"),),
            active_membership_intervals=(("2021-01-01", "2026-08-28"),),
        ),
        StableSecurityIdentity(
            security_id="US.MSFT",
            exchange_mic="XNAS",
            provider_aliases=("MSFT",),
            ticker_intervals=(("MSFT", "2021-01-01", "2026-08-28"),),
            active_membership_intervals=(("2021-01-01", "2026-08-28"),),
        ),
        StableSecurityIdentity(
            security_id="US.NVDA",
            exchange_mic="XNAS",
            provider_aliases=("NVDA",),
            ticker_intervals=(("NVDA", "2021-01-01", "2026-08-28"),),
            active_membership_intervals=(("2021-01-01", "2026-08-28"),),
        ),
        StableSecurityIdentity(
            security_id="US.META",
            exchange_mic="XNAS",
            provider_aliases=("FB", "META"),
            ticker_intervals=(
                ("FB", "2021-01-01", "2022-06-08"),
                ("META", "2022-06-09", "2026-08-28"),
            ),
            active_membership_intervals=(("2021-01-01", "2026-08-28"),),
        ),
        StableSecurityIdentity(
            security_id="US.HON",
            exchange_mic="XNAS",
            provider_aliases=("HON",),
            ticker_intervals=(("HON", "2021-07-21", "2026-08-28"),),
            active_membership_intervals=(("2021-07-21", "2026-08-28"),),
        ),
        StableSecurityIdentity(
            security_id="US.CRWD",
            exchange_mic="XNAS",
            provider_aliases=("CRWD",),
            ticker_intervals=(("CRWD", "2021-08-26", "2026-08-28"),),
            active_membership_intervals=(("2021-08-26", "2026-08-28"),),
        ),
        StableSecurityIdentity(
            security_id="US.CSCO",
            exchange_mic="XNAS",
            provider_aliases=("CSCO",),
            ticker_intervals=(("CSCO", "2021-01-01", "2026-08-28"),),
            active_membership_intervals=(("2021-01-01", "2026-08-28"),),
        ),
    ]

    resolver = SecurityIdentityResolver(universe_identities)
    universe_manifest = {
        "universe_id": "NDX100_SYNTHETIC_SMOKE_V11",
        "as_of": "2026-08-28",
        "provider": "synthetic_rng",
        "experiment_type": "SYNTHETIC_INTEGRATION_SMOKE",
        "certification_eligible": False,
        "securities_count": len(universe_identities),
        "identities": [ident.to_dict() for ident in universe_identities],
    }
    univ_manifest_path = artifacts_dir / "v11_pit_universe_manifest.json"
    univ_manifest_path.write_text(json.dumps(universe_manifest, indent=2), encoding="utf-8")

    # 2. Synthetic Market History
    print("[2/7] Generating synthetic market series for pipeline smoke...")
    raw_dates = pd.date_range("2021-01-04", "2026-08-28", freq="B").strftime("%Y-%m-%d").tolist()
    valid_sessions = []
    for d in raw_dates:
        try:
            get_session_close_utc(d)
            valid_sessions.append(d)
        except ValueError:
            pass

    n_sessions = len(valid_sessions)
    rng_mkt = np.random.default_rng(1001)
    mkt_rets = rng_mkt.normal(0.0004, 0.012, size=n_sessions)
    mkt_close = 300.0 * np.exp(np.cumsum(mkt_rets))
    mkt_df = pd.DataFrame(
        {
            "Open": mkt_close * 0.998,
            "High": mkt_close * 1.006,
            "Low": mkt_close * 0.994,
            "Close": mkt_close,
            "Volume": rng_mkt.uniform(3e7, 8e7, size=n_sessions),
        },
        index=valid_sessions,
    )

    rng_sec = np.random.default_rng(1002)
    sec_rets = 0.7 * mkt_rets + rng_sec.normal(0.0001, 0.008, size=n_sessions)
    sec_close = 150.0 * np.exp(np.cumsum(sec_rets))
    sec_df = pd.DataFrame(
        {
            "Open": sec_close * 0.998,
            "High": sec_close * 1.005,
            "Low": sec_close * 0.995,
            "Close": sec_close,
            "Volume": rng_sec.uniform(1e7, 3e7, size=n_sessions),
        },
        index=valid_sessions,
    )

    equities_ohlcv: dict[str, pd.DataFrame] = {}
    membership_masks: dict[str, list[tuple[str, str]]] = {}

    for idx, ident in enumerate(universe_identities):
        rng_eq = np.random.default_rng(2000 + idx)
        beta = 0.8 + 0.1 * idx
        idiosyncratic = rng_eq.normal(0.0001, 0.014, size=n_sessions)
        eq_rets = beta * mkt_rets + idiosyncratic
        eq_close = (100.0 + 30.0 * idx) * np.exp(np.cumsum(eq_rets))
        eq_df = pd.DataFrame(
            {
                "Open": eq_close * 0.997,
                "High": eq_close * 1.008,
                "Low": eq_close * 0.992,
                "Close": eq_close,
                "Volume": rng_eq.uniform(2e6, 1.2e7, size=n_sessions),
            },
            index=valid_sessions,
        )
        equities_ohlcv[ident.security_id] = eq_df
        membership_masks[ident.security_id] = list(ident.active_membership_intervals)

    market_manifest = {
        "snapshot_id": "MARKET_SNAPSHOT_SYNTHETIC_SMOKE_V11",
        "provider": "synthetic_rng",
        "license_id": "synthetic_dev_smoke",
        "experiment_type": "SYNTHETIC_INTEGRATION_SMOKE",
        "certification_eligible": False,
        "as_of_utc": "2026-08-28T20:00:00Z",
        "session_count": n_sessions,
        "date_range": [valid_sessions[0], valid_sessions[-1]],
        "security_count": len(equities_ohlcv),
    }
    mkt_manifest_path = artifacts_dir / "v11_market_snapshot_manifest.json"
    mkt_manifest_path.write_text(json.dumps(market_manifest, indent=2), encoding="utf-8")

    # 3. Synthetic News Lake
    print("[3/7] Generating synthetic news lake for smoke...")
    news_articles: list[EnrichedNewsArticle] = []
    event_types_pool = ["clinical_trial", "earnings", "analyst", "regulatory", "general"]
    sources_pool = ["Reuters", "Bloomberg", "WSJ", "PR_Newswire", "BusinessWire"]

    art_id_counter = 1
    rng_news = np.random.default_rng(3001)

    for d_idx, d_str in enumerate(valid_sessions):
        if d_idx % 3 == 0:
            news_articles.append(
                EnrichedNewsArticle(
                    article_id=f"ART_MACRO_{art_id_counter}",
                    ticker="MARKET",
                    headline=f"Macro update on {d_str}",
                    source=sources_pool[rng_news.integers(0, len(sources_pool))],
                    published_at=f"{d_str}T14:00:00Z",
                    first_seen_at=f"{d_str}T14:00:05Z",
                    delivery_time=f"{d_str}T14:00:10Z",
                    ticker_relevance=0.4,
                    event_type="macro",
                    sentiment_score=float(rng_news.uniform(-0.5, 0.5)),
                    sentiment_magnitude=float(rng_news.uniform(0.2, 0.8)),
                    severity_score=float(rng_news.uniform(0.1, 0.6)),
                    uncertainty_score=float(rng_news.uniform(0.1, 0.5)),
                    embedding_vector=list(rng_news.normal(0, 1, size=4)),
                )
            )
            art_id_counter += 1

        for ident in universe_identities:
            if (
                resolver.is_active_constituent(ident.security_id, d_str)
                and rng_news.uniform(0, 1) < 0.40
            ):
                news_articles.append(
                    EnrichedNewsArticle(
                        article_id=f"ART_COMP_{art_id_counter}",
                        ticker=ident.security_id,
                        headline=f"Corporate update for {ident.security_id} on {d_str}",
                        source=sources_pool[rng_news.integers(0, len(sources_pool))],
                        published_at=f"{d_str}T15:30:00Z",
                        first_seen_at=f"{d_str}T15:30:05Z",
                        delivery_time=f"{d_str}T15:30:10Z",
                        ticker_relevance=1.0,
                        event_type=event_types_pool[rng_news.integers(0, len(event_types_pool))],
                        sentiment_score=float(rng_news.uniform(-0.8, 0.8)),
                        sentiment_magnitude=float(rng_news.uniform(0.3, 0.9)),
                        severity_score=float(rng_news.uniform(0.2, 0.7)),
                        uncertainty_score=float(rng_news.uniform(0.1, 0.6)),
                        embedding_vector=list(rng_news.normal(0, 1, size=4)),
                    )
                )
                art_id_counter += 1

    news_manifest = {
        "snapshot_id": "NEWS_LAKE_SYNTHETIC_SMOKE_V11",
        "provider": "synthetic_rng",
        "experiment_type": "SYNTHETIC_INTEGRATION_SMOKE",
        "certification_eligible": False,
        "total_articles": len(news_articles),
        "as_of_utc": "2026-08-28T20:00:00Z",
    }
    news_manifest_path = artifacts_dir / "v11_news_snapshot_manifest.json"
    news_manifest_path.write_text(json.dumps(news_manifest, indent=2), encoding="utf-8")

    # 4. Historical Panel
    print("[4/7] Constructing synthetic panel...")
    panel_dataset = HistoricalPITDatasetBuilderV11.construct_panel_from_series(
        equities_ohlcv=equities_ohlcv,
        sector_ohlcv=sec_df,
        market_ohlcv=mkt_df,
        news_articles=news_articles,
        membership_masks=membership_masks,
        horizons=REQUIRED_TARGET_HORIZONS_V11,
        warmup_sessions=65,
    )

    stock_origins = list(zip(panel_dataset.security_ids, panel_dataset.dates, strict=True))
    coverage_report = NewsCoverageAuditorV11.audit_coverage(
        stock_dates=stock_origins,
        news_articles=news_articles,
        gate_config=NewsCoverageGateV11(
            min_company_nonempty_fraction=0.25,
            min_multi_source_fraction=0.10,
            min_total_origins=500,
            minimum_year_coverage_floor=0.15,
            minimum_security_count=len(universe_identities),
        ),
    )

    report_json_path = artifacts_dir / "v11_real_news_coverage_report.json"
    report_json_path.write_text(json.dumps(coverage_report.to_dict(), indent=2), encoding="utf-8")
    coverage_report.write_markdown_report(artifacts_dir / "v11_real_news_coverage_report.md")

    # 5. Split Manifest
    print("[5/7] Partitioning chronological split...")
    split = ChronologicalPartitionManager.create_70_15_15_split(
        dates=panel_dataset.dates,
        max_horizon_days=7,
        embargo_sessions=30,
    )

    split_manifest = SplitManifestBuilderV11.build_and_save_manifest(
        dates=panel_dataset.dates,
        security_ids=panel_dataset.security_ids,
        split=split,
        target_path=artifacts_dir / "v11_split_manifest.json",
    )

    # 6. Training & Selection
    print("[6/7] Training expanding folds and selecting champion...")
    sealed_store = SealedDatasetStoreV11(
        dates=panel_dataset.dates,
        numeric_features=panel_dataset.numeric_features,
        news_features=panel_dataset.news_features,
        same_origin_shuffled_news=panel_dataset.same_origin_shuffled_news,
        causal_delayed_news=panel_dataset.causal_delayed_news,
        returns_targets=panel_dataset.returns_targets,
        rv_targets=panel_dataset.rv_targets,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
        split_digest=split.split_digest,
        lock_dir=artifacts_dir / "sealed_test_locks",
    )

    dev_payload = sealed_store.load_development_dataset()
    seeds = [41, 42, 43]
    seed_bundles: dict[int, Any] = {}

    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        bundle = GlobalMultimodalTrainerV11.develop_and_freeze_bundle(
            dev_payload=dev_payload,
            max_epochs=12,
            patience=3,
            lr=0.004,
            n_expanding_folds=4,
            df=5.0,
        )
        seed_bundles[s] = bundle
        torch.save(bundle.m1_numeric_model.state_dict(), ckpt_dir / f"m1_numeric_seed_{s}.pt")
        torch.save(bundle.m2_multimodal_model.state_dict(), ckpt_dir / f"m2_multimodal_seed_{s}.pt")

    champion_bundle = seed_bundles[42]
    val_metrics = champion_bundle.manifest.validation_oof_metrics

    m_path, s_path, freeze_digest = CandidateFreezerV11.freeze_and_save_bundle(
        bundle=champion_bundle,
        output_dir=artifacts_dir,
        git_sha="smoke_dev",
        panel_sha=panel_dataset.panel_sha256,
        split_sha=split_manifest.split_sha256,
    )

    selected_champion = champion_bundle.manifest.selected_candidate_family

    comp_data = {
        "experiment_type": "SYNTHETIC_INTEGRATION_SMOKE",
        "certification_eligible": False,
        "sealed_test_eligible": False,
        "provider": "synthetic_rng",
        "selected_champion_family": selected_champion,
        "learned_promotion": "NONE" if selected_champion == "M0_HAR_BASELINE" else "PROMOTED",
        "models_development_metrics": {k: v.to_dict() for k, v in val_metrics.items()},
        "master_freeze_digest": freeze_digest,
        "sealed_test_status": "PERMANENTLY_CLOSED_SYNTHETIC",
    }
    (artifacts_dir / "v11_development_model_comparison.json").write_text(
        json.dumps(comp_data, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("V11 SYNTHETIC SMOKE RUN COMPLETED")
    print("=" * 70)
    print(f"Selected Champion:       {selected_champion}")
    print(
        f"Validation CRPS:         M0={val_metrics['M0_HAR_BASELINE'].crps_mean:.6f} |"
        f" M1={val_metrics['M1_NUMERIC'].crps_mean:.6f} |"
        f" M2={val_metrics['M2_MULTIMODAL_NEWS'].crps_mean:.6f}"
    )
    print("Sealed Test Status:      PERMANENTLY CLOSED (SYNTHETIC SMOKE)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_synthetic_smoke()
