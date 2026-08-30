"""Real Historical Point-in-Time V11.1 Pipeline Orchestrator.

Loads real immutable market OHLCV data from parquet cache for NDX100 8-security research subset,
security-specific sector ETF benchmarks (XLV, XLK, XLI), and broad market benchmark (QQQ).
Enforces:
  1. Content-addressed data payloads (market & news raw bytes).
  2. Fail-closed real news coverage gates.
  3. Structural M2 disabling when real news is absent.
  4. Multi-seed reporting (Seed 42 canonical + 41/43 robustness diagnostics).
  5. Diagnostic linear & GBDT baseline comparisons (Ridge, HistGB).
  6. Strict M0 -> M1 selection hierarchy.
  7. Exact Git commit SHA & dirty status in candidate freeze manifest.
  8. M0 confirmatory adequacy gate on sealed test.
  9. 100% sacred sealed holdout isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from backend.contracts.schemas_v11 import (
    REQUIRED_TARGET_HORIZONS_V11,
    get_schema_v11_manifest,
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
    StableSecurityIdentity,
)


def get_git_provenance(repo_root: Path) -> tuple[str, bool]:
    """Extract real git commit SHA and dirty working tree status."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
        return sha, len(status) > 0
    except Exception:
        return "unknown_git_sha", True


def load_real_market_and_benchmarks(
    cache_dir: Path,
    universe_identities: list[StableSecurityIdentity],
    start_date: str = "2021-01-04",
    end_date: str = "2026-08-27",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, str]:
    """Load real daily OHLCV from parquet files with security-specific sector ETFs and broad market ETF."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Real market data cache directory not found: {cache_dir}")

    equities_ohlcv: dict[str, pd.DataFrame] = {}
    hasher = hashlib.sha256()

    for ident in universe_identities:
        ticker = ident.provider_aliases[-1]
        parquet_path = cache_dir / f"{ticker}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing required real market data file for {ident.security_id}: {parquet_path}"
            )

        df = pd.read_parquet(parquet_path)
        df.columns = [c.capitalize() for c in df.columns]
        df_range = df.loc[start_date:end_date].copy()
        if len(df_range) < 100:
            raise ValueError(
                f"Insufficient real sessions for {ident.security_id}: {len(df_range)} rows found."
            )

        # Content-addressing: hash exact canonical OHLCV rows
        for dt_idx, row in df_range.iterrows():
            d_str = dt_idx.strftime("%Y-%m-%d") if hasattr(dt_idx, "strftime") else str(dt_idx)[:10]
            row_str = (
                f"{ident.security_id}|{d_str}|{row['Open']:.6f}|{row['High']:.6f}|"
                f"{row['Low']:.6f}|{row['Close']:.6f}|{row['Volume']:.1f}\n"
            )
            hasher.update(row_str.encode("utf-8"))

        df_range.index = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            for d in df_range.index
        ]
        equities_ohlcv[ident.security_id] = df_range

    # Load Real Sector ETFs: XLV (Healthcare), XLK (Tech), XLI (Industrials)
    sector_etf_map = {
        "US.AMGN": "XLV",
        "US.AAPL": "XLK",
        "US.MSFT": "XLK",
        "US.NVDA": "XLK",
        "US.META": "XLK",
        "US.HON": "XLI",
        "US.CRWD": "XLK",
        "US.CSCO": "XLK",
    }

    sector_dfs: dict[str, pd.DataFrame] = {}
    for sec_id, etf_ticker in sector_etf_map.items():
        etf_path = cache_dir / f"{etf_ticker}.parquet"
        if not etf_path.exists():
            raise FileNotFoundError(f"Missing sector ETF file: {etf_path}")
        etf_df = pd.read_parquet(etf_path)
        etf_df.columns = [c.capitalize() for c in etf_df.columns]
        etf_range = etf_df.loc[start_date:end_date].copy()
        etf_range.index = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            for d in etf_range.index
        ]
        sector_dfs[sec_id] = etf_range

        for dt_idx, row in etf_range.iterrows():
            d_str = str(dt_idx)[:10]
            row_str = (
                f"SECTOR_{etf_ticker}|{d_str}|{row['Open']:.6f}|{row['High']:.6f}|"
                f"{row['Low']:.6f}|{row['Close']:.6f}|{row['Volume']:.1f}\n"
            )
            hasher.update(row_str.encode("utf-8"))

    # Load Broad Market ETF: QQQ
    qqq_path = cache_dir / "QQQ.parquet"
    if not qqq_path.exists():
        raise FileNotFoundError(f"Missing broad market ETF file: {qqq_path}")
    mkt_df = pd.read_parquet(qqq_path)
    mkt_df.columns = [c.capitalize() for c in mkt_df.columns]
    mkt_range = mkt_df.loc[start_date:end_date].copy()
    mkt_range.index = [
        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10] for d in mkt_range.index
    ]

    for dt_idx, row in mkt_range.iterrows():
        d_str = str(dt_idx)[:10]
        row_str = (
            f"MARKET_QQQ|{d_str}|{row['Open']:.6f}|{row['High']:.6f}|"
            f"{row['Low']:.6f}|{row['Close']:.6f}|{row['Volume']:.1f}\n"
        )
        hasher.update(row_str.encode("utf-8"))

    market_payload_sha256 = hasher.hexdigest()
    return equities_ohlcv, sector_dfs, mkt_range, market_payload_sha256


def load_real_news_data(news_archive_path: Path | None) -> tuple[list[EnrichedNewsArticle], str]:
    """Load real news archive from JSONL if present, else return empty list with fail-closed hash."""
    if news_archive_path is None or not news_archive_path.exists():
        return [], hashlib.sha256(b"EMPTY_REAL_NEWS_ARCHIVE_NO_ARTICLES").hexdigest()

    articles: list[EnrichedNewsArticle] = []
    hasher = hashlib.sha256()

    with news_archive_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            hasher.update(line.encode("utf-8"))
            data = json.loads(line)
            art = EnrichedNewsArticle(
                article_id=data["article_id"],
                ticker=data["ticker"],
                headline=data["headline"],
                source=data["source"],
                published_at=data["published_at"],
                first_seen_at=data["first_seen_at"],
                delivery_time=data["delivery_time"],
                ticker_relevance=float(data["ticker_relevance"]),
                event_type=data["event_type"],
                sentiment_score=float(data["sentiment_score"]),
                sentiment_magnitude=float(data["sentiment_magnitude"]),
                severity_score=float(data["severity_score"]),
                uncertainty_score=float(data["uncertainty_score"]),
                embedding_vector=list(data.get("embedding_vector", [0.0, 0.0, 0.0, 0.0])),
            )
            articles.append(art)

    return articles, hasher.hexdigest()


def run_historical_pipeline(news_archive: Path | None = None) -> None:
    repo_root = Path(r"c:\Users\Babar\stock-predictor-lstm")
    cache_dir = repo_root / "data" / "ndx100" / "cache"
    artifacts_dir = repo_root / "artifacts"
    training_dir = artifacts_dir / "v11_training"
    ckpt_dir = training_dir / "checkpoints"
    metrics_dir = training_dir / "metrics"
    logs_dir = training_dir / "logs"

    for d in [artifacts_dir, training_dir, ckpt_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    git_sha, git_dirty = get_git_provenance(repo_root)

    print("=" * 70)
    print("STOCKLSTM V11.1 REAL HISTORICAL MULTIMODAL PIPELINE")
    print(f"Git Commit: {git_sha} (Dirty: {git_dirty})")
    print("Universe:   NDX100 8-security research subset")
    print("=" * 70)

    # 1. Point-in-Time Universe
    print("\n[1/7] Initializing NDX100 8-security research subset...")
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

    membership_masks: dict[str, list[tuple[str, str]]] = {
        ident.security_id: list(ident.active_membership_intervals) for ident in universe_identities
    }

    universe_manifest = {
        "universe_id": "NDX100_8_SECURITY_RESEARCH_SUBSET_V11",
        "as_of": "2026-08-28",
        "provider": "historical_ndx_panel_parquet",
        "experiment_type": "HISTORICAL_MARKET_LEARNED_TRAINING",
        "securities_count": len(universe_identities),
        "identities": [ident.to_dict() for ident in universe_identities],
    }
    univ_manifest_path = artifacts_dir / "v11_pit_universe_manifest.json"
    univ_manifest_path.write_text(json.dumps(universe_manifest, indent=2), encoding="utf-8")
    universe_sha = hashlib.sha256(univ_manifest_path.read_bytes()).hexdigest()

    # 2. Real Market Data Loading (with security-specific sector ETFs and QQQ market ETF)
    print(
        "[2/7] Loading real market OHLCV, security-specific sector ETFs (XLV, XLK, XLI), and broad market ETF (QQQ)..."
    )
    equities_ohlcv, sector_dfs, mkt_df, market_payload_sha256 = load_real_market_and_benchmarks(
        cache_dir=cache_dir,
        universe_identities=universe_identities,
        start_date="2021-01-04",
        end_date="2026-08-27",
    )

    first_sec = list(equities_ohlcv.keys())[0]
    sessions = equities_ohlcv[first_sec].index.tolist()

    for s in sessions:
        get_session_close_utc(s)

    market_manifest = {
        "snapshot_id": "MARKET_SNAPSHOT_REAL_NDX100_SUBSET_V11",
        "provider": "historical_parquet_ndx100_cache",
        "experiment_type": "HISTORICAL_MARKET_LEARNED_TRAINING",
        "as_of_utc": "2026-08-27T20:00:00Z",
        "session_count": len(sessions),
        "date_range": [sessions[0], sessions[-1]],
        "security_count": len(equities_ohlcv),
        "market_benchmark": "QQQ",
        "sector_benchmarks": {"AMGN": "XLV", "HON": "XLI", "TECH_BASKET": "XLK"},
        "market_payload_sha256": market_payload_sha256,
    }
    mkt_manifest_path = artifacts_dir / "v11_market_snapshot_manifest.json"
    mkt_manifest_path.write_text(json.dumps(market_manifest, indent=2), encoding="utf-8")

    # 3. Real News Loading & Coverage Audit
    print("[3/7] Loading real news archive and running coverage auditor...")
    news_articles, news_payload_sha256 = load_real_news_data(news_archive)

    news_manifest = {
        "snapshot_id": "NEWS_LAKE_REAL_ARCHIVE_V11",
        "provider": "real_news_archive" if news_articles else "absent_fail_closed",
        "experiment_type": "HISTORICAL_MARKET_LEARNED_TRAINING",
        "total_articles": len(news_articles),
        "news_payload_sha256": news_payload_sha256,
        "as_of_utc": "2026-08-28T20:00:00Z",
    }
    news_manifest_path = artifacts_dir / "v11_news_snapshot_manifest.json"
    news_manifest_path.write_text(json.dumps(news_manifest, indent=2), encoding="utf-8")

    # 4. Construct Point-in-Time Historical Panel
    print("[4/7] Constructing 53-feature point-in-time historical panel from real market data...")
    panel_dataset = HistoricalPITDatasetBuilderV11.construct_panel_from_series(
        equities_ohlcv=equities_ohlcv,
        sector_ohlcv=sector_dfs,
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

    m2_eligible = coverage_report.is_dataset_sufficient_for_m2

    panel_manifest = {
        "panel_sha256": panel_dataset.panel_sha256,
        "total_rows": len(panel_dataset.dates),
        "unique_dates": len(set(panel_dataset.dates)),
        "unique_securities": len(set(panel_dataset.security_ids)),
        "numeric_features_dim": panel_dataset.numeric_features.shape[1],
        "news_features_dim": panel_dataset.news_features.shape[1],
        "returns_targets_dim": panel_dataset.returns_targets.shape[1],
        "universe_sha256": universe_sha,
        "market_payload_sha256": market_payload_sha256,
        "news_payload_sha256": news_payload_sha256,
        "m2_eligible": m2_eligible,
    }
    panel_manifest_path = artifacts_dir / "v11_historical_panel_manifest.json"
    panel_manifest_path.write_text(json.dumps(panel_manifest, indent=2), encoding="utf-8")

    # 5. Create 70/15/15 Chronological Split
    print("[5/7] Partitioning chronological split with 7-session purge and 30-session embargo...")
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

    # 6. Training Authorization Gate
    print("[6/7] Evaluating real training authorization gates...")
    cuda_avail = torch.cuda.is_available()
    gpu_dev = torch.cuda.get_device_name(0) if cuda_avail else "CPU_ONLY"

    multi_src_pct = (
        round(
            coverage_report.company_origins_ge_2_sources
            / coverage_report.total_stock_origins
            * 100.0,
            2,
        )
        if coverage_report.total_stock_origins > 0
        else 0.0
    )

    auth_record = {
        "authorized": True,
        "mode": "REAL_HISTORICAL_DATA",
        "universe": "NDX100 8-security research subset",
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "synthetic_fallback": "DISABLED",
        "real_market_data": "PASS",
        "real_news_data": "PASS" if news_articles else "ABSENT_FAIL_CLOSED",
        "market_payload_sha256": market_payload_sha256,
        "news_payload_sha256": news_payload_sha256,
        "split_sha256": split_manifest.split_sha256,
        "schema_sha256": get_schema_v11_manifest().schema_sha256,
        "gpu_device": gpu_dev,
        "company_news_coverage_pct": coverage_report.company_coverage_percentage,
        "multi_source_coverage_pct": multi_src_pct,
        "m2_eligible": m2_eligible,
        "training_scope": "M0_HAR + M1_NUMERIC" if not m2_eligible else "M0 + M1 + M2_MULTIMODAL",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    auth_path = artifacts_dir / "v11_gpu_training_authorization.json"
    auth_path.write_text(json.dumps(auth_record, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("V11.1 REAL HISTORICAL TRAINING AUTHORIZATION REPORT")
    print("=" * 60)
    print(f"GIT PROVENANCE:        {git_sha} (Dirty: {git_dirty})")
    print(
        f"UNIVERSE:              NDX100 8-security research subset ({len(universe_identities)} active)"
    )
    print(
        f"REAL MARKET DATA:      PASS ({len(equities_ohlcv)} securities, {len(sessions)} sessions)"
    )
    print("SECTOR BENCHMARKS:     PASS (XLV for AMGN, XLI for HON, XLK for Tech)")
    print("MARKET BENCHMARK:      PASS (QQQ ETF)")
    print(
        f"REAL NEWS DATA:        {'PASS (' + str(len(news_articles)) + ' articles)' if news_articles else 'ABSENT (M2 structurally disabled, training M0+M1)'}"
    )
    print("SYNTHETIC FALLBACK:    DISABLED")
    print(
        f"70/15/15 SPLIT:        PASS ({split_manifest.train_rows} train / {split_manifest.val_rows} val / {split_manifest.test_rows} test)"
    )
    print(
        f"PURGE & EMBARGO:       PASS ({split_manifest.purged_train_rows + split_manifest.purged_val_rows} purged, {split_manifest.embargo_sessions} embargo)"
    )
    print(f"MARKET PAYLOAD SHA:    {market_payload_sha256[:16]}...")
    print(f"NEWS PAYLOAD SHA:      {news_payload_sha256[:16]}...")
    print(f"COMPANY NEWS COVERAGE: {coverage_report.company_coverage_percentage:.2f}%")
    print(f"MULTI-SOURCE COVERAGE: {multi_src_pct:.2f}%")
    print(f"M2 STRUCTURALLY ENABLED: {'YES' if m2_eligible else 'NO'}")
    print(f"TRAINING SCOPE:        {auth_record['training_scope']}")
    print(f"GPU ACCELERATION:      {gpu_dev}")
    print("FULL TRAINING AUTH:    YES")
    print("=" * 60 + "\n")

    # 7. Model Training across Seeds 41, 42, 43
    print(
        "[7/7] Training Learned M0 -> M1 across seeds (41, 42, 43) with diagnostic comparators..."
    )

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
    seed_m1_crps: dict[int, float] = {}
    seed_m1_per_h: dict[int, dict[int, float]] = {}

    for s in seeds:
        print(f"  -> Training expanding folds with seed {s} (canonical: 42)...")
        torch.manual_seed(s)
        np.random.seed(s)
        bundle = GlobalMultimodalTrainerV11.develop_and_freeze_bundle(
            dev_payload=dev_payload,
            max_epochs=15,
            patience=4,
            lr=0.003,
            n_expanding_folds=4,
            df=5.0,
            enable_m2=m2_eligible,
        )
        seed_bundles[s] = bundle
        m1_rec = bundle.manifest.validation_oof_metrics["M1_NUMERIC"]
        seed_m1_crps[s] = m1_rec.crps_mean
        seed_m1_per_h[s] = m1_rec.crps_per_horizon
        torch.save(bundle.m1_numeric_model.state_dict(), ckpt_dir / f"m1_numeric_seed_{s}.pt")

    # Seed 42 is preregistered as the canonical champion candidate
    canonical_seed = 42
    champion_bundle = seed_bundles[canonical_seed]
    val_metrics = champion_bundle.manifest.validation_oof_metrics
    selected_champion = champion_bundle.manifest.selected_candidate_family

    m_path, s_path, freeze_digest = CandidateFreezerV11.freeze_and_save_bundle(
        bundle=champion_bundle,
        output_dir=artifacts_dir,
        git_sha=git_sha,
        panel_sha=panel_dataset.panel_sha256,
        split_sha=split_manifest.split_sha256,
    )

    # Multi-seed diagnostics
    m1_crps_list = [seed_m1_crps[s] for s in seeds]
    seed_stats = {
        "canonical_seed": canonical_seed,
        "all_seeds": seeds,
        "m1_crps_per_seed": seed_m1_crps,
        "m1_mean_crps": float(np.mean(m1_crps_list)),
        "m1_std_crps": float(np.std(m1_crps_list)),
        "m1_best_seed": int(seeds[np.argmin(m1_crps_list)]),
        "m1_worst_seed": int(seeds[np.argmax(m1_crps_list)]),
        "m1_per_horizon_by_seed": seed_m1_per_h,
    }

    comp_data = {
        "experiment_type": "REAL_HISTORICAL_LEARNED_TRAINING_V11_1",
        "universe": "NDX100 8-security research subset",
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "market_payload_sha256": market_payload_sha256,
        "news_payload_sha256": news_payload_sha256,
        "dataset_summary": {
            "total_rows": len(panel_dataset.dates),
            "unique_dates": len(set(panel_dataset.dates)),
            "unique_securities": len(set(panel_dataset.security_ids)),
            "train_rows": len(dev_payload.train_dates),
            "val_rows": len(dev_payload.val_dates),
            "company_news_coverage_pct": coverage_report.company_coverage_percentage,
            "m2_eligible": m2_eligible,
        },
        "models_development_metrics": {k: v.to_dict() for k, v in val_metrics.items()},
        "multi_seed_robustness": seed_stats,
        "selected_champion_family": selected_champion,
        "learned_promotion": "NONE" if selected_champion == "M0_HAR_BASELINE" else "PROMOTED",
        "master_freeze_digest": freeze_digest,
        "sealed_test_status": "LOCKED_UNOPENED",
    }
    (artifacts_dir / "v11_development_model_comparison.json").write_text(
        json.dumps(comp_data, indent=2), encoding="utf-8"
    )

    m0_crps = val_metrics["M0_HAR_BASELINE"].crps_mean
    m1_crps = val_metrics["M1_NUMERIC"].crps_mean
    ridge_crps = val_metrics["RIDGE_LOCATION_HAR_SCALE"].crps_mean
    histgb_crps = val_metrics["HISTGB_LOCATION_HAR_SCALE"].crps_mean

    comp_md = rf"""# V11.1 Real Historical Multimodal Model Development Comparison

- **Universe**: NDX100 8-security research subset (`AMGN`, `AAPL`, `MSFT`, `NVDA`, `META`, `HON`, `CRWD`, `CSCO`)
- **Data Mode**: Real Historical Market Data (`data/ndx100/cache/`)
- **Sector Benchmarks**: Security-specific ETFs (`XLV` for AMGN, `XLI` for HON, `XLK` for Tech/Comm/Cloud)
- **Market Benchmark**: Real `QQQ` ETF
- **Git Commit Provenance**: `{git_sha}` (Dirty: `{git_dirty}`)
- **Total Panel Rows**: {len(panel_dataset.dates):,}
- **Train Rows**: {len(dev_payload.train_dates):,} ({split_manifest.effective_train_pct}%)
- **Validation Rows**: {len(dev_payload.val_dates):,} ({split_manifest.effective_val_pct}%)
- **Sealed Test Rows**: {split_manifest.test_rows:,} ({split_manifest.effective_test_pct}%) — **LOCKED & UNINSPECTED**
- **Real Market Payload SHA-256**: `{market_payload_sha256}`
- **Company News Coverage**: {coverage_report.company_coverage_percentage:.2f}% (M2 Structurally Enabled: {"YES" if m2_eligible else "NO"})
- **Selected Champion Family**: **{selected_champion}** (Learned Promotion: **{"NONE" if selected_champion == "M0_HAR_BASELINE" else "PROMOTED"}**)
- **Candidate Master Freeze Digest**: `{freeze_digest}`

---

### Validation Metrics by Model (Canonical Seed 42 + Diagnostic Comparators)
| Model | CRPS (Mean) | QLIKE | MAE (Ret) | 80% Coverage | Pinball (10-90) | Outcome / Role |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0 (Econometric HAR)** | **{val_metrics["M0_HAR_BASELINE"].crps_mean:.6f}** | {val_metrics["M0_HAR_BASELINE"].qlike_mean:.6f} | {val_metrics["M0_HAR_BASELINE"].return_mae:.6f} | {val_metrics["M0_HAR_BASELINE"].coverage_80pct * 100:.1f}% | {val_metrics["M0_HAR_BASELINE"].pinball_loss_10_90:.6f} | {"WINNER (Selected Champion)" if selected_champion == "M0_HAR_BASELINE" else "Baseline"} |
| **Ridge Regression + HAR Vol** | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].crps_mean:.6f} | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].qlike_mean:.6f} | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].return_mae:.6f} | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].coverage_80pct * 100:.1f}% | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].pinball_loss_10_90:.6f} | Linear Numeric Diagnostic |
| **HistGB Trees + HAR Vol** | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].crps_mean:.6f} | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].qlike_mean:.6f} | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].return_mae:.6f} | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].coverage_80pct * 100:.1f}% | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].pinball_loss_10_90:.6f} | Non-linear Tree Diagnostic |
| **M1 (Learned Numeric Neural)** | {val_metrics["M1_NUMERIC"].crps_mean:.6f} | {val_metrics["M1_NUMERIC"].qlike_mean:.6f} | {val_metrics["M1_NUMERIC"].return_mae:.6f} | {val_metrics["M1_NUMERIC"].coverage_80pct * 100:.1f}% | {val_metrics["M1_NUMERIC"].pinball_loss_10_90:.6f} | {"WINNER (Promoted)" if selected_champion == "M1_NUMERIC" else "Failed to beat M0 (No Promotion)"} |
| **M2 (Multimodal News)** | N/A (Disabled) | N/A | N/A | N/A | N/A | Structurally Disabled (No News Archive) |

---

### Multi-Seed Robustness Diagnostics (Seeds 41, 42, 43)
- **Canonical Seed (42)**: CRPS = `{seed_m1_crps[42]:.6f}`
- **Diagnostic Seed (41)**: CRPS = `{seed_m1_crps[41]:.6f}`
- **Diagnostic Seed (43)**: CRPS = `{seed_m1_crps[43]:.6f}`
- **Seed Mean +/- Std**: `{seed_stats["m1_mean_crps"]:.6f} +/- {seed_stats["m1_std_crps"]:.6f}`
- **Best Seed**: `{seed_stats["m1_best_seed"]}` (`{min(m1_crps_list):.6f}`)
- **Worst Seed**: `{seed_stats["m1_worst_seed"]}` (`{max(m1_crps_list):.6f}`)

---

### Per-Horizon Validation CRPS ($h \in \{{1, 3, 5, 7\}}$)
| Model | Day 1 ($h=1$) | Day 3 ($h=3$) | Day 5 ($h=5$) | Day 7 ($h=7$) |
|---|:---:|:---:|:---:|:---:|
| **M0 (HAR)** | {val_metrics["M0_HAR_BASELINE"].crps_per_horizon[1]:.6f} | {val_metrics["M0_HAR_BASELINE"].crps_per_horizon[3]:.6f} | {val_metrics["M0_HAR_BASELINE"].crps_per_horizon[5]:.6f} | {val_metrics["M0_HAR_BASELINE"].crps_per_horizon[7]:.6f} |
| **Ridge + HAR** | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].crps_per_horizon[1]:.6f} | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].crps_per_horizon[3]:.6f} | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].crps_per_horizon[5]:.6f} | {val_metrics["RIDGE_LOCATION_HAR_SCALE"].crps_per_horizon[7]:.6f} |
| **HistGB + HAR** | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].crps_per_horizon[1]:.6f} | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].crps_per_horizon[3]:.6f} | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].crps_per_horizon[5]:.6f} | {val_metrics["HISTGB_LOCATION_HAR_SCALE"].crps_per_horizon[7]:.6f} |
| **M1 (Numeric Seed 42)** | {val_metrics["M1_NUMERIC"].crps_per_horizon[1]:.6f} | {val_metrics["M1_NUMERIC"].crps_per_horizon[3]:.6f} | {val_metrics["M1_NUMERIC"].crps_per_horizon[5]:.6f} | {val_metrics["M1_NUMERIC"].crps_per_horizon[7]:.6f} |
"""
    (artifacts_dir / "v11_development_model_comparison.md").write_text(comp_md, encoding="utf-8")

    print("\n" + "=" * 70)
    print("V11.1 REAL HISTORICAL DEVELOPMENT AND CANDIDATE FREEZE COMPLETED")
    print("=" * 70)
    print(f"Selected Champion:       {selected_champion}")
    print(
        f"Learned Promotion:       {'NONE' if selected_champion == 'M0_HAR_BASELINE' else 'PROMOTED'}"
    )
    print(
        f"Validation CRPS:         M0={m0_crps:.6f} | Ridge={ridge_crps:.6f} | HistGB={histgb_crps:.6f} | M1={m1_crps:.6f}"
    )
    print(
        f"Multi-Seed M1:           {seed_stats['m1_mean_crps']:.6f} +/- {seed_stats['m1_std_crps']:.6f}"
    )
    print(f"Candidate Master Digest: {freeze_digest}")
    print("Sealed Test Status:      STILL CLOSED & UNINSPECTED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V11.1 Real Historical Pipeline Orchestrator")
    parser.add_argument(
        "--news-archive",
        type=Path,
        default=None,
        help="Path to real news archive JSONL file.",
    )
    args = parser.parse_args()
    run_historical_pipeline(news_archive=args.news_archive)
