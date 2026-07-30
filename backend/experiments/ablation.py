"""Controlled feature-group ablations using a shared experiment configuration."""

from __future__ import annotations

import pandas as pd

from config import FEATURE_CONFIG
from experiments.runner import ExperimentConfig, run_baseline_experiment

NEWS_FEATURES = ["News_Sentiment", "News_Article_Count", "News_Sentiment_Confidence"]


def feature_ablation_sets(*, include_news: bool = False) -> dict[str, list[str]]:
    base = list(FEATURE_CONFIG["base"])
    technical = list(FEATURE_CONFIG["technical"])
    market = list(FEATURE_CONFIG["market"])
    calendar = list(FEATURE_CONFIG["calendar"])
    sets = {
        "price": ["Close"],
        "ohlcv": base,
        "ohlcv_technical": base + technical,
        "ohlcv_market": base + market,
        "ohlcv_technical_market": base + technical + market,
        "all_market_features": base + technical + market + calendar,
    }
    if include_news:
        sets["ohlcv_technical_market_news"] = sets["ohlcv_technical_market"] + NEWS_FEATURES
    return sets


def run_feature_ablation(
    feature_frame: pd.DataFrame,
    *,
    feature_sets: tuple[str, ...] | None = None,
    config: ExperimentConfig | None = None,
) -> dict:
    """Evaluate requested feature sets without changing folds or model settings."""

    available = feature_ablation_sets(
        include_news=set(NEWS_FEATURES).issubset(feature_frame.columns)
    )
    selected = tuple(available) if feature_sets is None else feature_sets
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")

    reports = {}
    close_values = feature_frame["Close"].to_numpy(dtype=float)
    for name in selected:
        columns = available[name]
        missing = [column for column in columns if column not in feature_frame.columns]
        if missing:
            raise ValueError(f"Feature set {name} is missing columns: {missing}")
        reports[name] = run_baseline_experiment(
            feature_frame[columns].to_numpy(dtype=float),
            close_values,
            feature_names=columns,
            config=config,
        )
    return {
        "feature_sets": list(selected),
        "reports": reports,
    }
