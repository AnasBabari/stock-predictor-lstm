"""Build a causal, content-addressed v8 news matrix for market examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import VolatilityPanelExamples
from .news import NewsFeatureMatrix
from .news_alignment import (
    aggregate_news_for_market_rows,
    v8_calendar_map,
    validate_news_coverage,
)
from .news_exposures import load_news_exposure_map
from .news_snapshot import load_news_snapshot
from .news_snapshot_v8 import verify_v8_news_manifest
from .universe_v8 import universe_identity_maps, verify_universe_manifest


@dataclass(frozen=True)
class V8AlignedNewsMatrix:
    values: np.ndarray
    feature_names: tuple[str, ...]
    cutoffs: np.ndarray
    snapshot_sha256: str
    matrix_sha256: str
    exposure_sha256: str


def _matrix_digest(
    matrix: NewsFeatureMatrix,
    *,
    snapshot_sha256: str,
    exposure_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(snapshot_sha256.encode("ascii"))
    digest.update(exposure_sha256.encode("ascii"))
    digest.update(json.dumps(matrix.feature_names, separators=(",", ":")).encode("utf-8"))
    for ticker, cutoff in zip(matrix.tickers, matrix.cutoffs, strict=True):
        digest.update(str(ticker).strip().upper().encode("utf-8"))
        digest.update(b"|")
        digest.update(str(np.datetime64(cutoff, "ns")).encode("ascii"))
        digest.update(b"\n")
    digest.update(np.asarray(matrix.values, dtype="<f4", order="C").tobytes())
    return "sha256:" + digest.hexdigest()


def build_v8_aligned_news_matrix(
    examples: VolatilityPanelExamples,
    *,
    news_snapshot_dir: Path,
    news_manifest: dict[str, Any],
    universe_manifest: dict[str, Any],
    market_manifest: dict[str, Any],
    ticker_aliases_path: Path,
    exposure_map_path: Path,
) -> V8AlignedNewsMatrix:
    """Verify provenance and aggregate only events available at each exchange close."""

    universe = verify_universe_manifest(universe_manifest)
    verified_news = verify_v8_news_manifest(
        news_manifest,
        news_snapshot_dir=news_snapshot_dir,
        universe_manifest=universe,
        market_manifest=market_manifest,
        ticker_aliases_path=ticker_aliases_path,
    )
    if verified_news.get("coverage_complete") is not True:
        raise ValueError("v8 news matrix requires complete provider coverage")
    events, _base_manifest = load_news_snapshot(news_snapshot_dir)
    exchange_map, _security_map = universe_identity_maps(universe)
    panel_tickers = {str(ticker).strip().upper() for ticker in examples.tickers}
    if panel_tickers != set(exchange_map):
        raise ValueError("news matrix panel assets must exactly match the v8 universe")
    exposures = load_news_exposure_map(
        exposure_map_path,
        required_tickers=panel_tickers,
    )
    matrix = aggregate_news_for_market_rows(
        events,
        examples.tickers,
        examples.origin_dates,
        exposure_map=exposures.exposures,
        calendar_by_ticker=v8_calendar_map(exchange_map),
    )
    expected_tickers = np.char.upper(np.asarray(examples.tickers, dtype=str))
    if not np.array_equal(matrix.tickers, expected_tickers):
        raise RuntimeError("news aggregation changed example row order")
    validate_news_coverage(
        verified_news,
        matrix.cutoffs,
        lookback_days=int(verified_news["feature_lookback_days"]),
    )
    snapshot_sha = "sha256:" + str(verified_news["sha256"])
    return V8AlignedNewsMatrix(
        values=np.asarray(matrix.values, dtype=np.float32),
        feature_names=matrix.feature_names,
        cutoffs=np.asarray(matrix.cutoffs, dtype="datetime64[ns]"),
        snapshot_sha256=snapshot_sha,
        matrix_sha256=_matrix_digest(
            matrix,
            snapshot_sha256=snapshot_sha,
            exposure_sha256=exposures.source_sha256,
        ),
        exposure_sha256="sha256:" + exposures.source_sha256,
    )
