"""Repository-truth guards for the v8 feature schema and its documentation.

These tests exist because the repository previously drifted: a module docstring
claimed the numeric fallback was "certified today" while no v8 candidate had
been certified, and the news feature count was described inconsistently across
documents.  The guards below make the declared schema authoritative and fail
closed if documentation or constants drift away from the source of truth.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5
from research.volatility_forecasting.contracts import VolatilityForecastProtocol
from research.volatility_forecasting.features_v8 import (
    NEWS_V1_FEATURES,
    V8_NEWS_ONLY_FEATURE_COUNT,
    V8_NUMERIC_FEATURE_COUNT,
    V8_TOTAL_FEATURE_COUNT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_declared_feature_counts() -> None:
    """The frozen schema is 26 numeric + 22 news/macro = 48 total."""
    assert V8_NUMERIC_FEATURE_COUNT == 26
    assert V8_NEWS_ONLY_FEATURE_COUNT == 22
    assert V8_TOTAL_FEATURE_COUNT == 48


def test_counts_match_source_of_truth_tuples() -> None:
    """Constants must be derived from the tuples, never hardcoded apart."""
    numeric = len(DEPLOYABLE_FEATURE_COLUMNS_V5)
    news = len(NEWS_V1_FEATURES)
    assert numeric == V8_NUMERIC_FEATURE_COUNT
    assert news == V8_NEWS_ONLY_FEATURE_COUNT
    assert numeric + news == V8_TOTAL_FEATURE_COUNT


def test_news_tuple_has_no_accidental_duplicates() -> None:
    assert len(set(NEWS_V1_FEATURES)) == len(NEWS_V1_FEATURES)
    assert len(set(DEPLOYABLE_FEATURE_COLUMNS_V5)) == len(DEPLOYABLE_FEATURE_COLUMNS_V5)


def test_numeric_and_news_schemas_are_disjoint() -> None:
    assert not set(DEPLOYABLE_FEATURE_COLUMNS_V5).intersection(NEWS_V1_FEATURES)


def test_runtime_protocol_uses_numeric_schema() -> None:
    """The live numeric protocol must consume exactly the 26 numeric features."""
    protocol = VolatilityForecastProtocol()
    assert protocol.feature_count == V8_NUMERIC_FEATURE_COUNT
    assert protocol.feature_names == DEPLOYABLE_FEATURE_COLUMNS_V5


_PRUNED_DIRECTORIES = frozenset(
    {".venv", "venv", "node_modules", "__pycache__", ".git", "dist", ".mypy_cache", ".pytest_cache"}
)
_SCANNED_SUFFIXES = frozenset({".py", ".md", ".yml", ".json", ".toml"})


def _tracked_source_files() -> list[Path]:
    """Walk only the maintained trees, pruning vendored directories at the source.

    Pruning (rather than filtering after the fact) keeps this guard fast even
    though ``backend/.venv`` contains tens of thousands of files.
    """
    found: list[Path] = []
    for root in ("backend", "research", "scripts", "docs", "configs"):
        base = REPO_ROOT / root
        if not base.exists():
            continue
        pending = [base]
        while pending:
            current = pending.pop()
            for child in current.iterdir():
                if child.is_dir():
                    if child.name not in _PRUNED_DIRECTORIES:
                        pending.append(child)
                elif child.suffix in _SCANNED_SUFFIXES:
                    found.append(child)
    return found


def test_no_stale_feature_count_claims_in_tracked_sources() -> None:
    """No source file may still assert the historic 19-news / 45-feature counts."""
    stale = re.compile(r"19\s+news|45\s+features|45\s+max|19\s*\+\s*26", re.IGNORECASE)
    offenders: list[str] = []
    for path in _tracked_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if stale.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{number}")
    assert not offenders, f"stale feature-count claims found: {offenders}"


def test_features_v8_does_not_claim_certification() -> None:
    """The schema module must not assert a certification that has not happened."""
    source = (REPO_ROOT / "research" / "volatility_forecasting" / "features_v8.py").read_text(
        encoding="utf-8"
    )
    assert "certified today" not in source.lower()
    # The correction must positively state the actual repository truth.
    assert "no certified v8 candidate" in source
    assert "no opened v8 sealed test" in source


@pytest.mark.parametrize("count", [19, 45])
def test_numeric_feature_count_is_not_a_stale_value(count: int) -> None:
    """Guard against a silent schema regression to historic counts."""
    assert count != V8_NUMERIC_FEATURE_COUNT
    assert count != V8_TOTAL_FEATURE_COUNT
    assert count != V8_NEWS_ONLY_FEATURE_COUNT
