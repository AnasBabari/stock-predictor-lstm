"""Unit tests for Schema V11 contract."""

from backend.contracts.schemas_v11 import (
    MULTIMODAL_NEWS_FEATURE_COLUMNS_V11,
    MULTIMODAL_NUMERIC_FEATURE_COLUMNS_V11,
    MULTIMODAL_TOTAL_FEATURE_COLUMNS_V11,
    REQUIRED_TARGET_HORIZONS_V11,
    get_schema_v11_manifest,
)


def test_schema_v11_counts_and_integrity():
    assert len(MULTIMODAL_NUMERIC_FEATURE_COLUMNS_V11) == 34
    assert len(MULTIMODAL_NEWS_FEATURE_COLUMNS_V11) == 19
    assert len(MULTIMODAL_TOTAL_FEATURE_COLUMNS_V11) == 53
    assert REQUIRED_TARGET_HORIZONS_V11 == (1, 3, 5, 7)

    manifest = get_schema_v11_manifest()
    assert manifest.numeric_count == 34
    assert manifest.news_count == 19
    assert manifest.total_count == 53
    assert len(manifest.schema_sha256) == 64
