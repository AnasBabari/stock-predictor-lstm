from datetime import UTC, datetime

import pandas as pd

from news_features import merge_historical_news_features


def test_news_features_merge_only_for_offline_feature_frames():
    sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
    frame = pd.DataFrame({"Close": [100.0, 101.0]}, index=sessions)
    articles = [
        {
            "title": "Earnings beat expectations",
            "summary": "",
            "published_at": datetime(2024, 1, 2, 12, tzinfo=UTC),
            "publisher": None,
            "link": None,
        }
    ]

    merged = merge_historical_news_features(frame, articles)

    assert "News_Sentiment" in merged
    assert "News_Sentiment" not in frame
    assert merged.loc[sessions[0], "News_Article_Count"] == 0
