from experiments.ablation import NEWS_FEATURES, feature_ablation_sets


def test_news_ablation_is_available_only_when_timestamped_features_exist():
    without_news = feature_ablation_sets()
    with_news = feature_ablation_sets(include_news=True)

    assert "ohlcv_technical_market_news" not in without_news
    assert with_news["ohlcv_technical_market_news"][-3:] == NEWS_FEATURES
