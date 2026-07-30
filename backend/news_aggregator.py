"""Deprecated compatibility adapter; canonical implementation is news_features."""

import yfinance as yf

from news_features import deduplicate_articles, normalise_news_articles, score_news_articles


def get_financial_sentiment(ticker: str) -> dict:
    try:
        articles = score_news_articles(
            deduplicate_articles(normalise_news_articles(yf.Ticker(ticker).news))
        )
        if articles:
            return {
                "sentiment": {
                    "score": round(sum(a["sentiment_score"] for a in articles) / len(articles), 4),
                    "status": "live",
                    "provider": "yfinance",
                    "method": "vader_financial",
                }
            }
    except Exception:
        pass
    return {
        "sentiment": {
            "score": 0.0,
            "status": "fallback",
            "provider": "yfinance",
            "method": "vader_financial",
        }
    }
