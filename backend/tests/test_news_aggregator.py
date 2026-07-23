from unittest.mock import patch, MagicMock
from news_aggregator import get_financial_sentiment

@patch('news_aggregator.yf.Ticker')
def test_positive_headlines(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.news = [
        {"title": "Company XYZ beat expectations and had an upgrade."},
        {"title": "Earnings soared and showed a record high."}
    ]
    mock_ticker.return_value = mock_instance
    
    result = get_financial_sentiment('AAPL')
    
    assert "sentiment" in result
    assert result["sentiment"] > 0.0
    assert result["sentiment_source"] == "yfinance_vader"

@patch('news_aggregator.yf.Ticker')
def test_negative_headlines(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.news = [
        {"title": "Company XYZ issued a profit warning and faces a lawsuit."},
        {"title": "Guidance cut leads to selloff and bankruptcy fears."}
    ]
    mock_ticker.return_value = mock_instance
    
    result = get_financial_sentiment('AAPL')
    
    assert "sentiment" in result
    assert result["sentiment"] < 0.0
    assert result["sentiment_source"] == "yfinance_vader"

@patch('news_aggregator.yf.Ticker')
def test_fallback_on_no_news(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.news = []
    mock_ticker.return_value = mock_instance
    
    result = get_financial_sentiment('AAPL')
    
    assert result["sentiment"] == 0.0
    assert result["sentiment_source"] == "fallback"

@patch('news_aggregator.yf.Ticker')
def test_fallback_on_exception(mock_ticker):
    mock_ticker.side_effect = Exception("API down")
    
    result = get_financial_sentiment('AAPL')
    
    assert result["sentiment"] == 0.0
    assert result["sentiment_source"] == "fallback"
