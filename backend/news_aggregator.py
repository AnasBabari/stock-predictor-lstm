import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

logger = logging.getLogger(__name__)

def get_financial_sentiment(ticker: str) -> dict:
    try:
        try:
            analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            analyzer = SentimentIntensityAnalyzer()
            
        custom_dict = {
            "guidance cut": -2.0, 
            "missed estimates": -1.5, 
            "beat expectations": 2.0, 
            "upgrade": 1.5, 
            "downgrade": -1.5,
            "bullish": 2.0,
            "bearish": -2.0,
            "outperform": 1.5,
            "underperform": -1.5,
            "buy": 1.0,
            "sell": -1.0,
            "strong buy": 2.0,
            "strong sell": -2.0,
            "profit warning": -2.0,
            "dividend hike": 1.5,
            "dividend cut": -1.5,
            "layoffs": -1.5,
            "restructuring": -0.5,
            "bankruptcy": -3.0,
            "lawsuit": -1.5,
            "investigation": -1.5,
            "record high": 1.5,
            "record low": -1.5,
            "surged": 1.5,
            "plunged": -2.0,
            "selloff": -2.0,
            "rally": 1.5,
            "merger": 1.0,
            "acquisition": 1.0,
            "buyout": 1.5,
            "slumps": -1.5,
            "soars": 1.5
        }
        
        analyzer.lexicon.update(custom_dict)
        
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        
        if not news:
            return {"sentiment": 0.0, "sentiment_source": "fallback"}
            
        total_score = 0.0
        count = 0
        
        for article in news:
            title = article.get("title")
            if title:
                score = analyzer.polarity_scores(title)
                total_score += score["compound"]
                count += 1
                
        if count == 0:
            return {"sentiment": 0.0, "sentiment_source": "fallback"}
            
        average_score = total_score / count
        return {"sentiment": average_score, "sentiment_source": "yfinance_vader"}
        
    except Exception as e:
        logger.exception("Error fetching news sentiment for %s", ticker)
        return {"sentiment": 0.0, "sentiment_source": "fallback"}
