"""Sentiment analysis utilities for stock predictions."""
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List, Tuple

class SentimentAnalyzer:
    """Analyze sentiment from cached news data."""

    def __init__(self):
        self.db_path = Path('data/stock_cache.db')

    def get_stock_sentiment(self, symbol: str, days: int = 7) -> Optional[Dict]:
        """
        Get sentiment data for a specific stock.

        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back

        Returns:
            Dictionary with sentiment metrics or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    AVG(sentiment_score) as avg_score,
                    COUNT(*) as article_count,
                    SUM(CASE WHEN sentiment_label = 'Positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN sentiment_label = 'Negative' THEN 1 ELSE 0 END) as negative,
                    MAX(publish_time) as latest_news
                FROM news_articles
                WHERE symbol = ?
                AND publish_time >= datetime('now', '-' || ? || ' days')
            """, (symbol, days))

            result = cursor.fetchone()
            conn.close()

            if result and result[0] is not None:
                avg_score, count, positive, negative, latest = result
                return {
                    'sentiment_score': avg_score,
                    'article_count': count,
                    'positive_count': positive,
                    'negative_count': negative,
                    'latest_news_time': latest,
                    'sentiment_label': self._get_label(avg_score)
                }

            return None

        except Exception as e:
            print(f"Error getting sentiment for {symbol}: {e}")
            return None

    def get_market_sentiment(self, days: int = 7) -> Dict:
        """
        Get overall market sentiment.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with market sentiment metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(DISTINCT symbol) as stock_count,
                    COUNT(*) as article_count
                FROM news_articles
                WHERE publish_time >= datetime('now', '-' || ? || ' days')
            """, (days,))

            result = cursor.fetchone()
            conn.close()

            if result and result[0] is not None:
                avg_sentiment, stock_count, article_count = result
                return {
                    'sentiment_score': avg_sentiment,
                    'stock_count': stock_count,
                    'article_count': article_count,
                    'sentiment_label': self._get_label(avg_sentiment)
                }

            return {
                'sentiment_score': 0.0,
                'stock_count': 0,
                'article_count': 0,
                'sentiment_label': 'Neutral'
            }

        except Exception as e:
            print(f"Error getting market sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'stock_count': 0,
                'article_count': 0,
                'sentiment_label': 'Neutral'
            }

    def get_bulk_sentiment(self, symbols: List[str], days: int = 7) -> Dict[str, Dict]:
        """
        Get sentiment for multiple stocks at once.

        Args:
            symbols: List of stock ticker symbols
            days: Number of days to look back

        Returns:
            Dictionary mapping symbols to sentiment data
        """
        results = {}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Use parameterized query for safety
            placeholders = ','.join('?' * len(symbols))
            query = f"""
                SELECT
                    symbol,
                    AVG(sentiment_score) as avg_score,
                    COUNT(*) as article_count,
                    SUM(CASE WHEN sentiment_label = 'Positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN sentiment_label = 'Negative' THEN 1 ELSE 0 END) as negative
                FROM news_articles
                WHERE symbol IN ({placeholders})
                AND publish_time >= datetime('now', '-' || ? || ' days')
                GROUP BY symbol
            """

            cursor.execute(query, symbols + [days])

            for row in cursor.fetchall():
                symbol, avg_score, count, positive, negative = row
                if avg_score is not None:
                    results[symbol] = {
                        'sentiment_score': avg_score,
                        'article_count': count,
                        'positive_count': positive,
                        'negative_count': negative,
                        'sentiment_label': self._get_label(avg_score)
                    }

            conn.close()

        except Exception as e:
            print(f"Error getting bulk sentiment: {e}")

        return results

    def _get_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.2:
            return "Positive"
        elif score < -0.2:
            return "Negative"
        else:
            return "Neutral"

    def get_sentiment_color(self, score: float) -> str:
        """Get color for sentiment visualization."""
        if score > 0.2:
            return "green"
        elif score < -0.2:
            return "red"
        else:
            return "gray"
