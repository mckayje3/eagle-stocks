"""
Smart caching manager that uses SQLite database with Yahoo Finance fallback.

Caching Strategy:
- Price data: Cache for 24 hours (daily), 1 hour (intraday)
- Company info: Cache for 24 hours
- News: Cache for 1 hour
- Automatic fallback to Yahoo Finance when cache is stale or missing
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from utils.database import StockDatabase
from textblob import TextBlob
import config

class CachedDataFetcher:
    """Data fetcher with intelligent SQLite caching."""

    def __init__(self, db_path: str = "data/stock_cache.db",
                 price_cache_hours: int = 24,
                 info_cache_hours: int = 24,
                 news_cache_hours: int = 1):
        """
        Initialize cached data fetcher.

        Args:
            db_path: Path to SQLite database
            price_cache_hours: Hours to cache price data
            info_cache_hours: Hours to cache company info
            news_cache_hours: Hours to cache news
        """
        self.db = StockDatabase(db_path)
        self.price_cache_hours = price_cache_hours
        self.info_cache_hours = info_cache_hours
        self.news_cache_hours = news_cache_hours

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0

    def get_stock_data(self, symbol: str, period: str = "1mo", interval: str = "1d",
                      force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical stock data with caching.

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1d, 1h, 5m, etc.)
            force_refresh: Force fetch from API, bypass cache

        Returns:
            DataFrame with OHLCV data or None
        """
        symbol = symbol.upper()

        # Determine cache freshness requirement based on interval
        if interval in ['1m', '5m', '15m', '30m', '1h']:
            cache_hours = 1  # Intraday data cached for 1 hour
        else:
            cache_hours = self.price_cache_hours

        # Check cache first (unless force refresh)
        if not force_refresh:
            # Calculate date range from period
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = self._period_to_start_date(period)

            if self.db.is_price_data_fresh(symbol, interval, cache_hours):
                cached_data = self.db.get_price_data(symbol, start_date, end_date, interval)

                if cached_data is not None and not cached_data.empty:
                    # Verify we have enough data
                    expected_days = (datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')).days
                    actual_days = len(cached_data)

                    # If we have at least 80% of expected data, use cache
                    if actual_days >= expected_days * 0.8:
                        self.cache_hits += 1
                        print(f"[CACHE] Hit for {symbol} ({period}) - {len(cached_data)} records")
                        return cached_data

        # Cache miss or stale - fetch from Yahoo Finance
        self.cache_misses += 1
        self.api_calls += 1
        print(f"[API] Fetching {symbol} from Yahoo Finance...")

        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)

            if data.empty:
                print(f"[WARN] No data returned for {symbol}")
                return None

            # Store in cache
            self.db.insert_price_data(symbol, data, interval)
            print(f"[OK] Cached {len(data)} records for {symbol}")

            return data

        except Exception as e:
            print(f"[ERROR] Error fetching {symbol}: {str(e)}")
            # Try to return stale cache data as fallback
            cached_data = self.db.get_price_data(symbol, interval=interval)
            if cached_data is not None:
                print(f"[FALLBACK] Returning stale cache data for {symbol}")
                return cached_data
            return None

    def get_stock_info(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get stock information with caching.

        Args:
            symbol: Stock ticker symbol
            force_refresh: Force fetch from API

        Returns:
            Dictionary with stock information or None
        """
        symbol = symbol.upper()

        # Check cache first
        if not force_refresh and self.db.is_info_fresh(symbol, self.info_cache_hours):
            cached_info = self.db.get_stock_info(symbol)
            if cached_info:
                self.cache_hits += 1
                print(f"[CACHE] Hit for {symbol} info")
                return cached_info

        # Fetch from Yahoo Finance
        self.cache_misses += 1
        self.api_calls += 1
        print(f"[API] Fetching {symbol} info from Yahoo Finance...")

        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            if not info or 'symbol' not in info:
                print(f"[WARN] No info returned for {symbol}")
                return None

            # Process and standardize info
            processed_info = {
                'symbol': info.get('symbol', symbol),
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_low': info.get('dayLow', 0),
                'day_high': info.get('dayHigh', 0),
                'year_low': info.get('fiftyTwoWeekLow', 0),
                'year_high': info.get('fiftyTwoWeekHigh', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'eps': info.get('trailingEps', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
            }

            # Store in cache
            self.db.insert_stock_info(symbol, processed_info)
            print(f"[OK] Cached info for {symbol}")

            return processed_info

        except Exception as e:
            print(f"[ERROR] Error fetching info for {symbol}: {str(e)}")
            # Try to return stale cache as fallback
            cached_info = self.db.get_stock_info(symbol)
            if cached_info:
                print(f"[FALLBACK] Returning stale cache for {symbol} info")
                return cached_info
            return None

    def get_current_price(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """
        Get current/latest stock price with caching.

        Args:
            symbol: Stock ticker symbol
            force_refresh: Force fetch from API

        Returns:
            Current price or None
        """
        # For current price, check if info is fresh (within cache hours)
        if not force_refresh and self.db.is_info_fresh(symbol, self.info_cache_hours):
            info = self.db.get_stock_info(symbol)
            if info and info.get('current_price'):
                self.cache_hits += 1
                return float(info['current_price'])

        # Fetch fresh info
        info = self.get_stock_info(symbol, force_refresh=True)
        if info and info.get('current_price'):
            return float(info['current_price'])

        # Fallback to latest price from historical data
        data = self.get_stock_data(symbol, period="1d", force_refresh=force_refresh)
        if data is not None and not data.empty:
            return float(data['Close'].iloc[-1])

        return None

    def get_news(self, symbol: str, max_articles: int = 20,
                force_refresh: bool = False) -> List[Dict]:
        """
        Get news articles with caching and sentiment analysis.

        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles
            force_refresh: Force fetch from API

        Returns:
            List of news articles with sentiment
        """
        symbol = symbol.upper()

        # Check cache first
        if not force_refresh and self.db.is_news_fresh(symbol, self.news_cache_hours):
            cached_news = self.db.get_news_articles(symbol, limit=max_articles)
            if cached_news:
                self.cache_hits += 1
                print(f"[CACHE] Hit for {symbol} news - {len(cached_news)} articles")
                return cached_news

        # Fetch from Yahoo Finance
        self.cache_misses += 1
        self.api_calls += 1
        print(f"[API] Fetching news for {symbol} from Yahoo Finance...")

        try:
            stock = yf.Ticker(symbol)
            news = stock.news

            if not news:
                print(f"[WARN] No news available for {symbol}")
                return []

            # Process news and add sentiment
            processed_articles = []
            for article in news[:max_articles]:
                # Handle new yfinance structure where data is nested in 'content'
                if 'content' in article:
                    article_data = article['content']
                else:
                    article_data = article

                title = article_data.get('title', 'No title')

                # Perform sentiment analysis
                sentiment_score = self._analyze_sentiment(title)
                sentiment_label = self._get_sentiment_label(sentiment_score)

                # Extract publisher information
                provider = article_data.get('provider', {})
                publisher = provider.get('displayName', 'Unknown') if isinstance(provider, dict) else 'Unknown'

                # Extract link information
                click_through = article_data.get('clickThroughUrl', {})
                link = click_through.get('url', '') if isinstance(click_through, dict) else article_data.get('link', '')

                # Extract publish time
                pub_date = article_data.get('pubDate') or article_data.get('providerPublishTime')
                if pub_date:
                    # Convert ISO format to timestamp
                    from datetime import datetime
                    if isinstance(pub_date, str):
                        try:
                            dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            publish_time = int(dt.timestamp())
                        except:
                            publish_time = 0
                    else:
                        publish_time = pub_date
                else:
                    publish_time = 0

                # Extract thumbnail
                thumbnail = article_data.get('thumbnail', {})
                if isinstance(thumbnail, dict) and 'resolutions' in thumbnail:
                    thumb_url = thumbnail.get('resolutions', [{}])[0].get('url', '')
                else:
                    thumb_url = ''

                processed_article = {
                    'title': title,
                    'publisher': publisher,
                    'link': link,
                    'publish_time': publish_time,
                    'type': article_data.get('contentType', article_data.get('type', 'news')),
                    'thumbnail': thumb_url,
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label
                }

                processed_articles.append(processed_article)

            # Store in cache
            self.db.insert_news_articles(symbol, processed_articles)
            print(f"[OK] Cached {len(processed_articles)} articles for {symbol}")

            return processed_articles

        except Exception as e:
            print(f"[ERROR] Error fetching news for {symbol}: {str(e)}")
            # Return stale cache as fallback
            cached_news = self.db.get_news_articles(symbol, limit=max_articles)
            if cached_news:
                print(f"[FALLBACK] Returning stale cache for {symbol} news")
                return cached_news
            return []

    def get_multiple_prices(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, Optional[float]]:
        """
        Get current prices for multiple stocks efficiently.

        Args:
            symbols: List of stock ticker symbols
            force_refresh: Force fetch from API

        Returns:
            Dictionary mapping symbols to current prices
        """
        prices = {}

        # Check which symbols need refreshing
        symbols_to_fetch = []
        for symbol in symbols:
            if force_refresh or not self.db.is_info_fresh(symbol, self.info_cache_hours):
                symbols_to_fetch.append(symbol)
            else:
                # Get from cache
                info = self.db.get_stock_info(symbol)
                if info and info.get('current_price'):
                    prices[symbol] = float(info['current_price'])
                    self.cache_hits += 1
                else:
                    symbols_to_fetch.append(symbol)

        # Fetch missing symbols
        if symbols_to_fetch:
            print(f"[API] Fetching prices for {len(symbols_to_fetch)} symbols...")
            for symbol in symbols_to_fetch:
                price = self.get_current_price(symbol, force_refresh=True)
                prices[symbol] = price

        return prices

    def _period_to_start_date(self, period: str) -> str:
        """Convert period string to start date."""
        now = datetime.now()

        period_map = {
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90),
            '6mo': timedelta(days=180),
            '1y': timedelta(days=365),
            '2y': timedelta(days=730),
            '5y': timedelta(days=1825),
            '10y': timedelta(days=3650),
            'ytd': timedelta(days=(now - datetime(now.year, 1, 1)).days),
            'max': timedelta(days=7300)  # ~20 years
        }

        delta = period_map.get(period, timedelta(days=30))
        start_date = now - delta

        return start_date.strftime('%Y-%m-%d')

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0.0

    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label."""
        if sentiment_score > config.SENTIMENT_POSITIVE_THRESHOLD:
            return "Positive"
        elif sentiment_score < config.SENTIMENT_NEGATIVE_THRESHOLD:
            return "Negative"
        else:
            return "Neutral"

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        db_stats = self.db.get_cache_statistics()

        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 1),
            'api_calls': self.api_calls,
            **db_stats
        }

    def clear_old_news(self, days_to_keep: int = 30) -> int:
        """Clear old news articles from cache."""
        return self.db.clear_old_news(days_to_keep)

    def clear_symbol(self, symbol: str):
        """Clear all cached data for a symbol."""
        self.db.clear_cache_for_symbol(symbol.upper())

    def optimize_cache(self):
        """Optimize database (vacuum and rebuild indexes)."""
        print("[INFO] Optimizing cache database...")
        self.db.vacuum()
        print("[OK] Cache optimized")

    def close(self):
        """Close database connection."""
        self.db.close()


# Global instance for easy access
_global_cache = None

def get_cached_fetcher() -> CachedDataFetcher:
    """Get or create global cached data fetcher instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CachedDataFetcher()
    return _global_cache
