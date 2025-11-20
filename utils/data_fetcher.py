"""
Data fetching utilities for stock data from various sources.

Primary Data Source: Yahoo Finance (via yfinance library)
- No API key required
- Free to use
- Real-time and historical data

Now with SQLite caching for faster performance!
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import config

# Try to import cache manager, fall back to direct fetching if unavailable
try:
    from utils.cache_manager import get_cached_fetcher
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

class StockDataFetcher:
    """Centralized class for fetching stock market data with optional caching."""

    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """
        Validate if a stock symbol exists and is tradeable.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            # Check if we got valid data
            if not info or 'symbol' not in info:
                return False, f"Symbol '{symbol}' not found or invalid"

            # Check if it's a valid equity
            quote_type = info.get('quoteType', '')
            if quote_type not in ['EQUITY', 'ETF', 'MUTUALFUND']:
                return False, f"'{symbol}' is a {quote_type}, not a tradeable equity"

            return True, f"Valid symbol: {info.get('longName', symbol)}"

        except Exception as e:
            return False, f"Error validating symbol: {str(e)}"

    @staticmethod
    def get_stock_data(symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data with caching.

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data or None if error
        """
        # Use cache if available and enabled
        if CACHE_AVAILABLE and config.USE_CACHE:
            cache = get_cached_fetcher()
            return cache.get_stock_data(symbol, period, interval)

        # Fall back to direct fetching
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)

            if data.empty:
                return None

            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    @staticmethod
    def get_stock_info(symbol: str) -> Optional[Dict]:
        """
        Fetch detailed stock information with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information or None if error
        """
        # Use cache if available and enabled
        if CACHE_AVAILABLE and config.USE_CACHE:
            cache = get_cached_fetcher()
            return cache.get_stock_info(symbol)

        # Fall back to direct fetching
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            if not info:
                return None

            # Extract relevant information
            stock_info = {
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

            return stock_info

        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
            return None

    @staticmethod
    def get_current_price(symbol: str) -> Optional[float]:
        """
        Fetch current/latest stock price with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price or None if error
        """
        # Use cache if available and enabled
        if CACHE_AVAILABLE and config.USE_CACHE:
            cache = get_cached_fetcher()
            return cache.get_current_price(symbol)

        # Fall back to direct fetching
        try:
            stock = yf.Ticker(symbol)

            # Try to get from info
            info = stock.info
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))

            if price and price > 0:
                return float(price)

            # Fallback to latest history
            hist = stock.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            return None

        except Exception as e:
            print(f"Error fetching price for {symbol}: {str(e)}")
            return None

    @staticmethod
    def get_news(symbol: str, max_articles: int = 20) -> List[Dict]:
        """
        Fetch latest news articles for a stock with caching and sentiment analysis.

        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles to fetch

        Returns:
            List of news articles with metadata and sentiment
        """
        # Use cache if available and enabled
        if CACHE_AVAILABLE and config.USE_CACHE:
            cache = get_cached_fetcher()
            return cache.get_news(symbol, max_articles)

        # Fall back to direct fetching
        try:
            stock = yf.Ticker(symbol)
            news = stock.news

            if not news:
                return []

            # Process and format news
            articles = []
            for article in news[:max_articles]:
                # Handle new yfinance structure where data is nested in 'content'
                if 'content' in article:
                    article_data = article['content']
                else:
                    article_data = article

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
                    if isinstance(pub_date, str):
                        try:
                            from datetime import datetime
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

                articles.append({
                    'title': article_data.get('title', 'No title'),
                    'publisher': publisher,
                    'link': link,
                    'publish_time': publish_time,
                    'type': article_data.get('contentType', article_data.get('type', 'news')),
                    'thumbnail': thumb_url
                })

            return articles

        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return []

    @staticmethod
    def get_multiple_prices(symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Fetch current prices for multiple stocks efficiently with caching.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to current prices
        """
        # Use cache if available and enabled
        if CACHE_AVAILABLE and config.USE_CACHE:
            cache = get_cached_fetcher()
            return cache.get_multiple_prices(symbols)

        # Fall back to direct fetching
        prices = {}

        try:
            # Use yfinance's batch download feature
            tickers = yf.Tickers(' '.join(symbols))

            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    price = info.get('currentPrice', info.get('regularMarketPrice', 0))

                    if not price or price == 0:
                        # Fallback to history
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            price = hist['Close'].iloc[-1]

                    prices[symbol] = float(price) if price else None

                except Exception:
                    prices[symbol] = None

            return prices

        except Exception as e:
            print(f"Error fetching multiple prices: {str(e)}")
            # Fallback to individual fetches
            for symbol in symbols:
                prices[symbol] = StockDataFetcher.get_current_price(symbol)

            return prices

    @staticmethod
    def search_symbol(query: str) -> List[Dict]:
        """
        Search for stock symbols by company name or partial symbol.

        Args:
            query: Search query (company name or partial symbol)

        Returns:
            List of matching stocks with their symbols and names
        """
        try:
            # Note: yfinance doesn't have a built-in search, but we can try common variations
            # This is a basic implementation - for production, consider using a dedicated API

            query_upper = query.upper()

            # Try exact match first
            stock = yf.Ticker(query_upper)
            info = stock.info

            if info and 'symbol' in info:
                return [{
                    'symbol': info.get('symbol', query_upper),
                    'name': info.get('longName', info.get('shortName', 'N/A')),
                    'type': info.get('quoteType', 'N/A')
                }]

            return []

        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")
            return []


class DataCache:
    """Simple in-memory cache for stock data to reduce API calls."""

    def __init__(self, cache_duration_minutes: int = 5):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)

    def get(self, key: str) -> Optional[any]:
        """Get cached data if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            else:
                # Cache expired, remove it
                del self.cache[key]
        return None

    def set(self, key: str, data: any):
        """Store data in cache with current timestamp."""
        self.cache[key] = (data, datetime.now())

    def clear(self):
        """Clear all cached data."""
        self.cache.clear()

    def remove(self, key: str):
        """Remove specific key from cache."""
        if key in self.cache:
            del self.cache[key]
