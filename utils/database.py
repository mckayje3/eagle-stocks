"""
SQLite database module for caching stock market data.

Database Schema:
- stock_prices: Historical OHLCV data
- stock_info: Company information and fundamentals
- news_articles: News articles with metadata
- cache_metadata: Track cache freshness and statistics
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import json
from pathlib import Path

class StockDatabase:
    """SQLite database manager for stock market data."""

    def __init__(self, db_path: str = "data/stock_cache.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys=ON")

    def create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Stock prices table (historical OHLCV data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                interval TEXT DEFAULT 'daily',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, interval)
            )
        """)

        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_symbol_date
            ON stock_prices(symbol, date DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_symbol_interval
            ON stock_prices(symbol, interval, date DESC)
        """)

        # Stock info table (company information)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                current_price REAL,
                previous_close REAL,
                open REAL,
                day_low REAL,
                day_high REAL,
                year_low REAL,
                year_high REAL,
                volume INTEGER,
                avg_volume INTEGER,
                pe_ratio REAL,
                eps REAL,
                dividend_yield REAL,
                beta REAL,
                raw_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # News articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                publisher TEXT,
                link TEXT,
                publish_time TIMESTAMP,
                article_type TEXT,
                thumbnail_url TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, title, publish_time)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_symbol_time
            ON news_articles(symbol, publish_time DESC)
        """)

        # Cache metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                symbol TEXT PRIMARY KEY,
                last_price_update TIMESTAMP,
                last_info_update TIMESTAMP,
                last_news_update TIMESTAMP,
                price_data_start_date DATE,
                price_data_end_date DATE,
                total_price_records INTEGER DEFAULT 0,
                total_news_articles INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def insert_price_data(self, symbol: str, df: pd.DataFrame, interval: str = 'daily'):
        """
        Insert historical price data into database.

        Args:
            symbol: Stock ticker symbol
            df: DataFrame with OHLCV data (index must be datetime)
            interval: Data interval ('daily', '1h', '5m', etc.)
        """
        if df is None or df.empty:
            return

        cursor = self.conn.cursor()

        # Prepare data for insertion
        records = []
        for date, row in df.iterrows():
            records.append((
                symbol,
                date.strftime('%Y-%m-%d'),
                float(row.get('Open', 0)) if pd.notna(row.get('Open')) else None,
                float(row.get('High', 0)) if pd.notna(row.get('High')) else None,
                float(row.get('Low', 0)) if pd.notna(row.get('Low')) else None,
                float(row.get('Close', 0)) if pd.notna(row.get('Close')) else None,
                int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None,
                float(row.get('Adj Close', row.get('Close', 0))) if pd.notna(row.get('Adj Close', row.get('Close'))) else None,
                interval
            ))

        # Insert or replace records
        cursor.executemany("""
            INSERT OR REPLACE INTO stock_prices
            (symbol, date, open, high, low, close, volume, adj_close, interval)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

        # Update metadata
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')

        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata
            (symbol, last_price_update, price_data_start_date, price_data_end_date, total_price_records)
            VALUES (
                ?,
                CURRENT_TIMESTAMP,
                ?,
                ?,
                (SELECT COUNT(*) FROM stock_prices WHERE symbol = ? AND interval = ?)
            )
        """, (symbol, start_date, end_date, symbol, interval))

        self.conn.commit()

    def get_price_data(self, symbol: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None, interval: str = 'daily') -> Optional[pd.DataFrame]:
        """
        Retrieve historical price data from database.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            DataFrame with OHLCV data or None
        """
        query = """
            SELECT date, open, high, low, close, volume, adj_close
            FROM stock_prices
            WHERE symbol = ? AND interval = ?
        """
        params = [symbol, interval]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        try:
            df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['date'], index_col='date')

            if df.empty:
                return None

            # Rename columns to match yfinance format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

            return df

        except Exception as e:
            print(f"Error retrieving price data: {e}")
            return None

    def insert_stock_info(self, symbol: str, info: Dict):
        """
        Insert or update stock information.

        Args:
            symbol: Stock ticker symbol
            info: Dictionary with stock information
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO stock_info
            (symbol, name, sector, industry, market_cap, current_price, previous_close,
             open, day_low, day_high, year_low, year_high, volume, avg_volume,
             pe_ratio, eps, dividend_yield, beta, raw_data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            symbol,
            info.get('name', info.get('longName', info.get('shortName'))),
            info.get('sector'),
            info.get('industry'),
            info.get('market_cap', info.get('marketCap')),
            info.get('current_price', info.get('currentPrice', info.get('regularMarketPrice'))),
            info.get('previous_close', info.get('previousClose')),
            info.get('open'),
            info.get('day_low', info.get('dayLow')),
            info.get('day_high', info.get('dayHigh')),
            info.get('year_low', info.get('fiftyTwoWeekLow')),
            info.get('year_high', info.get('fiftyTwoWeekHigh')),
            info.get('volume'),
            info.get('avg_volume', info.get('averageVolume')),
            info.get('pe_ratio', info.get('trailingPE')),
            info.get('eps', info.get('trailingEps')),
            info.get('dividend_yield', info.get('dividendYield')),
            info.get('beta'),
            json.dumps(info)  # Store raw data for future use
        ))

        # Update metadata
        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata (symbol, last_info_update)
            VALUES (?, CURRENT_TIMESTAMP)
        """, (symbol,))

        self.conn.commit()

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Retrieve stock information from database.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM stock_info WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()

        if not row:
            return None

        # Convert row to dictionary
        info = dict(row)

        # Parse raw_data if available
        if info.get('raw_data'):
            try:
                raw = json.loads(info['raw_data'])
                info.update(raw)
            except:
                pass

        return info

    def insert_news_articles(self, symbol: str, articles: List[Dict]):
        """
        Insert news articles into database.

        Args:
            symbol: Stock ticker symbol
            articles: List of article dictionaries
        """
        if not articles:
            return

        cursor = self.conn.cursor()

        records = []
        for article in articles:
            # Convert timestamp to datetime
            pub_time = article.get('publish_time', 0)
            if isinstance(pub_time, int) and pub_time > 0:
                pub_datetime = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M:%S')
            else:
                pub_datetime = None

            records.append((
                symbol,
                article.get('title', 'No title'),
                article.get('publisher', 'Unknown'),
                article.get('link', ''),
                pub_datetime,
                article.get('type', 'news'),
                article.get('thumbnail', ''),
                article.get('sentiment_score'),
                article.get('sentiment_label')
            ))

        cursor.executemany("""
            INSERT OR IGNORE INTO news_articles
            (symbol, title, publisher, link, publish_time, article_type, thumbnail_url, sentiment_score, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

        # Update metadata
        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata
            (symbol, last_news_update, total_news_articles)
            VALUES (
                ?,
                CURRENT_TIMESTAMP,
                (SELECT COUNT(*) FROM news_articles WHERE symbol = ?)
            )
        """, (symbol, symbol))

        self.conn.commit()

    def get_news_articles(self, symbol: str, limit: int = 20, days_back: int = 30) -> List[Dict]:
        """
        Retrieve news articles from database.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles
            days_back: How many days back to retrieve

        Returns:
            List of article dictionaries
        """
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM news_articles
            WHERE symbol = ? AND publish_time >= ?
            ORDER BY publish_time DESC
            LIMIT ?
        """, (symbol, cutoff_date, limit))

        articles = []
        for row in cursor.fetchall():
            articles.append(dict(row))

        return articles

    def is_price_data_fresh(self, symbol: str, interval: str = 'daily',
                           max_age_hours: int = 24) -> bool:
        """
        Check if cached price data is fresh enough.

        Args:
            symbol: Stock ticker symbol
            interval: Data interval
            max_age_hours: Maximum age in hours

        Returns:
            True if cache is fresh, False otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT last_price_update FROM cache_metadata WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()
        if not row or not row['last_price_update']:
            return False

        last_update = datetime.strptime(row['last_price_update'], '%Y-%m-%d %H:%M:%S')
        age = datetime.now() - last_update

        return age.total_seconds() / 3600 < max_age_hours

    def is_info_fresh(self, symbol: str, max_age_hours: int = 24) -> bool:
        """Check if cached stock info is fresh."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT last_info_update FROM cache_metadata WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()
        if not row or not row['last_info_update']:
            return False

        last_update = datetime.strptime(row['last_info_update'], '%Y-%m-%d %H:%M:%S')
        age = datetime.now() - last_update

        return age.total_seconds() / 3600 < max_age_hours

    def is_news_fresh(self, symbol: str, max_age_hours: int = 1) -> bool:
        """Check if cached news is fresh."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT last_news_update FROM cache_metadata WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()
        if not row or not row['last_news_update']:
            return False

        last_update = datetime.strptime(row['last_news_update'], '%Y-%m-%d %H:%M:%S')
        age = datetime.now() - last_update

        return age.total_seconds() / 3600 < max_age_hours

    def get_cache_statistics(self) -> Dict:
        """Get overall cache statistics."""
        cursor = self.conn.cursor()

        stats = {}

        # Total symbols cached
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM stock_prices")
        stats['total_symbols_with_prices'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM stock_info")
        stats['total_symbols_with_info'] = cursor.fetchone()[0]

        # Total records
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        stats['total_price_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM news_articles")
        stats['total_news_articles'] = cursor.fetchone()[0]

        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = cursor.fetchone()[0]
        stats['database_size_mb'] = round(size_bytes / (1024 * 1024), 2)

        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM stock_prices")
        row = cursor.fetchone()
        stats['earliest_price_date'] = row[0]
        stats['latest_price_date'] = row[1]

        return stats

    def clear_old_news(self, days_to_keep: int = 30):
        """Delete news articles older than specified days."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM news_articles WHERE publish_time < ?", (cutoff_date,))
        deleted = cursor.rowcount
        self.conn.commit()

        return deleted

    def clear_cache_for_symbol(self, symbol: str):
        """Clear all cached data for a specific symbol."""
        cursor = self.conn.cursor()

        cursor.execute("DELETE FROM stock_prices WHERE symbol = ?", (symbol,))
        cursor.execute("DELETE FROM stock_info WHERE symbol = ?", (symbol,))
        cursor.execute("DELETE FROM news_articles WHERE symbol = ?", (symbol,))
        cursor.execute("DELETE FROM cache_metadata WHERE symbol = ?", (symbol,))

        self.conn.commit()

    def vacuum(self):
        """Optimize database (reclaim space, rebuild indexes)."""
        self.conn.execute("VACUUM")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
