"""
Configuration file for Stock Analyzer Pro

This file contains all configuration settings and data source information.
"""

# ==============================================================================
# DATA SOURCES
# ==============================================================================

"""
PRIMARY DATA SOURCE: Yahoo Finance
- Provider: Yahoo Finance (yfinance Python library)
- Cost: FREE - No API key required
- Data Available:
  * Real-time stock prices (15-20 minute delay for free tier)
  * Historical price data (OHLCV)
  * Company information and fundamentals
  * Financial news articles
  * Trading volume and market statistics
- Documentation: https://pypi.org/project/yfinance/
- Terms: For personal use only, subject to Yahoo's terms of service
"""

DATA_SOURCES = {
    'stock_data': {
        'provider': 'Yahoo Finance',
        'library': 'yfinance',
        'api_key_required': False,
        'cost': 'Free',
        'rate_limits': 'Reasonable use (2000 requests/hour recommended)',
        'data_delay': '15-20 minutes for real-time data',
    },
    'news': {
        'provider': 'Yahoo Finance News',
        'library': 'yfinance',
        'api_key_required': False,
        'cost': 'Free',
    },
    'sentiment_analysis': {
        'provider': 'TextBlob (Local NLP)',
        'library': 'textblob',
        'api_key_required': False,
        'cost': 'Free',
        'processing': 'Local machine (no external API calls)',
    },
    'technical_indicators': {
        'provider': 'pandas-ta (Local calculation)',
        'library': 'pandas_ta',
        'api_key_required': False,
        'cost': 'Free',
        'processing': 'Calculated locally from price data',
    }
}

# ==============================================================================
# APPLICATION SETTINGS
# ==============================================================================

# Window settings
WINDOW_TITLE = "Stock Analyzer Pro"
WINDOW_SIZE = "1400x900"
APPEARANCE_MODE = "dark"  # "dark", "light", or "system"
COLOR_THEME = "blue"      # "blue", "green", or "dark-blue"

# Data settings
DEFAULT_CACHE_DURATION = 5  # minutes (legacy, for in-memory cache)
MAX_NEWS_ARTICLES = 20
DEFAULT_TIME_PERIOD = "1mo"

# SQLite Cache settings
USE_CACHE = True  # Enable SQLite caching for better performance
CACHE_DB_PATH = "data/stock_cache.db"
PRICE_CACHE_HOURS = 24  # Cache price data for 24 hours
INFO_CACHE_HOURS = 24   # Cache company info for 24 hours
NEWS_CACHE_HOURS = 1    # Cache news for 1 hour
AUTO_CLEANUP_OLD_NEWS = True  # Automatically clean old news
NEWS_RETENTION_DAYS = 30  # Keep news for 30 days

# Portfolio settings
PORTFOLIO_FILE = "data/portfolio.json"

# Chart settings
CHART_DPI = 100
CHART_STYLE = "default"  # matplotlib style

# Technical indicator settings
MA_PERIODS = [20, 50]  # Moving average periods
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD = 0.2
SENTIMENT_NEGATIVE_THRESHOLD = -0.2

# ==============================================================================
# ALTERNATIVE DATA SOURCES (for future implementation)
# ==============================================================================

"""
Note: The following are alternative data sources that could be integrated
in future versions. Most require API keys and may have costs.

PAID/PREMIUM OPTIONS:
- Alpha Vantage: Free tier available, 5 API requests/minute
- IEX Cloud: Free tier available, limited data
- Polygon.io: Free tier available for delayed data
- Finnhub: Free tier available with rate limits
- News API: Free tier for news (newsapi.org)

BROKERAGE APIS (for live trading):
- Alpaca: Commission-free trading API
- Interactive Brokers: Professional trading platform
- TD Ameritrade: thinkorswim API

FUNDAMENTAL DATA:
- Financial Modeling Prep: Financial statements, ratios
- Quandl: Economic and financial data
- SEC Edgar: Official SEC filings (free)

CRYPTO DATA:
- CoinGecko: Free crypto data API
- Binance: Cryptocurrency exchange API
"""

ALTERNATIVE_DATA_SOURCES = {
    'alpha_vantage': {
        'api_key_required': True,
        'free_tier': '5 requests/minute, 500 requests/day',
        'website': 'https://www.alphavantage.co/',
    },
    'iex_cloud': {
        'api_key_required': True,
        'free_tier': '50,000 messages/month',
        'website': 'https://iexcloud.io/',
    },
    'finnhub': {
        'api_key_required': True,
        'free_tier': '60 API calls/minute',
        'website': 'https://finnhub.io/',
    },
}

# ==============================================================================
# DATA UPDATE FREQUENCIES
# ==============================================================================

UPDATE_FREQUENCIES = {
    'stock_prices': '15-20 minutes (Yahoo Finance delay)',
    'company_info': 'Daily',
    'news': 'Real-time to hourly',
    'historical_data': 'End of trading day',
}

# ==============================================================================
# SUPPORTED MARKETS
# ==============================================================================

"""
Yahoo Finance supports most major global exchanges:
- US: NYSE, NASDAQ, AMEX
- International: LSE, TSE, ASX, HKG, etc.

Symbol format examples:
- US stocks: AAPL, MSFT, GOOGL
- International: TSLA.L (London), 0700.HK (Hong Kong)
- ETFs: SPY, QQQ, VOO
- Indices: ^GSPC (S&P 500), ^DJI (Dow Jones)
- Crypto: BTC-USD, ETH-USD
"""

SUPPORTED_MARKETS = [
    'US (NYSE, NASDAQ, AMEX)',
    'UK (London Stock Exchange)',
    'Europe (major exchanges)',
    'Asia (Tokyo, Hong Kong, Shanghai)',
    'Australia (ASX)',
    'Canada (TSX)',
    'Cryptocurrency pairs',
]

# ==============================================================================
# DISCLAIMERS
# ==============================================================================

DISCLAIMER = """
IMPORTANT DISCLAIMERS:

1. Data Accuracy: Stock data is provided by Yahoo Finance and may have delays
   or inaccuracies. Always verify critical information from official sources.

2. Not Financial Advice: This application is for educational and informational
   purposes only. It does not constitute financial, investment, or trading advice.

3. Investment Risk: Trading stocks involves substantial risk of loss. Past
   performance does not guarantee future results.

4. Personal Use: This application uses free data sources intended for personal,
   non-commercial use only.

5. No Guarantees: The developers make no warranties about the accuracy,
   completeness, or reliability of the information provided.

6. Consult Professionals: Always consult with qualified financial advisors
   before making investment decisions.
"""

# ==============================================================================
# DEEP LEARNING PREDICTIONS SETTINGS
# ==============================================================================

PREDICTIONS = {
    # Model settings
    'model': {
        'type': 'lstm',  # 'lstm', 'gru', or 'transformer'
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'bidirectional': False,
        'num_heads': 4,  # For transformer only
    },

    # Data settings
    'data': {
        'sequence_length': 30,  # Look back 30 days
        'forecast_horizon': 5,  # Predict 5 days ahead
        'test_size': 0.2,  # 20% for testing
        'batch_size': 32,
        'min_training_samples': 100,  # Minimum samples needed to train
    },

    # Training settings
    'training': {
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'device': 'auto',  # 'auto', 'cuda', or 'cpu'
        'early_stopping': {
            'enabled': True,
            'patience': 15,
            'min_delta': 0.0001,
        },
        'checkpoint': {
            'enabled': True,
            'directory': 'data/models',
        },
    },

    # Feature engineering
    'features': {
        'scaler': 'standard',  # 'standard', 'minmax', or 'robust'
        'handle_missing': 'ffill',  # 'ffill', 'bfill', 'drop', or 'zero'
        'technical_indicators': {
            'enabled': True,
            'include_rsi': True,
            'include_macd': True,
            'include_bollinger': True,
        },
        'lag_features': {
            'enabled': True,
            'lags': [1, 2, 3, 5, 10],
        },
        'rolling_features': {
            'enabled': True,
            'windows': [5, 10, 20],
            'functions': ['mean', 'std'],
        },
    },
}
