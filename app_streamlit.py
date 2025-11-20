"""
Stock Market Analyzer - Streamlit Web App

A comprehensive stock analysis tool with ML predictions, sentiment analysis, and market insights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import pickle
from pathlib import Path
import os

# Import local modules
import config
from utils.data_fetcher import StockDataFetcher

# Page configuration
st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #ff1744;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = StockDataFetcher()

if 'top_movers_cache' not in st.session_state:
    cache_file = Path('data/top_movers_cache.pkl')
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                if age_hours < 24:
                    st.session_state.top_movers_cache = cache_data
        except:
            pass

# Main title
st.markdown('<h1 class="main-header">📈 Stock Market Analyzer</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Market Dashboard", "📰 News & Sentiment", "🔮 Predictions", "🚀 Top Movers"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip:** All data is cached to improve performance. "
    "Refresh the page to fetch the latest data."
)

# Main content based on selected page
if page == "🏠 Market Dashboard":
    from streamlit_modules import market_dashboard
    market_dashboard.show()

elif page == "📰 News & Sentiment":
    from streamlit_modules import news_sentiment
    news_sentiment.show()

elif page == "🔮 Predictions":
    from streamlit_modules import predictions
    predictions.show()

elif page == "🚀 Top Movers":
    from streamlit_modules import top_movers
    top_movers.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Stock Market Analyzer v2.0<br>
    Built with Streamlit & PyTorch
    </div>
    """,
    unsafe_allow_html=True
)
