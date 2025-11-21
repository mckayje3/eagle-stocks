"""Stock Price Predictions - Streamlit Module"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Import deep-eagle framework and dependencies with fallback
MODELS_AVAILABLE = False
IMPORT_ERROR_MSG = None

try:
    import torch
    from core import LSTMModel, TimeSeriesDataset
    from core.features import FeatureEngine
    from utils.data_fetcher import StockDataFetcher
    import plotly.graph_objects as go
    MODELS_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR_MSG = str(e)
    # Fallback imports for basic functionality
    try:
        import plotly.graph_objects as go
    except ImportError:
        pass


def show():
    """Display the Predictions page."""
    st.header("🔮 Stock Price Predictions")
    st.markdown("ML-powered stock price forecasts using LSTM neural networks")

    # Check if models framework is available
    if not MODELS_AVAILABLE:
        st.warning("⚠️ Deep learning framework not available. Predictions are disabled.")
        st.info(
            "💡 **Note:** ML predictions require the deep-eagle framework. "
            "To enable predictions, ensure the framework is properly installed."
        )

        # Show technical details in expander for debugging
        if IMPORT_ERROR_MSG:
            with st.expander("🔧 Technical Details"):
                st.code(f"Import Error: {IMPORT_ERROR_MSG}", language="text")
                st.markdown("""
                **Common fixes:**
                - Ensure `deep-eagle` repository is accessible
                - Check that PyTorch is installed (`torch>=2.0.0`)
                - Verify all dependencies in requirements.txt are installed
                """)

        st.markdown("---")

        # Continue to show informational content below
        # Don't return early - let users see what predictions look like

    # Show upload models option
    st.info(
        "💡 **Tip:** Models are trained locally on your desktop. "
        "You can upload specific model files to enable cloud predictions, or use this "
        "page to explore model information."
    )

    st.markdown("---")

    # Show feature description
    st.subheader("🧠 About the Prediction Models")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Model Architecture:**
        - 🔷 LSTM Neural Networks
        - 📊 136,517 parameters per model
        - 🎯 5-day price forecasts
        - 📈 Trained on 2 years of historical data
        """)

    with col2:
        st.markdown("""
        **Features Used:**
        - 📉 Technical Indicators (RSI, MACD, Bollinger Bands)
        - 🔄 Lag Features (1-10 days)
        - 📊 Rolling Statistics (7, 14, 30 day windows)
        - 🎲 336 total engineered features
        """)

    st.markdown("---")

    # Desktop app info
    st.subheader("🖥️ Using Predictions (Desktop App)")

    st.markdown("""
    To use the prediction features:

    1. **Run the desktop app** (`app.py`) on your local computer
    2. Go to the **Predictions** tab
    3. **Load or train models** for stocks you're interested in
    4. **Make predictions** and view detailed charts
    5. **Top Movers** feature ranks all predictions

    The desktop app has access to:
    - ✅ All 205+ trained models
    - ✅ Generic universal model (averaged from all stocks)
    - ✅ Full PyTorch framework for inference
    - ✅ Real-time prediction visualization
    """)

    st.markdown("---")

    # Example predictions
    st.subheader("📊 What Predictions Look Like")

    st.markdown("""
    The prediction system provides:

    - **Interactive Charts**: Predicted vs actual prices
    - **Error Analysis**: Prediction accuracy over time
    - **Performance Metrics**: MSE, MAE, MAPE
    - **Confidence Intervals**: Uncertainty quantification
    """)

    # Sample visualization placeholder
    st.image("https://via.placeholder.com/800x400/1f77b4/ffffff?text=Sample+Prediction+Chart",
             caption="Example: Stock price predictions with actual vs predicted comparison")

    st.markdown("---")

    # Call to action
    st.info(
        "🚀 **Want to try predictions?** Run the desktop app on your computer to access "
        "the full ML prediction suite with all 205 trained models!"
    )
