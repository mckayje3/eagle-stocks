"""Stock Price Predictions - Streamlit Module"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta

# Import deep-eagle framework and dependencies with fallback
MODELS_AVAILABLE = False
IMPORT_ERROR_MSG = None

try:
    import torch
    from core import LSTMModel, TimeSeriesDataset
    from core.features import FeatureEngine
    from utils.data_fetcher import StockDataFetcher
    import plotly.graph_objects as go
    from core.training import Trainer
    from core.utils import get_device
    import config
    MODELS_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR_MSG = str(e)
    # Fallback imports for basic functionality
    try:
        import plotly.graph_objects as go
    except ImportError:
        pass


def get_available_models():
    """Get list of available trained models."""
    models_dir = Path('data/models')
    if not models_dir.exists():
        return []

    model_files = list(models_dir.glob('*_lstm_best.pt'))
    symbols = sorted([f.stem.replace('_lstm_best', '') for f in model_files])
    return symbols


def load_model_and_predict(symbol):
    """Load a trained model and make predictions."""
    try:
        # Model paths
        model_path = Path(f'data/models/{symbol}_lstm_best.pt')
        features_path = Path(f'data/models/{symbol}_lstm_features.pkl')

        if not model_path.exists() or not features_path.exists():
            return None, f"Model files not found for {symbol}"

        # Load feature engine
        with open(features_path, 'rb') as f:
            feature_engine = pickle.load(f)

        # Get stock data (2 years)
        data_fetcher = StockDataFetcher()
        data = data_fetcher.get_stock_data(symbol, period="2y", interval="1d")

        if data is None or len(data) < 100:
            return None, f"Insufficient data for {symbol}"

        # Prepare features using transform (not fit_transform for loaded engine)
        features = feature_engine.transform(data)
        if features is None or len(features) < 100:
            return None, "Feature engineering failed"

        # Create model
        device = get_device()
        input_dim = features.shape[1]
        sequence_length = config.PREDICTIONS['data']['sequence_length']
        forecast_horizon = config.PREDICTIONS['data']['forecast_horizon']

        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=config.PREDICTIONS['model']['hidden_dim'],
            num_layers=config.PREDICTIONS['model']['num_layers'],
            output_dim=1,
            forecast_horizon=forecast_horizon,
            dropout=config.PREDICTIONS['model']['dropout']
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Prepare test data (last 20%)
        test_size = config.PREDICTIONS['data']['test_size']
        train_size = int(len(features) * (1 - test_size))
        test_features = features[train_size:]
        test_targets = test_features[:, 0]

        # Create dataset
        test_dataset = TimeSeriesDataset(
            data=test_features,
            targets=test_targets,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
        )

        # Make predictions
        trainer = Trainer(model=model, device=device)
        predictions_raw = trainer.predict(test_dataset)

        # Extract final predictions
        if len(predictions_raw.shape) == 3:
            predictions = predictions_raw[:, -1, 0]
        elif len(predictions_raw.shape) == 2:
            predictions = predictions_raw[:, -1]
        else:
            predictions = predictions_raw.flatten()

        test_actuals = test_targets[sequence_length + forecast_horizon - 1:]

        # Get corresponding dates
        dates = data.index[train_size + sequence_length + forecast_horizon - 1:]

        # Calculate metrics
        mse = np.mean((test_actuals - predictions) ** 2)
        mae = np.mean(np.abs(test_actuals - predictions))
        mape = np.mean(np.abs((test_actuals - predictions) / test_actuals)) * 100

        # Get current price and predicted change
        current_price = data['Close'].iloc[-1]
        last_prediction = predictions[-1]
        predicted_change = ((last_prediction - current_price) / current_price) * 100

        return {
            'predictions': predictions,
            'actuals': test_actuals,
            'dates': dates,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'mape': mape
            },
            'current_price': current_price,
            'predicted_price': last_prediction,
            'predicted_change': predicted_change,
            'data': data
        }, None

    except Exception as e:
        return None, f"Error: {str(e)}"


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

        return

    # Get available models
    available_models = get_available_models()

    if not available_models:
        st.warning("⚠️ No trained models found in data/models/")
        st.info("Models will be available after training on the desktop app or uploading to the cloud.")
        return

    st.success(f"✅ {len(available_models)} trained models available!")

    # Stock selection
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Stock Symbol:",
            options=available_models,
            help="Choose a stock to see predictions"
        )

    with col2:
        predict_button = st.button("📊 Make Prediction", use_container_width=True, type="primary")

    st.markdown("---")

    # Make prediction when button clicked or symbol changes
    if predict_button or 'last_symbol' not in st.session_state or st.session_state.last_symbol != selected_symbol:
        st.session_state.last_symbol = selected_symbol

        with st.spinner(f"Loading model and making predictions for {selected_symbol}..."):
            result, error = load_model_and_predict(selected_symbol)

        if error:
            st.error(f"❌ {error}")
            return

        if result:
            # Store in session state
            st.session_state.prediction_result = result

    # Display results if available
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result

        # Key metrics
        st.subheader(f"📈 {selected_symbol} Prediction Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"${result['current_price']:.2f}"
            )

        with col2:
            st.metric(
                "Predicted Price",
                f"${result['predicted_price']:.2f}",
                f"{result['predicted_change']:+.2f}%"
            )

        with col3:
            st.metric(
                "MAE",
                f"${result['metrics']['mae']:.2f}",
                help="Mean Absolute Error"
            )

        with col4:
            st.metric(
                "MAPE",
                f"{result['metrics']['mape']:.2f}%",
                help="Mean Absolute Percentage Error"
            )

        st.markdown("---")

        # Prediction chart
        st.subheader("📊 Prediction vs Actual Prices")

        fig = go.Figure()

        # Actual prices
        fig.add_trace(go.Scatter(
            x=result['dates'],
            y=result['actuals'],
            name='Actual',
            mode='lines',
            line=dict(color='blue', width=2)
        ))

        # Predicted prices
        fig.add_trace(go.Scatter(
            x=result['dates'],
            y=result['predictions'],
            name='Predicted',
            mode='lines',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f"{selected_symbol} Price Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Historical price chart
        with st.expander("📈 View Full Historical Price Chart"):
            fig2 = go.Figure()

            fig2.add_trace(go.Candlestick(
                x=result['data'].index,
                open=result['data']['Open'],
                high=result['data']['High'],
                low=result['data']['Low'],
                close=result['data']['Close'],
                name=selected_symbol
            ))

            fig2.update_layout(
                title=f"{selected_symbol} Historical Prices (2 Years)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig2, use_container_width=True)

        # Model information
        with st.expander("🧠 Model Details"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **Model Architecture:**
                - Type: LSTM Neural Network
                - Parameters: ~136,517
                - Training: 2 years historical data
                - Sequence Length: 60 days
                - Forecast Horizon: 5 days
                """)

            with col2:
                st.markdown("""
                **Performance Metrics:**
                """)
                st.metric("MSE", f"{result['metrics']['mse']:.4f}")
                st.metric("MAE", f"${result['metrics']['mae']:.2f}")
                st.metric("MAPE", f"{result['metrics']['mape']:.2f}%")

    else:
        # Show instructions
        st.info("👆 Select a stock symbol above and click 'Make Prediction' to see ML-powered price forecasts")

        # Show model info
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
