"""Stock Price Predictions - Streamlit Module"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path
import pickle
import os

import config
from core import LSTMModel, TimeSeriesDataset, TimeSeriesDataLoader
from core.features import TechnicalIndicators, LagFeatures, RollingWindow
from core.data import FeatureEngine
from utils.data_fetcher import StockDataFetcher


def show():
    """Display the Predictions page."""
    st.header("🔮 Stock Price Predictions")
    st.markdown("ML-powered stock price forecasts using LSTM neural networks")

    # Input controls
    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter stock ticker symbol"
        ).upper()

    with col2:
        model_type = st.selectbox(
            "Model Type",
            options=["LSTM", "GRU", "Transformer"],
            index=0
        )

    with col3:
        forecast_horizon = st.number_input(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=5
        )

    # Check if model exists
    model_path = Path(config.PREDICTIONS['training']['checkpoint']['directory']) / \
                 f"{symbol}_{model_type.lower()}_best.pt"
    feature_path = Path(config.PREDICTIONS['training']['checkpoint']['directory']) / \
                   f"{symbol}_{model_type.lower()}_features.pkl"

    if not model_path.exists():
        st.warning(f"⚠️ No trained model found for {symbol}")
        st.info(
            f"To use predictions, you need to train a model first. "
            f"Models are trained using the desktop app or training scripts."
        )

        # Show available models
        models_dir = Path(config.PREDICTIONS['training']['checkpoint']['directory'])
        if models_dir.exists():
            model_files = list(models_dir.glob('*_lstm_best.pt'))
            if model_files:
                st.subheader("📦 Available Models")
                available_symbols = [f.stem.replace('_lstm_best', '') for f in model_files]
                st.write(", ".join(sorted(available_symbols[:20])))
        return

    # Make predictions button
    if st.button("🔮 Make Predictions", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Generating predictions for {symbol}..."):
                # Fetch data
                fetcher = StockDataFetcher()
                data = fetcher.get_stock_data(symbol, period="2y")

                if data is None or len(data) == 0:
                    st.error(f"No data available for {symbol}")
                    return

                # Load feature engine
                with open(feature_path, 'rb') as f:
                    feature_engine = pickle.load(f)

                # Prepare features
                df = data[['Close']].copy() if 'Close' in data.columns else data.copy()
                features = feature_engine.transform(df)

                # Create test dataset
                test_size = config.PREDICTIONS['data']['test_size']
                sequence_length = config.PREDICTIONS['data']['sequence_length']
                train_size = int(len(features) * (1 - test_size))
                test_features = features[train_size:]
                test_targets = test_features[:, 0]

                test_dataset = TimeSeriesDataset(
                    data=test_features,
                    targets=test_targets,
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                )

                # Load model
                model = LSTMModel(
                    input_dim=features.shape[1],
                    hidden_dim=config.PREDICTIONS['model']['hidden_dim'],
                    output_dim=1,
                    num_layers=config.PREDICTIONS['model']['num_layers'],
                    dropout=config.PREDICTIONS['model']['dropout'],
                    forecast_horizon=forecast_horizon,
                )

                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                # Make predictions
                predictions_list = []
                with torch.no_grad():
                    for i in range(len(test_dataset)):
                        seq, target = test_dataset[i]
                        seq = seq.unsqueeze(0)  # Add batch dimension
                        pred = model(seq)

                        # Extract final forecast
                        if len(pred.shape) == 3:
                            predictions_list.append(pred[0, -1, 0].item())
                        else:
                            predictions_list.append(pred[0, -1].item())

                predictions = np.array(predictions_list)
                actuals = test_targets[sequence_length + forecast_horizon - 1:]

                # Calculate metrics
                mse_val = np.mean((actuals - predictions) ** 2)
                mae_val = np.mean(np.abs(actuals - predictions))
                mape_val = np.mean(np.abs((actuals - predictions) / actuals)) * 100

                # Display metrics
                st.success("✅ Predictions generated successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Squared Error", f"{mse_val:.4f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{mae_val:.4f}")
                with col3:
                    st.metric("MAPE", f"{mape_val:.2f}%")

                # Plot predictions
                st.subheader("📈 Predictions vs Actual Prices")

                dates = data.index[train_size + sequence_length + forecast_horizon - 1:]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=actuals,
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ))

                fig.update_layout(
                    title=f"{symbol} - {forecast_horizon} Day Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_white",
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Error plot
                st.subheader("📊 Prediction Errors")

                errors = actuals - predictions

                fig_error = go.Figure()

                fig_error.add_trace(go.Scatter(
                    x=dates,
                    y=errors,
                    mode='lines+markers',
                    name='Error',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.1)'
                ))

                fig_error.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

                fig_error.update_layout(
                    title="Prediction Errors Over Time",
                    xaxis_title="Date",
                    yaxis_title="Error ($)",
                    template="plotly_white",
                    height=300
                )

                st.plotly_chart(fig_error, use_container_width=True)

        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.exception(e)
