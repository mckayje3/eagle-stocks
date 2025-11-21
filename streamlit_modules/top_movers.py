"""Top Movers Prediction - Streamlit Module"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Import config with fallback
try:
    import config
    import torch
    from core import LSTMModel, TimeSeriesDataset
    from core.features import FeatureEngine
    from core.training import Trainer
    from core.utils import get_device
    from utils.data_fetcher import StockDataFetcher
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # Fallback config values
    class config:
        PREDICTIONS = {
            'training': {
                'checkpoint': {
                    'directory': 'data/models'
                }
            }
        }


def get_available_models():
    """Get list of available trained models."""
    models_dir = Path('data/models')
    if not models_dir.exists():
        return []

    model_files = list(models_dir.glob('*_lstm_best.pt'))
    symbols = sorted([f.stem.replace('_lstm_best', '') for f in model_files])
    return symbols


def calculate_prediction(symbol):
    """Calculate prediction for a single stock."""
    try:
        # Model paths
        model_path = Path(f'data/models/{symbol}_lstm_best.pt')
        features_path = Path(f'data/models/{symbol}_lstm_features.pkl')

        if not model_path.exists() or not features_path.exists():
            return None

        # Load feature engine
        with open(features_path, 'rb') as f:
            feature_engine = pickle.load(f)

        # Get stock data
        data_fetcher = StockDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years

        data = data_fetcher.get_stock_data(symbol, start_date, end_date)
        if data is None or len(data) < 100:
            return None

        # Prepare features
        features = feature_engine.fit_transform(data)
        if features is None or len(features) < 100:
            return None

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

        # Use last few data points to make forward prediction
        test_features = features[-sequence_length-10:]
        test_targets = test_features[:, 0]

        test_dataset = TimeSeriesDataset(
            data=test_features,
            targets=test_targets,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
        )

        # Make prediction
        trainer = Trainer(model=model, device=device)
        predictions_raw = trainer.predict(test_dataset)

        # Extract final prediction
        if len(predictions_raw.shape) == 3:
            prediction = predictions_raw[-1, -1, 0]
        elif len(predictions_raw.shape) == 2:
            prediction = predictions_raw[-1, -1]
        else:
            prediction = predictions_raw[-1]

        # Get current price
        current_price = data['Close'].iloc[-1]
        predicted_change_pct = ((prediction - current_price) / current_price) * 100

        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(prediction),
            'predicted_change_pct': float(predicted_change_pct)
        }

    except Exception as e:
        return None


def calculate_all_predictions(symbols, progress_callback=None):
    """Calculate predictions for all symbols."""
    predictions = []
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(i + 1, total, symbol)

        result = calculate_prediction(symbol)
        if result:
            predictions.append(result)

    return predictions


def load_cached_predictions():
    """Load cached predictions if available and fresh."""
    cache_file = Path('data/top_movers_cache.pkl')

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600

                if age_hours < 24:  # Cache valid for 24 hours
                    return cache_data['predictions'], cache_time
        except:
            pass

    return None, None


def save_predictions_cache(predictions):
    """Save predictions to cache."""
    cache_file = Path('data/top_movers_cache.pkl')
    cache_data = {
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    }

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        return True
    except:
        return False


def show():
    """Display the Top Movers page."""
    st.header("🚀 Predict Top Movers")
    st.markdown("Find the best predicted stock performers using ML models")

    if not MODELS_AVAILABLE:
        st.warning("⚠️ ML framework not available. Cannot calculate predictions.")
        st.info("This feature requires PyTorch and the deep-eagle framework.")
        return

    # Get available models
    available_symbols = get_available_models()

    if not available_symbols:
        st.warning("⚠️ No trained models found.")
        st.info("Train models using the desktop app to enable Top Movers predictions.")
        return

    st.info(f"📊 Analyzing predictions for {len(available_symbols)} stocks")

    # Load cached predictions
    cached_predictions, cache_time = load_cached_predictions()

    col1, col2 = st.columns([2, 1])

    with col1:
        if cached_predictions:
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            st.success(f"✅ Loaded cached predictions from {cache_time.strftime('%Y-%m-%d %H:%M')} ({age_hours:.1f} hours ago)")
        else:
            st.info("💡 No cached predictions found. Click 'Calculate Predictions' to generate new predictions.")

    with col2:
        calculate_button = st.button("🔄 Calculate Predictions", use_container_width=True, type="primary")

    # Calculate new predictions if requested
    if calculate_button:
        st.markdown("---")
        st.subheader("📊 Calculating Predictions...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total, symbol):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Processing {current}/{total}: {symbol}")

        predictions = calculate_all_predictions(available_symbols, update_progress)

        if predictions:
            # Save to cache
            save_predictions_cache(predictions)
            cached_predictions = predictions
            cache_time = datetime.now()

            progress_bar.progress(1.0)
            status_text.text(f"✅ Complete! Calculated {len(predictions)} predictions")
            st.success(f"Successfully calculated predictions for {len(predictions)} stocks!")
        else:
            st.error("Failed to calculate predictions. Please try again.")
            return

    if not cached_predictions:
        return

    st.markdown("---")

    # Sort predictions
    df = pd.DataFrame(cached_predictions)
    df_sorted = df.sort_values('predicted_change_pct', ascending=False)

    # Display tabs for winners and losers
    tab1, tab2, tab3 = st.tabs(["🟢 Top Winners", "🔴 Top Losers", "📊 All Predictions"])

    with tab1:
        st.subheader("📈 Predicted Top Gainers")

        winners = df_sorted[df_sorted['predicted_change_pct'] > 0].head(20)

        if len(winners) > 0:
            for idx, row in winners.iterrows():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"**{row['symbol']}**")

                with col2:
                    st.metric("Current", f"${row['current_price']:.2f}")

                with col3:
                    st.metric("Predicted", f"${row['predicted_price']:.2f}")

                with col4:
                    st.metric("Change", f"+{abs(row['predicted_change_pct']):.2f}%",
                             delta=f"+{abs(row['predicted_change_pct']):.2f}%")

                st.markdown("---")
        else:
            st.info("No winners predicted")

    with tab2:
        st.subheader("📉 Predicted Top Losers")

        losers = df_sorted[df_sorted['predicted_change_pct'] < 0].tail(20).sort_values('predicted_change_pct')

        if len(losers) > 0:
            for idx, row in losers.iterrows():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"**{row['symbol']}**")

                with col2:
                    st.metric("Current", f"${row['current_price']:.2f}")

                with col3:
                    st.metric("Predicted", f"${row['predicted_price']:.2f}")

                with col4:
                    st.metric("Change", f"-{abs(row['predicted_change_pct']):.2f}%",
                             delta=f"-{abs(row['predicted_change_pct']):.2f}%",
                             delta_color="inverse")

                st.markdown("---")
        else:
            st.info("No losers predicted")

    with tab3:
        st.subheader("📊 All Predictions")

        # Format dataframe for display
        display_df = df_sorted.copy()
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"${x:.2f}")
        display_df['predicted_change_pct'] = display_df['predicted_change_pct'].apply(lambda x: f"{x:+.2f}%")

        display_df.columns = ['Symbol', 'Current Price', 'Predicted Price', 'Predicted Change %']

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button
        csv = df_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name=f"top_movers_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Clear cache button
    st.markdown("---")
    if st.button("🗑️ Clear Cache"):
        cache_file = Path('data/top_movers_cache.pkl')
        if cache_file.exists():
            cache_file.unlink()
            st.success("✅ Cache cleared! Click 'Calculate Predictions' to generate new predictions.")
            st.rerun()
