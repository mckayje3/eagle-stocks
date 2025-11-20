"""Top Movers Prediction - Streamlit Module"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

import config


def show():
    """Display the Top Movers page."""
    st.header("🚀 Predict Top Movers")
    st.markdown("Find the best predicted stock performers using ML models")

    # Load cached predictions if available
    cache_file = Path('data/top_movers_cache.pkl')
    cached_data = None

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600

                if age_hours < 24:
                    cached_data = cache_data
                    st.info(f"📦 Loaded cached predictions from {cache_time.strftime('%Y-%m-%d %H:%M')} ({age_hours:.1f} hours ago)")
        except:
            pass

    if cached_data:
        predictions = cached_data['predictions']

        # Display tabs for winners and losers
        tab1, tab2 = st.tabs(["🟢 Top Winners", "🔴 Top Losers"])

        with tab1:
            st.subheader("📈 Predicted Top Gainers")

            if predictions['winners']:
                df_winners = pd.DataFrame(predictions['winners'])

                # Format for display
                for idx, row in df_winners.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                    with col1:
                        st.markdown(f"**{row['symbol']}**")

                    with col2:
                        st.metric("Current", f"${row['current_price']:.2f}")

                    with col3:
                        st.metric("Predicted", f"${row['predicted_price']:.2f}")

                    with col4:
                        st.metric("Change", f"+{abs(row['predicted_change_pct']):.2f}%", delta=f"+{abs(row['predicted_change_pct']):.2f}%")

                    st.markdown("---")
            else:
                st.info("No winners predicted")

        with tab2:
            st.subheader("📉 Predicted Top Losers")

            if predictions['losers']:
                df_losers = pd.DataFrame(predictions['losers'])

                # Format for display
                for idx, row in df_losers.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                    with col1:
                        st.markdown(f"**{row['symbol']}**")

                    with col2:
                        st.metric("Current", f"${row['current_price']:.2f}")

                    with col3:
                        st.metric("Predicted", f"${row['predicted_price']:.2f}")

                    with col4:
                        st.metric("Change", f"-{abs(row['predicted_change_pct']):.2f}%", delta=f"-{abs(row['predicted_change_pct']):.2f}%", delta_color="inverse")

                    st.markdown("---")
            else:
                st.info("No losers predicted")

        # Clear cache button
        if st.button("🗑️ Clear Cached Predictions"):
            if cache_file.exists():
                cache_file.unlink()
                st.success("Cache cleared! Refresh the page to see updates.")
                st.rerun()

    else:
        st.warning("⚠️ No cached predictions available")
        st.info(
            "Predictions are generated using the desktop app. "
            "Run 'Predict Top Movers' in the desktop app to generate predictions, "
            "then refresh this page to view them."
        )

        # Show available models
        models_dir = Path(config.PREDICTIONS['training']['checkpoint']['directory'])
        if models_dir.exists():
            model_files = list(models_dir.glob('*_lstm_best.pt'))
            if model_files:
                st.subheader("📦 Available Models")
                st.write(f"You have {len(model_files)} trained models ready for predictions")

                # Show a sample
                available_symbols = [f.stem.replace('_lstm_best', '') for f in model_files[:20]]
                st.write(", ".join(sorted(available_symbols)))

                if len(model_files) > 20:
                    st.write(f"...and {len(model_files) - 20} more")
