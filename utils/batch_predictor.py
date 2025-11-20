"""
Batch Predictor - Make predictions for multiple stocks efficiently

This module handles batch predictions for identifying top movers.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
from pathlib import Path

from core import TimeSeriesDataset, TimeSeriesDataLoader
from core.utils import get_device
from utils.data_fetcher import StockDataFetcher
from utils.prediction_tracker import PredictionTracker
import config


class BatchPredictor:
    """Make predictions for multiple stocks using trained models."""

    def __init__(self, model_type='lstm', forecast_horizon=1):
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.device = get_device() if config.PREDICTIONS['training']['device'] == 'auto' \
                      else config.PREDICTIONS['training']['device']
        self.tracker = PredictionTracker()
        self.fetcher = StockDataFetcher()

    def load_model_for_symbol(self, symbol):
        """
        Load a trained model for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            tuple: (model, feature_engine) or (None, None) if not found
        """
        try:
            # Model path
            model_path = os.path.join(
                config.PREDICTIONS['training']['checkpoint']['directory'],
                f"{symbol}_{self.model_type}_best.pt"
            )

            # Feature engine path
            feature_path = os.path.join(
                config.PREDICTIONS['training']['checkpoint']['directory'],
                f"{symbol}_{self.model_type}_features.pkl"
            )

            if not os.path.exists(model_path) or not os.path.exists(feature_path):
                return None, None

            # Load feature engine
            with open(feature_path, 'rb') as f:
                feature_engine = pickle.load(f)

            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Recreate model (we need to know the architecture)
            from core import LSTMModel, GRUModel, TransformerModel

            # Get input dimension from a sample transformation
            # This is a bit hacky but works
            sample_data = self.fetcher.get_stock_data(symbol, period="5d")
            if sample_data is None:
                return None, None

            df = sample_data[['Close']].copy() if 'Close' in sample_data.columns else sample_data.copy()
            features = feature_engine.transform(df)
            input_dim = features.shape[1]

            # Create model based on type
            if self.model_type == 'lstm':
                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=config.PREDICTIONS['model']['hidden_dim'],
                    output_dim=1,
                    num_layers=config.PREDICTIONS['model']['num_layers'],
                    dropout=config.PREDICTIONS['model']['dropout'],
                    forecast_horizon=self.forecast_horizon,
                )
            elif self.model_type == 'gru':
                model = GRUModel(
                    input_dim=input_dim,
                    hidden_dim=config.PREDICTIONS['model']['hidden_dim'],
                    output_dim=1,
                    num_layers=config.PREDICTIONS['model']['num_layers'],
                    dropout=config.PREDICTIONS['model']['dropout'],
                    forecast_horizon=self.forecast_horizon,
                )
            else:  # transformer
                model = TransformerModel(
                    input_dim=input_dim,
                    hidden_dim=config.PREDICTIONS['model']['hidden_dim'],
                    output_dim=1,
                    num_heads=config.PREDICTIONS['model']['num_heads'],
                    dropout=config.PREDICTIONS['model']['dropout'],
                    forecast_horizon=self.forecast_horizon,
                )

            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            return model, feature_engine

        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            return None, None

    def predict_for_symbol(self, symbol, model=None, feature_engine=None):
        """
        Make prediction for a single symbol.

        Args:
            symbol: Stock symbol
            model: Pre-loaded model (optional)
            feature_engine: Pre-loaded feature engine (optional)

        Returns:
            dict with prediction results or None
        """
        try:
            # Load model if not provided
            if model is None or feature_engine is None:
                model, feature_engine = self.load_model_for_symbol(symbol)
                if model is None:
                    return None

            # Get recent data
            data = self.fetcher.get_stock_data(symbol, period="3mo")
            if data is None or len(data) < config.PREDICTIONS['data']['sequence_length']:
                return None

            # Get current price
            current_price = data['Close'].iloc[-1]

            # Prepare features
            df = data[['Close']].copy() if 'Close' in data.columns else data.copy()
            features = feature_engine.transform(df)

            # Create dataset
            sequence_length = config.PREDICTIONS['data']['sequence_length']
            dataset = TimeSeriesDataset(
                data=features,
                targets=features[:, 0],  # Close price
                sequence_length=sequence_length,
                forecast_horizon=self.forecast_horizon,
            )

            if len(dataset) == 0:
                return None

            # Get the last sequence for prediction
            last_sequence = dataset[-1][0].unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                prediction = model(last_sequence)
                predicted_price = prediction.cpu().numpy()[0, 0]

            # Calculate predicted change
            predicted_change_pct = ((predicted_price - current_price) / current_price) * 100

            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'predicted_change_pct': float(predicted_change_pct),
                'forecast_horizon': self.forecast_horizon,
                'model_type': self.model_type,
            }

        except Exception as e:
            print(f"Error predicting for {symbol}: {e}")
            return None

    def predict_batch(self, symbols, save_predictions=True):
        """
        Make predictions for multiple symbols.

        Args:
            symbols: List of stock symbols
            save_predictions: Whether to save to database (default True)

        Returns:
            DataFrame with predictions sorted by predicted change %
        """
        predictions = []

        for symbol in symbols:
            print(f"Predicting for {symbol}...")
            prediction = self.predict_for_symbol(symbol)

            if prediction:
                predictions.append(prediction)

                # Save to tracker
                if save_predictions:
                    self.tracker.record_prediction(
                        symbol=prediction['symbol'],
                        model_type=prediction['model_type'],
                        current_price=prediction['current_price'],
                        predicted_price=prediction['predicted_price'],
                        forecast_horizon=prediction['forecast_horizon']
                    )

        if not predictions:
            return pd.DataFrame()

        # Convert to DataFrame and sort
        df = pd.DataFrame(predictions)
        df = df.sort_values('predicted_change_pct', ascending=False)

        return df

    def get_top_movers_prediction(self, symbols, top_n=5):
        """
        Predict top winners and losers from a list of symbols.

        Args:
            symbols: List of stock symbols
            top_n: Number of winners/losers to identify

        Returns:
            dict with 'winners' and 'losers' DataFrames
        """
        # Make batch predictions
        predictions_df = self.predict_batch(symbols)

        if len(predictions_df) == 0:
            return {'winners': pd.DataFrame(), 'losers': pd.DataFrame()}

        # Get top winners and losers
        winners = predictions_df.head(top_n).copy()
        losers = predictions_df.tail(top_n).copy()
        losers = losers.sort_values('predicted_change_pct')

        return {
            'winners': winners,
            'losers': losers,
            'all_predictions': predictions_df
        }

    def update_predictions_with_actuals(self, prediction_date=None):
        """
        Update predictions with actual results for learning.

        Args:
            prediction_date: Date to update (default: today)

        Returns:
            Number of predictions updated
        """
        return self.tracker.update_predictions_for_date(prediction_date)

    def analyze_model_performance(self, days=30):
        """
        Analyze model performance over time.

        Args:
            days: Number of days to analyze

        Returns:
            dict with performance metrics and insights
        """
        metrics = self.tracker.calculate_model_performance(self.model_type)

        if not metrics:
            return None

        # Generate improvement insights
        insights = self.tracker.generate_improvement_insights(self.model_type)

        # Get performance history
        history = self.tracker.get_performance_history(self.model_type, days)

        return {
            'current_metrics': metrics,
            'insights': insights,
            'history': history
        }

    def get_model_recommendations(self):
        """
        Get recommendations for improving the model based on performance.

        Returns:
            list of recommendation dicts
        """
        insights_df = self.tracker.get_insights(self.model_type)

        if len(insights_df) == 0:
            return []

        recommendations = []
        for _, insight in insights_df.iterrows():
            recommendations.append({
                'type': insight['insight_type'],
                'description': insight['insight_description'],
                'action': insight['suggested_action'],
                'priority': insight['priority'],
                'date': insight['insight_date']
            })

        return recommendations

    def retrain_with_feedback(self, symbol, use_recent_performance=True):
        """
        Retrain model for a symbol using recent performance feedback.

        This adjusts training parameters based on recent prediction accuracy.

        Args:
            symbol: Stock symbol to retrain
            use_recent_performance: Whether to adjust based on performance

        Returns:
            dict with retraining results
        """
        if use_recent_performance:
            # Get performance metrics
            metrics = self.tracker.calculate_model_performance(
                self.model_type,
                symbol=symbol
            )

            if metrics and metrics['total_predictions'] > 10:
                # Adjust config based on performance
                adjustments = {}

                # If direction accuracy is low, add more features
                if metrics['direction_accuracy'] < 55:
                    adjustments['add_features'] = True
                    adjustments['increase_sequence_length'] = True

                # If MAE is high, increase model capacity
                if metrics['mean_absolute_error'] > 5:
                    adjustments['increase_hidden_dim'] = True
                    adjustments['train_longer'] = True

                return {
                    'symbol': symbol,
                    'current_performance': metrics,
                    'suggested_adjustments': adjustments,
                    'ready_for_retrain': True
                }

        return {
            'symbol': symbol,
            'ready_for_retrain': False,
            'reason': 'Insufficient performance data'
        }

    def close(self):
        """Close tracker connection."""
        self.tracker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
