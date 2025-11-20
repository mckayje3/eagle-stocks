"""
Prediction Tracker - Track model predictions and learn from performance

This module tracks daily predictions, compares them with actual outcomes,
and provides insights for model improvement.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import config


class PredictionTracker:
    """Track and analyze prediction performance over time."""

    def __init__(self, db_path="data/prediction_tracker.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the prediction tracking database."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Daily predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date DATE NOT NULL,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                current_price REAL NOT NULL,
                predicted_price REAL NOT NULL,
                predicted_change_pct REAL NOT NULL,
                forecast_horizon INTEGER NOT NULL,
                target_date DATE NOT NULL,
                actual_price REAL,
                actual_change_pct REAL,
                prediction_error REAL,
                accuracy_score REAL,
                is_winner INTEGER,
                is_loser INTEGER,
                rank_predicted INTEGER,
                rank_actual INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(prediction_date, symbol, model_type, forecast_horizon)
            )
        """)

        # Model performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_date DATE NOT NULL,
                model_type TEXT NOT NULL,
                symbol TEXT,
                total_predictions INTEGER NOT NULL,
                correct_direction INTEGER NOT NULL,
                direction_accuracy REAL NOT NULL,
                mean_absolute_error REAL NOT NULL,
                mean_squared_error REAL NOT NULL,
                top_movers_accuracy REAL,
                winner_precision REAL,
                loser_precision REAL,
                avg_prediction_error REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(evaluation_date, model_type, symbol)
            )
        """)

        # Model improvement insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_date DATE NOT NULL,
                model_type TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                insight_description TEXT NOT NULL,
                suggested_action TEXT,
                priority INTEGER DEFAULT 1,
                implemented INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Top movers tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS top_movers_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movement_date DATE NOT NULL,
                symbol TEXT NOT NULL,
                open_price REAL NOT NULL,
                close_price REAL NOT NULL,
                change_pct REAL NOT NULL,
                volume INTEGER,
                is_winner INTEGER DEFAULT 0,
                is_loser INTEGER DEFAULT 0,
                rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(movement_date, symbol)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_date
            ON daily_predictions(prediction_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol
            ON daily_predictions(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_target_date
            ON daily_predictions(target_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_movers_date
            ON top_movers_daily(movement_date)
        """)

        self.conn.commit()

    def record_prediction(self, symbol, model_type, current_price, predicted_price,
                         forecast_horizon=1, prediction_date=None):
        """
        Record a daily prediction for a stock.

        Args:
            symbol: Stock symbol
            model_type: Model used (lstm, gru, transformer)
            current_price: Current stock price
            predicted_price: Predicted future price
            forecast_horizon: Days ahead (default 1)
            prediction_date: Date of prediction (default today)
        """
        if prediction_date is None:
            prediction_date = datetime.now().date()

        target_date = prediction_date + timedelta(days=forecast_horizon)
        predicted_change_pct = ((predicted_price - current_price) / current_price) * 100

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO daily_predictions (
                prediction_date, symbol, model_type, current_price,
                predicted_price, predicted_change_pct, forecast_horizon,
                target_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_date, symbol, model_type.lower(), current_price,
            predicted_price, predicted_change_pct, forecast_horizon, target_date
        ))
        self.conn.commit()

        return cursor.lastrowid

    def update_prediction_actual(self, prediction_id, actual_price):
        """
        Update a prediction with actual outcome.

        Args:
            prediction_id: ID of the prediction
            actual_price: Actual price at target date
        """
        cursor = self.conn.cursor()

        # Get the prediction
        cursor.execute("""
            SELECT current_price, predicted_price FROM daily_predictions
            WHERE id = ?
        """, (prediction_id,))

        result = cursor.fetchone()
        if not result:
            return False

        current_price, predicted_price = result
        actual_change_pct = ((actual_price - current_price) / current_price) * 100
        prediction_error = abs(predicted_price - actual_price)

        # Calculate accuracy score (inverse of error, capped at 100%)
        accuracy_score = max(0, 100 - (prediction_error / actual_price * 100))

        cursor.execute("""
            UPDATE daily_predictions
            SET actual_price = ?,
                actual_change_pct = ?,
                prediction_error = ?,
                accuracy_score = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (actual_price, actual_change_pct, prediction_error, accuracy_score, prediction_id))

        self.conn.commit()
        return True

    def update_predictions_for_date(self, target_date=None):
        """
        Update all predictions where target_date has arrived with actual prices.

        Args:
            target_date: Date to update (default: today)
        """
        if target_date is None:
            target_date = datetime.now().date()

        from utils.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, symbol, current_price, target_date
            FROM daily_predictions
            WHERE target_date = ? AND actual_price IS NULL
        """, (target_date,))

        predictions = cursor.fetchall()
        updated = 0

        for pred_id, symbol, current_price, _ in predictions:
            try:
                # Get actual price for target date
                data = fetcher.get_stock_data(symbol, period="5d")
                if data is not None and len(data) > 0:
                    # Get the close price for target date
                    target_data = data[data.index.date == target_date]
                    if len(target_data) > 0:
                        actual_price = target_data['Close'].iloc[-1]
                        self.update_prediction_actual(pred_id, actual_price)
                        updated += 1
            except Exception as e:
                print(f"Error updating prediction for {symbol}: {e}")

        return updated

    def get_top_predictions(self, prediction_date=None, top_n=5):
        """
        Get top predicted winners and losers for a date.

        Args:
            prediction_date: Date of predictions (default: today)
            top_n: Number of winners/losers to return (default: 5)

        Returns:
            dict with 'winners' and 'losers' DataFrames
        """
        if prediction_date is None:
            prediction_date = datetime.now().date()

        query = """
            SELECT symbol, model_type, current_price, predicted_price,
                   predicted_change_pct, forecast_horizon, target_date,
                   actual_price, actual_change_pct, accuracy_score
            FROM daily_predictions
            WHERE prediction_date = ?
            ORDER BY predicted_change_pct DESC
        """

        df = pd.read_sql_query(query, self.conn, params=(prediction_date,))

        if len(df) == 0:
            return {'winners': pd.DataFrame(), 'losers': pd.DataFrame()}

        winners = df.head(top_n)
        losers = df.tail(top_n).sort_values('predicted_change_pct')

        return {'winners': winners, 'losers': losers}

    def calculate_model_performance(self, model_type, start_date=None, end_date=None, symbol=None):
        """
        Calculate performance metrics for a model.

        Args:
            model_type: Type of model (lstm, gru, transformer)
            start_date: Start of evaluation period (default: 30 days ago)
            end_date: End of evaluation period (default: today)
            symbol: Specific symbol to evaluate (optional)

        Returns:
            dict with performance metrics
        """
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Build query
        query = """
            SELECT
                COUNT(*) as total,
                AVG(CASE
                    WHEN (predicted_change_pct > 0 AND actual_change_pct > 0) OR
                         (predicted_change_pct < 0 AND actual_change_pct < 0)
                    THEN 1 ELSE 0 END) as direction_accuracy,
                AVG(ABS(predicted_price - actual_price)) as mae,
                AVG((predicted_price - actual_price) * (predicted_price - actual_price)) as mse,
                AVG(prediction_error) as avg_error,
                AVG(accuracy_score) as avg_accuracy
            FROM daily_predictions
            WHERE model_type = ?
                AND target_date BETWEEN ? AND ?
                AND actual_price IS NOT NULL
        """
        params = [model_type.lower(), start_date, end_date]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()

        if not result or result[0] == 0:
            return None

        metrics = {
            'model_type': model_type,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'total_predictions': result[0],
            'direction_accuracy': result[1] * 100 if result[1] else 0,
            'mean_absolute_error': result[2] if result[2] else 0,
            'mean_squared_error': result[3] if result[3] else 0,
            'avg_prediction_error': result[4] if result[4] else 0,
            'avg_accuracy_score': result[5] if result[5] else 0,
        }

        # Calculate top movers accuracy
        top_movers_acc = self._calculate_top_movers_accuracy(
            model_type, start_date, end_date
        )
        metrics['top_movers_accuracy'] = top_movers_acc

        return metrics

    def _calculate_top_movers_accuracy(self, model_type, start_date, end_date, top_n=5):
        """Calculate how well the model predicts top movers."""
        cursor = self.conn.cursor()

        # Get all predictions with actuals for the period
        cursor.execute("""
            SELECT prediction_date, symbol, predicted_change_pct, actual_change_pct
            FROM daily_predictions
            WHERE model_type = ?
                AND target_date BETWEEN ? AND ?
                AND actual_price IS NOT NULL
            ORDER BY prediction_date, predicted_change_pct DESC
        """, (model_type.lower(), start_date, end_date))

        predictions = cursor.fetchall()
        if not predictions:
            return 0

        # Group by date and calculate accuracy
        dates = {}
        for pred_date, symbol, pred_change, actual_change in predictions:
            if pred_date not in dates:
                dates[pred_date] = []
            dates[pred_date].append({
                'symbol': symbol,
                'predicted': pred_change,
                'actual': actual_change
            })

        total_accuracy = 0
        date_count = 0

        for pred_date, preds in dates.items():
            if len(preds) < top_n * 2:
                continue

            # Sort by predicted change
            preds_sorted = sorted(preds, key=lambda x: x['predicted'], reverse=True)
            predicted_winners = set([p['symbol'] for p in preds_sorted[:top_n]])
            predicted_losers = set([p['symbol'] for p in preds_sorted[-top_n:]])

            # Sort by actual change
            actual_sorted = sorted(preds, key=lambda x: x['actual'], reverse=True)
            actual_winners = set([p['symbol'] for p in actual_sorted[:top_n]])
            actual_losers = set([p['symbol'] for p in actual_sorted[-top_n:]])

            # Calculate accuracy
            winner_matches = len(predicted_winners & actual_winners)
            loser_matches = len(predicted_losers & actual_losers)
            accuracy = (winner_matches + loser_matches) / (top_n * 2) * 100

            total_accuracy += accuracy
            date_count += 1

        return total_accuracy / date_count if date_count > 0 else 0

    def save_performance_metrics(self, metrics):
        """Save performance metrics to database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO model_performance (
                evaluation_date, model_type, symbol, total_predictions,
                correct_direction, direction_accuracy, mean_absolute_error,
                mean_squared_error, top_movers_accuracy, avg_prediction_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics['end_date'],
            metrics['model_type'],
            metrics.get('symbol'),
            metrics['total_predictions'],
            int(metrics['total_predictions'] * metrics['direction_accuracy'] / 100),
            metrics['direction_accuracy'],
            metrics['mean_absolute_error'],
            metrics['mean_squared_error'],
            metrics.get('top_movers_accuracy', 0),
            metrics['avg_prediction_error']
        ))
        self.conn.commit()

    def generate_improvement_insights(self, model_type, min_predictions=20):
        """
        Analyze model performance and generate improvement insights.

        Args:
            model_type: Model to analyze
            min_predictions: Minimum predictions needed for analysis

        Returns:
            list of insight dictionaries
        """
        metrics = self.calculate_model_performance(model_type)

        if not metrics or metrics['total_predictions'] < min_predictions:
            return []

        insights = []
        today = datetime.now().date()

        # Direction accuracy insight
        if metrics['direction_accuracy'] < 55:
            insights.append({
                'date': today,
                'model_type': model_type,
                'type': 'low_direction_accuracy',
                'description': f"Direction accuracy is {metrics['direction_accuracy']:.1f}%, barely better than random (50%)",
                'action': "Consider adding more features (lag features, volatility indicators) or increasing sequence length",
                'priority': 1
            })
        elif metrics['direction_accuracy'] > 70:
            insights.append({
                'date': today,
                'model_type': model_type,
                'type': 'good_direction_accuracy',
                'description': f"Excellent direction accuracy at {metrics['direction_accuracy']:.1f}%",
                'action': "Model is performing well. Consider reducing regularization or increasing capacity.",
                'priority': 3
            })

        # MAE insight
        if metrics['mean_absolute_error'] > 5:
            insights.append({
                'date': today,
                'model_type': model_type,
                'type': 'high_mae',
                'description': f"Mean absolute error is high at ${metrics['mean_absolute_error']:.2f}",
                'action': "Try increasing model capacity (hidden_dim, num_layers) or training longer",
                'priority': 2
            })

        # Top movers accuracy
        if metrics.get('top_movers_accuracy', 0) < 30:
            insights.append({
                'date': today,
                'model_type': model_type,
                'type': 'low_top_movers_accuracy',
                'description': f"Top movers prediction accuracy is only {metrics.get('top_movers_accuracy', 0):.1f}%",
                'action': "Add volatility features, increase training data, or use ensemble methods",
                'priority': 1
            })

        # Save insights to database
        cursor = self.conn.cursor()
        for insight in insights:
            cursor.execute("""
                INSERT INTO model_insights (
                    insight_date, model_type, insight_type,
                    insight_description, suggested_action, priority
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                insight['date'], insight['model_type'], insight['type'],
                insight['description'], insight['action'], insight['priority']
            ))
        self.conn.commit()

        return insights

    def get_performance_history(self, model_type, days=30):
        """Get historical performance metrics for a model."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        query = """
            SELECT evaluation_date, direction_accuracy, mean_absolute_error,
                   top_movers_accuracy, total_predictions
            FROM model_performance
            WHERE model_type = ?
                AND evaluation_date BETWEEN ? AND ?
            ORDER BY evaluation_date
        """

        return pd.read_sql_query(query, self.conn, params=(model_type.lower(), start_date, end_date))

    def get_insights(self, model_type=None, implemented_only=False):
        """Get model improvement insights."""
        query = "SELECT * FROM model_insights WHERE 1=1"
        params = []

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type.lower())

        if implemented_only:
            query += " AND implemented = 1"

        query += " ORDER BY priority, insight_date DESC"

        return pd.read_sql_query(query, self.conn, params=params if params else None)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
