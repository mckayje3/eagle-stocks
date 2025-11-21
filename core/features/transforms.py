"""Time-series feature transformations"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union


class RollingWindow:
    """
    Compute rolling window statistics

    Args:
        windows: List of window sizes
        functions: List of functions to apply ('mean', 'std', 'min', 'max', 'sum')
    """

    def __init__(self, windows: List[int], functions: List[str] = ['mean', 'std']):
        self.windows = windows
        self.functions = functions
        self.function_map = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'sum': np.sum,
        }

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Apply rolling window transformations"""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        result = data.copy()

        for col in data.columns:
            for window in self.windows:
                for func_name in self.functions:
                    if func_name not in self.function_map:
                        raise ValueError(f"Unknown function: {func_name}")

                    func = self.function_map[func_name]
                    result[f'{col}_rolling_{window}_{func_name}'] = (
                        data[col].rolling(window=window, min_periods=1).apply(func, raw=True)
                    )

        return result

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Fit and transform (no fitting needed for rolling windows)"""
        return self.transform(data)


class LagFeatures:
    """
    Create lag features

    Args:
        lags: List of lag periods
        columns: Specific columns to create lags for (None = all columns)
    """

    def __init__(self, lags: List[int], columns: Optional[List[str]] = None):
        self.lags = lags
        self.columns = columns

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Create lag features"""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        result = data.copy()
        target_cols = self.columns if self.columns is not None else data.columns

        for col in target_cols:
            for lag in self.lags:
                result[f'{col}_lag_{lag}'] = data[col].shift(lag)

        return result

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Fit and transform (no fitting needed for lags)"""
        return self.transform(data)


class TechnicalIndicators:
    """
    Compute technical indicators commonly used in finance

    Args:
        include_rsi: Include Relative Strength Index
        include_macd: Include Moving Average Convergence Divergence
        include_bollinger: Include Bollinger Bands
        rsi_period: Period for RSI calculation (default: 14)
        macd_fast: Fast period for MACD (default: 12)
        macd_slow: Slow period for MACD (default: 26)
        macd_signal: Signal period for MACD (default: 9)
        bollinger_window: Window for Bollinger Bands (default: 20)
        bollinger_std: Standard deviations for Bollinger Bands (default: 2)
    """

    def __init__(
        self,
        include_rsi: bool = True,
        include_macd: bool = True,
        include_bollinger: bool = True,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_window: int = 20,
        bollinger_std: float = 2,
    ):
        self.include_rsi = include_rsi
        self.include_macd = include_macd
        self.include_bollinger = include_bollinger
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = prices.ewm(span=slow, adjust=False, min_periods=1).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2
    ) -> tuple:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()

        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        return upper_band, rolling_mean, lower_band

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Apply technical indicators"""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])

        result = data.copy()

        # Assume first column is price (or apply to all numeric columns)
        for col in data.select_dtypes(include=[np.number]).columns:
            if self.include_rsi:
                result[f'{col}_rsi'] = self._calculate_rsi(data[col], self.rsi_period)

            if self.include_macd:
                macd, macd_signal, macd_hist = self._calculate_macd(
                    data[col], self.macd_fast, self.macd_slow, self.macd_signal
                )
                result[f'{col}_macd'] = macd
                result[f'{col}_macd_signal'] = macd_signal
                result[f'{col}_macd_hist'] = macd_hist

            if self.include_bollinger:
                upper, middle, lower = self._calculate_bollinger_bands(
                    data[col], self.bollinger_window, self.bollinger_std
                )
                result[f'{col}_bb_upper'] = upper
                result[f'{col}_bb_middle'] = middle
                result[f'{col}_bb_lower'] = lower

        return result

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Fit and transform"""
        return self.transform(data)


class DateTimeFeatures:
    """
    Extract datetime features from timestamp index or column

    Args:
        include_cyclical: Include cyclical encoding (sin/cos) for periodic features
        include_day_of_week: Extract day of week
        include_month: Extract month
        include_hour: Extract hour (if timestamp has time)
        include_is_weekend: Binary weekend indicator
    """

    def __init__(
        self,
        include_cyclical: bool = True,
        include_day_of_week: bool = True,
        include_month: bool = True,
        include_hour: bool = True,
        include_is_weekend: bool = True,
    ):
        self.include_cyclical = include_cyclical
        self.include_day_of_week = include_day_of_week
        self.include_month = include_month
        self.include_hour = include_hour
        self.include_is_weekend = include_is_weekend

    def transform(self, data: Union[pd.DataFrame, pd.DatetimeIndex]) -> pd.DataFrame:
        """Extract datetime features"""
        if isinstance(data, pd.DataFrame):
            # Assume index is datetime
            dt_index = data.index
            result = data.copy()
        else:
            dt_index = data
            result = pd.DataFrame(index=dt_index)

        if not isinstance(dt_index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex or be a DatetimeIndex")

        if self.include_day_of_week:
            result['day_of_week'] = dt_index.dayofweek

            if self.include_cyclical:
                result['day_of_week_sin'] = np.sin(2 * np.pi * dt_index.dayofweek / 7)
                result['day_of_week_cos'] = np.cos(2 * np.pi * dt_index.dayofweek / 7)

        if self.include_month:
            result['month'] = dt_index.month

            if self.include_cyclical:
                result['month_sin'] = np.sin(2 * np.pi * dt_index.month / 12)
                result['month_cos'] = np.cos(2 * np.pi * dt_index.month / 12)

        if self.include_hour and hasattr(dt_index, 'hour'):
            result['hour'] = dt_index.hour

            if self.include_cyclical:
                result['hour_sin'] = np.sin(2 * np.pi * dt_index.hour / 24)
                result['hour_cos'] = np.cos(2 * np.pi * dt_index.hour / 24)

        if self.include_is_weekend:
            result['is_weekend'] = (dt_index.dayofweek >= 5).astype(int)

        return result

    def fit_transform(self, data: Union[pd.DataFrame, pd.DatetimeIndex]) -> pd.DataFrame:
        """Fit and transform"""
        return self.transform(data)
