"""Feature engineering pipeline"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureEngine:
    """
    Orchestrates feature engineering pipeline for time-series data

    Args:
        transformers: List of feature transformers to apply in sequence
        scaler: Type of scaler ('standard', 'minmax', 'robust', or None)
        handle_missing: How to handle missing values ('ffill', 'bfill', 'drop', or 'zero')
    """

    def __init__(
        self,
        transformers: Optional[List] = None,
        scaler: Optional[str] = 'standard',
        handle_missing: str = 'ffill',
    ):
        self.transformers = transformers or []
        self.scaler_type = scaler
        self.handle_missing = handle_missing
        self.scaler = None
        self.fitted = False
        self.feature_names = None

        # Initialize scaler
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler == 'robust':
            self.scaler = RobustScaler()
        elif scaler is not None:
            raise ValueError(f"Unknown scaler type: {scaler}")

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to strategy"""
        if self.handle_missing == 'ffill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif self.handle_missing == 'bfill':
            return data.fillna(method='bfill').fillna(method='ffill')
        elif self.handle_missing == 'drop':
            return data.dropna()
        elif self.handle_missing == 'zero':
            return data.fillna(0)
        else:
            raise ValueError(f"Unknown missing value strategy: {self.handle_missing}")

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> 'FeatureEngine':
        """
        Fit the feature engineering pipeline

        Args:
            data: Input data to fit on

        Returns:
            self
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Apply transformers
        transformed = data.copy()
        for transformer in self.transformers:
            if hasattr(transformer, 'fit'):
                transformer.fit(transformed)
            transformed = transformer.transform(transformed) if hasattr(transformer, 'transform') else transformed

        # Handle missing values
        transformed = self._handle_missing_values(transformed)

        # Store feature names
        self.feature_names = transformed.columns.tolist()

        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(transformed.values)

        self.fitted = True
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted pipeline

        Args:
            data: Input data to transform

        Returns:
            Transformed data as numpy array
        """
        if not self.fitted:
            raise RuntimeError("FeatureEngine must be fitted before transform. Call fit() first.")

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Apply transformers
        transformed = data.copy()
        for transformer in self.transformers:
            transformed = transformer.transform(transformed) if hasattr(transformer, 'transform') else transformed

        # Handle missing values
        transformed = self._handle_missing_values(transformed)

        # Ensure same columns as training
        if self.feature_names is not None:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in transformed.columns:
                    transformed[col] = 0

            # Reorder to match training
            transformed = transformed[self.feature_names]

        # Apply scaler
        if self.scaler is not None:
            return self.scaler.transform(transformed.values)

        return transformed.values

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit and transform data in one step

        Args:
            data: Input data

        Returns:
            Transformed data as numpy array
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Apply transformers
        transformed = data.copy()
        for transformer in self.transformers:
            if hasattr(transformer, 'fit_transform'):
                transformed = transformer.fit_transform(transformed)
            elif hasattr(transformer, 'transform'):
                transformed = transformer.transform(transformed)

        # Handle missing values
        transformed = self._handle_missing_values(transformed)

        # Store feature names
        self.feature_names = transformed.columns.tolist()

        # Fit and transform with scaler
        if self.scaler is not None:
            result = self.scaler.fit_transform(transformed.values)
        else:
            result = transformed.values

        self.fitted = True
        return result

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data

        Args:
            data: Scaled data

        Returns:
            Original scale data
        """
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data

    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation"""
        if self.feature_names is None:
            raise RuntimeError("FeatureEngine must be fitted first")
        return self.feature_names

    def add_transformer(self, transformer) -> 'FeatureEngine':
        """Add a transformer to the pipeline"""
        self.transformers.append(transformer)
        self.fitted = False  # Need to refit
        return self
