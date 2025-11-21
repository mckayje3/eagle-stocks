"""Time-series cross-validation strategies"""

import numpy as np
from typing import Generator, Tuple, Optional


class TimeSeriesSplit:
    """
    Time-series cross-validation with expanding window

    This respects temporal order by ensuring training data always comes before test data.
    Each fold adds more training data while keeping test size consistent.

    Args:
        n_splits: Number of splits (default: 5)
        test_size: Size of test set in each split
        gap: Number of samples to skip between train and test (default: 0)
    """

    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices

        Args:
            X: Input data (only used to determine length)

        Yields:
            train_indices, test_indices
        """
        n_samples = len(X)

        # Calculate test size if not provided
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # Ensure we have enough data
        min_train_size = test_size
        if n_samples < min_train_size + (self.n_splits * test_size) + (self.n_splits * self.gap):
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits "
                f"with test_size={test_size} and gap={self.gap}"
            )

        # Generate splits
        for i in range(1, self.n_splits + 1):
            # Test indices
            test_end = n_samples - (self.n_splits - i) * test_size
            test_start = test_end - test_size

            # Train indices (everything before test, minus gap)
            train_end = test_start - self.gap
            train_start = 0

            if train_end <= train_start:
                raise ValueError(
                    f"Not enough training data for split {i}. "
                    f"Consider reducing n_splits, test_size, or gap."
                )

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Get number of splits"""
        return self.n_splits


class WalkForwardSplit:
    """
    Walk-forward validation (rolling window)

    Uses a fixed-size training window that slides forward in time.
    This is useful when you want consistent training set size across folds.

    Args:
        n_splits: Number of splits (default: 5)
        train_size: Size of training window
        test_size: Size of test window
        gap: Number of samples to skip between train and test (default: 0)
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices

        Args:
            X: Input data (only used to determine length)

        Yields:
            train_indices, test_indices
        """
        n_samples = len(X)

        # Calculate sizes if not provided
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 2)
        else:
            test_size = self.test_size

        if self.train_size is None:
            train_size = n_samples // 2
        else:
            train_size = self.train_size

        # Calculate step size
        total_window = train_size + self.gap + test_size
        remaining = n_samples - total_window
        step = max(1, remaining // (self.n_splits - 1)) if self.n_splits > 1 else 0

        # Ensure we have enough data
        if total_window > n_samples:
            raise ValueError(
                f"Not enough samples ({n_samples}) for train_size={train_size}, "
                f"test_size={test_size}, and gap={self.gap}"
            )

        # Generate splits
        for i in range(self.n_splits):
            # Calculate window position
            start = i * step
            train_end = start + train_size

            # Check if we've exceeded data length
            if train_end + self.gap + test_size > n_samples:
                break

            # Train indices
            train_indices = np.arange(start, train_end)

            # Test indices
            test_start = train_end + self.gap
            test_end = test_start + test_size
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Get number of splits"""
        return self.n_splits


def train_test_split_temporal(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    gap: int = 0,
) -> Tuple:
    """
    Simple temporal train/test split

    Args:
        X: Input features
        y: Target values (optional)
        test_size: Proportion of data for test set (default: 0.2)
        gap: Number of samples to skip between train and test (default: 0)

    Returns:
        X_train, X_test, y_train (if y provided), y_test (if y provided)
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    train_end = n_samples - test_samples - gap

    if train_end <= 0:
        raise ValueError(f"Not enough data for test_size={test_size} and gap={gap}")

    X_train = X[:train_end]
    X_test = X[train_end + gap:]

    if y is not None:
        y_train = y[:train_end]
        y_test = y[train_end + gap:]
        return X_train, X_test, y_train, y_test

    return X_train, X_test
