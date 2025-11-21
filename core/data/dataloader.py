"""Time-series specific data loader"""

from torch.utils.data import DataLoader
from typing import Optional


class TimeSeriesDataLoader(DataLoader):
    """
    DataLoader optimized for time-series data.

    Extends PyTorch's DataLoader with time-series specific defaults.
    By default, shuffle is False to maintain temporal order.

    Args:
        dataset: TimeSeriesDataset instance
        batch_size: Batch size (default: 32)
        shuffle: Whether to shuffle data. For time-series, typically False (default: False)
        num_workers: Number of worker processes (default: 0)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        drop_last: Whether to drop the last incomplete batch (default: False)
        **kwargs: Additional arguments passed to DataLoader
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )

    @property
    def n_features(self) -> int:
        """Number of input features"""
        return self.dataset.n_features

    @property
    def n_targets(self) -> int:
        """Number of target features"""
        return self.dataset.n_targets

    @property
    def sequence_length(self) -> int:
        """Sequence length"""
        return self.dataset.sequence_length
