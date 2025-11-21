"""Time-series dataset implementation"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Union, List


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series data with windowing support.

    Args:
        data: Input data (N, features) or (N,) for univariate
        targets: Target data (N,) or (N, target_features)
        sequence_length: Length of input sequences
        forecast_horizon: Number of steps to predict ahead (default: 1)
        stride: Stride for creating sequences (default: 1)
        target_offset: Offset for target alignment (default: 0, means targets are sequence_length steps ahead)
        scaler: Optional scaler object with transform/inverse_transform methods
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        stride: int = 1,
        target_offset: int = 0,
        scaler: Optional[object] = None,
    ):
        self.data = self._to_tensor(data)
        self.targets = self._to_tensor(targets) if targets is not None else None
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.target_offset = target_offset
        self.scaler = scaler

        # Ensure 2D data
        if self.data.dim() == 1:
            self.data = self.data.unsqueeze(-1)

        if self.targets is not None and self.targets.dim() == 1:
            self.targets = self.targets.unsqueeze(-1)

        # Calculate valid indices
        self._calculate_valid_indices()

    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor, None]) -> Optional[torch.Tensor]:
        """Convert numpy array to torch tensor"""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        return data.float()

    def _calculate_valid_indices(self):
        """Calculate valid starting indices for sequences"""
        n_samples = len(self.data)
        max_end_idx = n_samples - self.sequence_length - self.forecast_horizon - self.target_offset + 1

        if max_end_idx <= 0:
            raise ValueError(
                f"Not enough data for sequence_length={self.sequence_length}, "
                f"forecast_horizon={self.forecast_horizon}, and target_offset={self.target_offset}"
            )

        self.valid_indices = list(range(0, max_end_idx, self.stride))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a sequence and its target

        Returns:
            sequence: (sequence_length, n_features)
            target: (forecast_horizon, n_target_features) or None
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        sequence = self.data[start_idx:end_idx]

        if self.targets is not None:
            target_start = end_idx + self.target_offset
            target_end = target_start + self.forecast_horizon
            target = self.targets[target_start:target_end]

            # If forecast_horizon is 1, squeeze the time dimension
            if self.forecast_horizon == 1:
                target = target.squeeze(0)

            return sequence, target

        return sequence, None

    def get_raw_data(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get data without any transformations"""
        return self.__getitem__(idx)

    def inverse_transform_target(self, target: torch.Tensor) -> torch.Tensor:
        """Inverse transform target using the scaler"""
        if self.scaler is None:
            return target

        # Handle batch dimension
        original_shape = target.shape
        target_np = target.cpu().numpy().reshape(-1, target.shape[-1])
        inverse = self.scaler.inverse_transform(target_np)
        return torch.from_numpy(inverse).reshape(original_shape).to(target.device)

    @property
    def n_features(self) -> int:
        """Number of input features"""
        return self.data.shape[-1]

    @property
    def n_targets(self) -> int:
        """Number of target features"""
        if self.targets is None:
            return 0
        return self.targets.shape[-1]
