"""Base model for time-series prediction"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseTimeSeriesModel(nn.Module, ABC):
    """
    Abstract base class for time-series models

    All models should inherit from this class and implement the forward method.
    This provides a consistent interface across different architectures.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Number of output features
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, forecast_horizon, output_dim)
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
        }

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config(),
        }, path)

    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint.get('config', {})
