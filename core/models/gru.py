"""GRU model for time-series prediction"""

import torch
import torch.nn as nn
from .base_model import BaseTimeSeriesModel
from typing import Optional


class GRUModel(BaseTimeSeriesModel):
    """
    GRU-based model for time-series prediction

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension
        output_dim: Number of output features
        num_layers: Number of GRU layers (default: 1)
        dropout: Dropout rate (default: 0.1)
        bidirectional: Use bidirectional GRU (default: False)
        forecast_horizon: Number of steps to predict (default: 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        forecast_horizon: int = 1,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)

        self.bidirectional = bidirectional
        self.forecast_horizon = forecast_horizon
        self.num_directions = 2 if bidirectional else 1

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim * forecast_horizon)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden: Optional hidden state h0

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)

        # GRU forward pass
        gru_out, hidden = self.gru(x, hidden)

        # Take the last output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim * num_directions)

        # Apply dropout
        last_output = self.dropout_layer(last_output)

        # Fully connected layer
        output = self.fc(last_output)  # (batch_size, output_dim * forecast_horizon)

        # Reshape if forecast_horizon > 1
        if self.forecast_horizon > 1:
            output = output.view(batch_size, self.forecast_horizon, self.output_dim)

        return output

    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'bidirectional': self.bidirectional,
            'forecast_horizon': self.forecast_horizon,
        })
        return config
