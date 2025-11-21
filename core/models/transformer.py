"""Transformer model for time-series prediction"""

import torch
import torch.nn as nn
import math
from .base_model import BaseTimeSeriesModel


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


class TransformerModel(BaseTimeSeriesModel):
    """
    Transformer-based model for time-series prediction

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden/model dimension (d_model)
        output_dim: Number of output features
        num_layers: Number of transformer encoder layers (default: 1)
        num_heads: Number of attention heads (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout rate (default: 0.1)
        forecast_horizon: Number of steps to predict (default: 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        forecast_horizon: int = 1,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)

        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.forecast_horizon = forecast_horizon

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim * forecast_horizon)

    def forward(self, x: torch.Tensor, src_mask=None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            src_mask: Optional mask for attention

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)

        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x, src_mask)  # (batch_size, seq_len, hidden_dim)

        # Take the last output
        x = x[:, -1, :]  # (batch_size, hidden_dim)

        # Apply dropout
        x = self.dropout_layer(x)

        # Output projection
        output = self.fc(x)  # (batch_size, output_dim * forecast_horizon)

        # Reshape if forecast_horizon > 1
        if self.forecast_horizon > 1:
            output = output.view(batch_size, self.forecast_horizon, self.output_dim)

        return output

    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'dim_feedforward': self.dim_feedforward,
            'forecast_horizon': self.forecast_horizon,
        })
        return config
