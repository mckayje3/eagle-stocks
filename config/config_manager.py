"""Configuration management for projects"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration management class

    Supports loading from YAML or JSON files and provides easy access to nested configs.

    Example:
        config = Config.from_yaml('config.yaml')
        batch_size = config.get('training.batch_size', default=32)
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create from dictionary"""
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Key in dot notation (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
            key: Key in dot notation (e.g., 'training.batch_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._config.copy()

    def to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def to_json(self, filepath: str, indent: int = 2):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=indent)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"Config({self._config})"


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration template

    Returns:
        Default configuration dictionary
    """
    return {
        'project': {
            'name': 'time_series_project',
            'description': 'Time-series deep learning project',
            'seed': 42,
        },
        'data': {
            'sequence_length': 30,
            'forecast_horizon': 1,
            'batch_size': 32,
            'test_size': 0.2,
            'val_size': 0.1,
        },
        'features': {
            'scaler': 'standard',  # 'standard', 'minmax', 'robust', or null
            'handle_missing': 'ffill',  # 'ffill', 'bfill', 'drop', or 'zero'
            'technical_indicators': {
                'enabled': False,
                'include_rsi': True,
                'include_macd': True,
                'include_bollinger': True,
            },
            'datetime_features': {
                'enabled': False,
                'include_cyclical': True,
            },
        },
        'model': {
            'type': 'lstm',  # 'lstm', 'gru', or 'transformer'
            'input_dim': None,  # Will be inferred from data
            'hidden_dim': 64,
            'output_dim': 1,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': False,  # For LSTM/GRU
            'num_heads': 4,  # For Transformer
            'dim_feedforward': 256,  # For Transformer
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',  # 'adam', 'sgd', 'adamw'
            'loss': 'mse',  # 'mse', 'mae', 'huber'
            'device': 'auto',  # 'auto', 'cuda', or 'cpu'
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.0001,
            },
            'checkpoint': {
                'enabled': True,
                'filepath': 'checkpoints/best_model.pt',
            },
            'lr_scheduler': {
                'enabled': False,
                'type': 'reduce_on_plateau',  # 'reduce_on_plateau', 'step', 'exponential'
                'patience': 5,
                'factor': 0.5,
            },
        },
        'validation': {
            'method': 'simple_split',  # 'simple_split', 'time_series_cv', 'walk_forward'
            'n_splits': 5,
            'gap': 0,
        },
    }


def save_default_config(filepath: str, format: str = 'yaml'):
    """
    Save default configuration to file

    Args:
        filepath: Path to save configuration
        format: File format ('yaml' or 'json')
    """
    config = Config.from_dict(create_default_config())

    if format == 'yaml':
        config.to_yaml(filepath)
    elif format == 'json':
        config.to_json(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == '__main__':
    # Generate default config files
    save_default_config('config_template.yaml', format='yaml')
    save_default_config('config_template.json', format='json')
    print("Default configuration files created!")
