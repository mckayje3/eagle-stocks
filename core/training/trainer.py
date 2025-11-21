"""Training loop implementation"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any, Callable
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Trainer for time-series models

    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on ('cuda' or 'cpu')
        callbacks: List of callback functions
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        callbacks: Optional[List] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks or []

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }

        self.current_epoch = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        metrics: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            metrics: Dictionary of metric name -> metric function

        Returns:
            Dictionary of average loss and metrics
        """
        self.model.train()
        total_loss = 0
        metric_values = {name: 0 for name in (metrics or {})}
        n_batches = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')

        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move to device
            data = data.to(self.device)
            target = target.to(self.device) if target is not None else None

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)

            # Calculate loss
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            n_batches += 1

            if metrics:
                with torch.no_grad():
                    for name, metric_fn in metrics.items():
                        metric_values[name] += metric_fn(output, target).item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / n_batches,
                **{k: v / n_batches for k, v in metric_values.items()}
            })

        # Calculate averages
        results = {
            'loss': total_loss / n_batches,
            **{k: v / n_batches for k, v in metric_values.items()}
        }

        return results

    def validate(
        self,
        val_loader: DataLoader,
        metrics: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, float]:
        """
        Validate the model

        Args:
            val_loader: Validation data loader
            metrics: Dictionary of metric name -> metric function

        Returns:
            Dictionary of average loss and metrics
        """
        self.model.eval()
        total_loss = 0
        metric_values = {name: 0 for name in (metrics or {})}
        n_batches = 0

        progress_bar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(progress_bar):
                # Move to device
                data = data.to(self.device)
                target = target.to(self.device) if target is not None else None

                # Forward pass
                output = self.model(data)

                # Calculate loss
                loss = self.criterion(output, target)

                # Track metrics
                total_loss += loss.item()
                n_batches += 1

                if metrics:
                    for name, metric_fn in metrics.items():
                        metric_values[name] += metric_fn(output, target).item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': total_loss / n_batches,
                    **{k: v / n_batches for k, v in metric_values.items()}
                })

        # Calculate averages
        results = {
            'loss': total_loss / n_batches,
            **{k: v / n_batches for k, v in metric_values.items()}
        }

        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        metrics: Optional[Dict[str, Callable]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            metrics: Dictionary of metric name -> metric function
            verbose: Whether to print progress

        Returns:
            Training history
        """
        for epoch in range(epochs):
            self.current_epoch = epoch

            # Call on_epoch_begin callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(epoch, self)

            # Train
            train_results = self.train_epoch(train_loader, metrics)
            self.history['train_loss'].append(train_results['loss'])

            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {train_results['loss']:.4f}")
                for name, value in train_results.items():
                    if name != 'loss':
                        print(f"  Train {name}: {value:.4f}")

            # Validate
            if val_loader is not None:
                val_results = self.validate(val_loader, metrics)
                self.history['val_loss'].append(val_results['loss'])

                if verbose:
                    print(f"  Val Loss: {val_results['loss']:.4f}")
                    for name, value in val_results.items():
                        if name != 'loss':
                            print(f"  Val {name}: {value:.4f}")

            # Call on_epoch_end callbacks
            stop_training = False
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    if callback.on_epoch_end(epoch, self, train_results, val_results if val_loader else None):
                        stop_training = True

            if stop_training:
                if verbose:
                    print("\nEarly stopping triggered")
                break

        return self.history

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions

        Args:
            data_loader: Data loader

        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc='Predicting'):
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
