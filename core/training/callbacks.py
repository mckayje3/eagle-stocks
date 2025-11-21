"""Training callbacks"""

import torch
import numpy as np
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving

    Args:
        patience: Number of epochs to wait for improvement (default: 5)
        min_delta: Minimum change to qualify as improvement (default: 0)
        mode: 'min' or 'max' - whether to minimize or maximize the metric (default: 'min')
        verbose: Whether to print messages (default: True)
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0,
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(
        self,
        epoch: int,
        trainer,
        train_results: Dict[str, float],
        val_results: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Called at the end of each epoch

        Args:
            epoch: Current epoch number
            trainer: Trainer instance
            train_results: Training metrics
            val_results: Validation metrics

        Returns:
            True if training should stop, False otherwise
        """
        if val_results is None:
            return False

        score = val_results['loss']

        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"  Validation loss improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class ModelCheckpoint:
    """
    Save model checkpoint when validation loss improves

    Args:
        filepath: Path to save checkpoint
        monitor: Metric to monitor (default: 'val_loss')
        mode: 'min' or 'max' - whether to minimize or maximize the metric (default: 'min')
        save_best_only: Only save when metric improves (default: True)
        verbose: Whether to print messages (default: True)
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_score = None

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(
        self,
        epoch: int,
        trainer,
        train_results: Dict[str, float],
        val_results: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Called at the end of each epoch

        Args:
            epoch: Current epoch number
            trainer: Trainer instance
            train_results: Training metrics
            val_results: Validation metrics

        Returns:
            False (never stops training)
        """
        # Determine which results to use
        results = val_results if val_results is not None else train_results

        # Extract score
        if self.monitor == 'val_loss':
            score = results.get('loss', None)
        elif self.monitor == 'train_loss':
            score = train_results.get('loss', None)
        else:
            score = results.get(self.monitor, None)

        if score is None:
            if self.verbose:
                print(f"  Warning: {self.monitor} not found in results")
            return False

        # Check if we should save
        should_save = False
        if not self.save_best_only:
            should_save = True
        elif self.best_score is None:
            should_save = True
            self.best_score = score
        elif self.monitor_op(score, self.best_score):
            should_save = True
            self.best_score = score

        if should_save:
            trainer.save_checkpoint(self.filepath)
            if self.verbose:
                print(f"  Model checkpoint saved to {self.filepath}")

        return False


class LearningRateScheduler:
    """
    Adjust learning rate based on schedule

    Args:
        scheduler: PyTorch learning rate scheduler
        verbose: Whether to print messages (default: True)
    """

    def __init__(self, scheduler, verbose: bool = True):
        self.scheduler = scheduler
        self.verbose = verbose

    def on_epoch_end(
        self,
        epoch: int,
        trainer,
        train_results: Dict[str, float],
        val_results: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Called at the end of each epoch

        Args:
            epoch: Current epoch number
            trainer: Trainer instance
            train_results: Training metrics
            val_results: Validation metrics

        Returns:
            False (never stops training)
        """
        # Check if scheduler needs validation loss
        if hasattr(self.scheduler, 'step'):
            if 'ReduceLROnPlateau' in self.scheduler.__class__.__name__:
                if val_results is not None:
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step(train_results['loss'])
            else:
                self.scheduler.step()

        if self.verbose:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.6f}")

        return False
