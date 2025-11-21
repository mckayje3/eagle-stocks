"""Evaluation metrics for time-series"""

import torch
import torch.nn.functional as F


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error"""
    return F.mse_loss(predictions, targets)


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error"""
    return torch.sqrt(F.mse_loss(predictions, targets))


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error"""
    return F.l1_loss(predictions, targets)


def mape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error"""
    return torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """R-squared (coefficient of determination)"""
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))
