"""
Loss functions for fall detection.

Includes standard losses and class-weighted variants for imbalanced data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss = -alpha * (1 - p)^gamma * log(p)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Args:
            inputs: Model predictions (batch, 1)
            targets: Ground truth labels (batch, 1)

        Returns:
            loss: Focal loss value
        """
        # TODO: Implement focal loss
        raise NotImplementedError


class WeightedBCELoss(nn.Module):
    """Binary Cross-Entropy with class weights."""

    def __init__(self, pos_weight=None):
        """
        Initialize weighted BCE loss.

        Args:
            pos_weight: Weight for positive class (fall)
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        """
        Compute weighted BCE loss.

        Args:
            inputs: Model logits (batch, 1)
            targets: Ground truth labels (batch, 1)

        Returns:
            loss: BCE loss value
        """
        return self.bce(inputs, targets)


def get_loss_function(loss_type='bce', class_weights=None):
    """
    Factory function to get loss function.

    Args:
        loss_type: Type of loss ('bce', 'weighted_bce', 'focal')
        class_weights: Class weights for weighted loss

    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss(pos_weight=class_weights)
    elif loss_type == 'focal':
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
