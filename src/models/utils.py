"""
Utility functions for models.

Includes model loading, saving, and parameter counting.
"""

import torch


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        num_params: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer to load state

    Returns:
        epoch: Epoch number
        metrics: Dictionary of metrics
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['metrics']


def initialize_weights(model):
    """
    Initialize model weights.

    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
