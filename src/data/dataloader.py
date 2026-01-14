"""
DataLoader factory for fall detection datasets.

Creates train/val/test dataloaders with appropriate batching and shuffling.
"""

from torch.utils.data import DataLoader


def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary with dataset parameters

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
    """
    # TODO: Implement dataloader creation
    raise NotImplementedError


def collate_fn(batch):
    """
    Custom collate function for batching sequences of variable length.

    Args:
        batch: List of (keypoints, label) tuples

    Returns:
        Batched and padded tensors
    """
    # TODO: Implement custom collation if needed
    raise NotImplementedError
