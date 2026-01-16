"""
Preprocessing utilities for pose keypoints.

Includes normalization, centering, and scaling operations.
"""

import numpy as np


def normalize_keypoints(keypoints, method='torso_height'):
    """
    Normalize keypoints for scale and translation invariance.

    Steps:
    1. Center on hip center
    2. Scale by torso height
    3. Optionally compute velocities/accelerations

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)
        method: Normalization method ('torso_height', 'bbox', 'none')

    Returns:
        normalized_keypoints: Normalized keypoint array
    """
    # TODO: Implement normalization
    raise NotImplementedError


def compute_velocities(keypoints):
    """
    Compute velocities from keypoint positions.

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)

    Returns:
        velocities: Array of shape (T-1, num_joints, 2 or 3)
    """
    # TODO: Implement velocity computation
    raise NotImplementedError


def compute_accelerations(keypoints):
    """
    Compute accelerations from keypoint positions.

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)

    Returns:
        accelerations: Array of shape (T-2, num_joints, 2 or 3)
    """
    # TODO: Implement acceleration computation
    raise NotImplementedError
