"""
Data augmentation techniques for temporal keypoint sequences.

Includes temporal jittering, speed variation, noise injection, and more.
"""

import numpy as np


def temporal_jitter(keypoints, jitter_ratio=0.1):
    """
    Apply temporal jittering to keypoint sequence.

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)
        jitter_ratio: Maximum ratio of temporal shift

    Returns:
        jittered_keypoints: Temporally jittered sequence
    """
    # TODO: Implement temporal jittering
    raise NotImplementedError


def speed_variation(keypoints, speed_factor=None):
    """
    Vary the playback speed of the sequence.

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)
        speed_factor: Speed multiplier (0.8-1.2). If None, randomly sample.

    Returns:
        resampled_keypoints: Speed-varied sequence
    """
    # TODO: Implement speed variation
    raise NotImplementedError


def horizontal_flip(keypoints):
    """
    Horizontally flip keypoints (mirror left-right).

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)

    Returns:
        flipped_keypoints: Horizontally flipped sequence
    """
    # TODO: Implement horizontal flip with joint swapping
    raise NotImplementedError


def add_noise(keypoints, noise_std=0.01):
    """
    Add Gaussian noise to keypoint coordinates.

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        noisy_keypoints: Keypoints with added noise
    """
    # TODO: Implement noise injection
    raise NotImplementedError


def keypoint_dropout(keypoints, dropout_prob=0.1):
    """
    Randomly drop (zero out) some keypoints to simulate occlusions.

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)
        dropout_prob: Probability of dropping each keypoint

    Returns:
        dropped_keypoints: Keypoints with random dropout
    """
    # TODO: Implement keypoint dropout
    raise NotImplementedError


def random_crop_temporal(keypoints, crop_length):
    """
    Randomly crop a subsequence from the keypoint sequence.

    Args:
        keypoints: Array of shape (T, num_joints, 2 or 3)
        crop_length: Length of cropped sequence

    Returns:
        cropped_keypoints: Randomly cropped subsequence
    """
    # TODO: Implement random temporal cropping
    raise NotImplementedError
