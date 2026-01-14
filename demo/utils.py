"""
Utility functions for demo application.

Includes visualization helpers and video processing utilities.
"""

import cv2
import numpy as np
from typing import List, Tuple


def draw_skeleton(frame, keypoints, connections, confidence_threshold=0.5):
    """
    Draw skeleton on frame.

    Args:
        frame: Video frame (H, W, 3)
        keypoints: Array of shape (num_joints, 3) with (x, y, confidence)
        connections: List of (joint_idx1, joint_idx2) tuples
        confidence_threshold: Minimum confidence to draw keypoint

    Returns:
        annotated_frame: Frame with skeleton overlay
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw connections
    for joint1_idx, joint2_idx in connections:
        if (keypoints[joint1_idx, 2] > confidence_threshold and
            keypoints[joint2_idx, 2] > confidence_threshold):

            pt1 = (int(keypoints[joint1_idx, 0] * w),
                   int(keypoints[joint1_idx, 1] * h))
            pt2 = (int(keypoints[joint2_idx, 0] * w),
                   int(keypoints[joint2_idx, 1] * h))

            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            center = (int(x * w), int(y * h))
            cv2.circle(annotated, center, 4, (0, 0, 255), -1)

    return annotated


def draw_prediction_overlay(
    frame,
    is_fall,
    probability,
    fall_count=0,
    fps=None
):
    """
    Draw prediction overlay on frame.

    Args:
        frame: Video frame
        is_fall: Boolean fall prediction
        probability: Fall probability
        fall_count: Number of falls detected
        fps: Frames per second (optional)

    Returns:
        annotated_frame: Frame with overlay
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Status box
    status_text = "FALL DETECTED!" if is_fall else "Normal Activity"
    status_color = (0, 0, 255) if is_fall else (0, 255, 0)

    # Draw semi-transparent background for text
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

    # Draw status text
    cv2.putText(
        annotated,
        status_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3
    )

    # Draw probability
    prob_text = f"Confidence: {probability:.2%}"
    cv2.putText(
        annotated,
        prob_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Draw fall count
    if fall_count > 0:
        count_text = f"Falls Detected: {fall_count}"
        cv2.putText(
            annotated,
            count_text,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    # Draw FPS if provided
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            annotated,
            fps_text,
            (w - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return annotated


def get_coco_skeleton_connections():
    """
    Get COCO skeleton connections.

    Returns:
        connections: List of (joint_idx1, joint_idx2) tuples
    """
    return [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]


def format_detection_summary(detections):
    """
    Format detection results as text summary.

    Args:
        detections: List of (timestamp, frame_idx, probability, is_fall) tuples

    Returns:
        summary: Formatted text summary
    """
    fall_detections = [d for d in detections if d[3]]

    summary = f"Video Analysis Complete\n"
    summary += f"=" * 50 + "\n\n"
    summary += f"Total Frames Analyzed: {len(detections)}\n"
    summary += f"Falls Detected: {len(fall_detections)}\n\n"

    if fall_detections:
        summary += "Fall Timestamps:\n"
        summary += "-" * 50 + "\n"
        for timestamp, frame_idx, probability, _ in fall_detections:
            summary += f"  Frame {frame_idx:5d} | "
            summary += f"Time {timestamp:6.2f}s | "
            summary += f"Confidence {probability:.2%}\n"
    else:
        summary += "No falls detected in this video.\n"

    return summary
