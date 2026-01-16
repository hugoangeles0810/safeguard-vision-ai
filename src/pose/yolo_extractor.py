"""
Pose extraction using Ultralytics YOLOv8-Pose.

More accurate than MediaPipe but heavier computationally.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOv8Extractor:
    """Pose extractor using YOLOv8-Pose."""

    def __init__(self, model_name='yolov8n-pose.pt', device='cuda'):
        """
        Initialize YOLOv8 pose extractor.

        Args:
            model_name: YOLO model variant ('yolov8n-pose', 'yolov8s-pose', etc.)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(device)

    def extract_from_frame(self, frame, conf_threshold=0.5):
        """
        Extract pose keypoints from a single frame.

        Args:
            frame: BGR image (H, W, 3)
            conf_threshold: Confidence threshold for detections

        Returns:
            keypoints: Array of shape (17, 3) with (x, y, confidence)
                      Returns None if no pose detected
                      Uses COCO format: 17 keypoints
        """
        # TODO: Implement frame processing
        # Run YOLO inference
        # Extract keypoints for highest confidence detection
        raise NotImplementedError

    def extract_from_video(self, video_path, output_dir=None, conf_threshold=0.5):
        """
        Extract pose keypoints from entire video.

        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted keypoints (optional)
            conf_threshold: Confidence threshold for detections

        Returns:
            keypoints_sequence: Array of shape (T, 17, 3)
            metadata: Dictionary with video metadata
        """
        # TODO: Implement video processing
        raise NotImplementedError

    def draw_keypoints(self, frame, keypoints):
        """
        Draw pose keypoints on frame.

        Args:
            frame: BGR image
            keypoints: Array of shape (17, 3)

        Returns:
            annotated_frame: Frame with pose overlay
        """
        # TODO: Implement keypoint drawing with skeleton connections
        raise NotImplementedError

    def get_keypoint_names(self):
        """
        Get COCO keypoint names.

        Returns:
            names: List of 17 keypoint names
        """
        return [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
