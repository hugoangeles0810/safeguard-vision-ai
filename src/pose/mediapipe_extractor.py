"""
Pose extraction using Google MediaPipe.

Fast and lightweight pose estimation suitable for real-time applications.
"""

import os
import urllib.request
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


class MediaPipeExtractor:
    """Pose extractor using MediaPipe Pose."""

    def __init__(
        self,
        model_path=None,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode='IMAGE'
    ):
        """
        Initialize MediaPipe pose extractor.

        Args:
            model_path: Path to pose landmarker model file (.task)
                       If None, downloads default model
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            running_mode: 'IMAGE', 'VIDEO', or 'LIVE_STREAM'
        """
        # Download model if not provided
        if model_path is None:
            model_dir = os.path.expanduser("~/.mediapipe/models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "pose_landmarker_lite.task")

            if not os.path.exists(model_path):
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, model_path)

        # Configure options
        base_options = python.BaseOptions(model_asset_path=model_path)

        # Map string to enum
        mode_map = {
            'IMAGE': vision.RunningMode.IMAGE,
            'VIDEO': vision.RunningMode.VIDEO,
            'LIVE_STREAM': vision.RunningMode.LIVE_STREAM
        }
        running_mode_enum = mode_map.get(running_mode.upper(), vision.RunningMode.IMAGE)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode_enum,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract_from_frame(self, frame):
        """
        Extract pose keypoints from a single frame.

        Args:
            frame: Can be either:
                  - BGR image as numpy array (H, W, 3)
                  - Path to image file (str) - supports PNG, JPG, JPEG

        Returns:
            keypoints: Array of shape (33, 3) with (x, y, visibility)
                      Returns None if no pose detected

        Raises:
            ValueError: If image file path is invalid or cannot be loaded
            TypeError: If frame is neither a numpy array nor a string path
        """
        # Handle string path to image file
        if isinstance(frame, str):
            if not os.path.exists(frame):
                raise ValueError(f"Image file not found: {frame}")

            # Load image from file
            loaded_frame = cv2.imread(frame)

            if loaded_frame is None:
                raise ValueError(f"Failed to load image from: {frame}")

            frame = loaded_frame
        elif not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Frame must be either a numpy array or a file path string, "
                f"got {type(frame).__name__}"
            )

        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        # Detect pose landmarks
        detection_result = self.landmarker.detect(mp_image)

        # Extract landmarks if pose detected
        if not detection_result.pose_landmarks:
            return None

        # Convert landmarks to numpy array (33, 3)
        # Use first detected pose
        landmarks = detection_result.pose_landmarks[0]
        keypoints = np.array([
            [landmark.x, landmark.y, landmark.visibility]
            for landmark in landmarks
        ])

        return keypoints

    def draw_landmarks(self, frame, detection_result):
        """
        Draw pose landmarks on frame.

        Args:
            frame: BGR image
            detection_result: MediaPipe pose detection result

        Returns:
            annotated_frame: Frame with pose overlay
        """
        annotated_frame = frame.copy()

        if not detection_result.pose_landmarks:
            return annotated_frame

        # Get frame dimensions
        h, w = frame.shape[:2]

        # Draw landmarks as circles
        for pose_landmarks in detection_result.pose_landmarks:
            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

        # Draw connections (simplified version)
        # MediaPipe Pose has 33 landmarks with specific connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face
            (0, 4), (4, 5), (5, 6), (6, 8),  # Face
            (9, 10),  # Mouth
            (11, 12),  # Shoulders
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
        ]

        for pose_landmarks in detection_result.pose_landmarks:
            landmarks_list = list(pose_landmarks)
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                    start = landmarks_list[start_idx]
                    end = landmarks_list[end_idx]

                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))

                    cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)

        return annotated_frame

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
