"""
Unit tests for MediaPipe pose extractor.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.pose.mediapipe_extractor import MediaPipeExtractor


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame for testing."""
    # Create a simple 640x480 BGR image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_png_image(tmp_path):
    """Create a sample PNG image file for testing."""
    image_path = tmp_path / "test_image.png"

    # Create a simple 640x480 BGR image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Save as PNG
    cv2.imwrite(str(image_path), frame)

    return str(image_path)


@pytest.fixture
def mock_landmarks():
    """Create mock MediaPipe landmarks for new API."""
    landmarks = []
    for i in range(33):
        landmark = Mock()
        landmark.x = np.random.rand()
        landmark.y = np.random.rand()
        landmark.z = np.random.rand()
        landmark.visibility = np.random.rand()
        landmarks.append(landmark)
    return landmarks


@pytest.fixture
def mock_detection_result(mock_landmarks):
    """Create mock detection result."""
    result = Mock()
    result.pose_landmarks = [mock_landmarks]  # List of detected poses
    return result


class TestMediaPipeExtractor:
    """Test suite for MediaPipeExtractor class."""

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_initialization(self, mock_makedirs, mock_exists, mock_create):
        """Test extractor initialization with default parameters."""
        mock_exists.return_value = True  # Model already downloaded
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()

        assert extractor.landmarker is not None
        mock_create.assert_called_once()

        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_initialization_with_custom_params(self, mock_makedirs, mock_exists, mock_create):
        """Test extractor initialization with custom parameters."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            running_mode='VIDEO'
        )

        assert extractor.landmarker is not None
        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_extract_from_frame_success(self, mock_makedirs, mock_exists, mock_create, sample_frame, mock_detection_result):
        """Test successful keypoint extraction from a frame."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_landmarker.detect.return_value = mock_detection_result
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()
        keypoints = extractor.extract_from_frame(sample_frame)

        # Assertions
        assert keypoints is not None
        assert keypoints.shape == (33, 3)
        assert np.all(keypoints[:, 0] >= 0) and np.all(keypoints[:, 0] <= 1)  # x coordinates
        assert np.all(keypoints[:, 1] >= 0) and np.all(keypoints[:, 1] <= 1)  # y coordinates
        assert np.all(keypoints[:, 2] >= 0) and np.all(keypoints[:, 2] <= 1)  # visibility

        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_extract_from_frame_no_pose_detected(self, mock_makedirs, mock_exists, mock_create, sample_frame):
        """Test extraction when no pose is detected."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_result = Mock()
        mock_result.pose_landmarks = []  # Empty list = no pose detected
        mock_landmarker.detect.return_value = mock_result
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()
        keypoints = extractor.extract_from_frame(sample_frame)

        assert keypoints is None
        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_extract_from_frame_with_png_path(self, mock_makedirs, mock_exists, mock_create, sample_png_image, mock_detection_result):
        """Test successful keypoint extraction from PNG file path."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_landmarker.detect.return_value = mock_detection_result
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()
        keypoints = extractor.extract_from_frame(sample_png_image)

        # Assertions
        assert keypoints is not None
        assert keypoints.shape == (33, 3)
        assert np.all(keypoints[:, 0] >= 0) and np.all(keypoints[:, 0] <= 1)
        assert np.all(keypoints[:, 1] >= 0) and np.all(keypoints[:, 1] <= 1)
        assert np.all(keypoints[:, 2] >= 0) and np.all(keypoints[:, 2] <= 1)

        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_extract_from_frame_invalid_file_path(self, mock_makedirs, mock_exists, mock_create):
        """Test extraction with non-existent image file."""
        # Mock exists to return True only for model path, False for image file
        def exists_side_effect(path):
            if 'pose_landmarker_lite.task' in path:
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()

        with pytest.raises(ValueError, match="Image file not found"):
            extractor.extract_from_frame("/invalid/path/to/image.png")

        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    @patch('src.pose.mediapipe_extractor.cv2.imread')
    def test_extract_from_frame_corrupted_image(self, mock_imread, mock_makedirs, mock_exists, mock_create, tmp_path):
        """Test extraction with corrupted image file."""
        # Create a dummy file path
        corrupted_path = tmp_path / "corrupted.png"
        corrupted_path.touch()

        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker
        mock_imread.return_value = None  # Simulate failed image load

        extractor = MediaPipeExtractor()

        with pytest.raises(ValueError, match="Failed to load image from"):
            extractor.extract_from_frame(str(corrupted_path))

        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_extract_from_frame_invalid_type(self, mock_makedirs, mock_exists, mock_create):
        """Test extraction with invalid frame type."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()

        with pytest.raises(TypeError, match="Frame must be either a numpy array or a file path string"):
            extractor.extract_from_frame(123)  # Invalid type

        with pytest.raises(TypeError, match="Frame must be either a numpy array or a file path string"):
            extractor.extract_from_frame([1, 2, 3])  # List instead of array

        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_draw_landmarks(self, mock_makedirs, mock_exists, mock_create, sample_frame, mock_detection_result):
        """Test drawing landmarks on frame."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()
        annotated_frame = extractor.draw_landmarks(sample_frame, mock_detection_result)

        # Check that frame was not modified in place
        assert annotated_frame.shape == sample_frame.shape
        # Should return annotated frame (may be same shape even if drawing failed in mock)
        assert isinstance(annotated_frame, np.ndarray)

        extractor.close()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_close(self, mock_makedirs, mock_exists, mock_create):
        """Test resource cleanup."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()

        # Should not raise any errors
        extractor.close()
        mock_landmarker.close.assert_called_once()

    @patch('src.pose.mediapipe_extractor.vision.PoseLandmarker.create_from_options')
    @patch('src.pose.mediapipe_extractor.os.path.exists')
    @patch('src.pose.mediapipe_extractor.os.makedirs')
    def test_keypoint_format_consistency(self, mock_makedirs, mock_exists, mock_create, sample_frame, mock_detection_result):
        """Test that keypoints maintain consistent format across calls."""
        mock_exists.return_value = True
        mock_landmarker = Mock()
        mock_landmarker.detect.return_value = mock_detection_result
        mock_create.return_value = mock_landmarker

        extractor = MediaPipeExtractor()

        keypoints1 = extractor.extract_from_frame(sample_frame)
        keypoints2 = extractor.extract_from_frame(sample_frame)

        assert keypoints1.shape == keypoints2.shape
        assert keypoints1.dtype == keypoints2.dtype

        extractor.close()
