"""Tests for backend/camera.py with mocked OpenCV hardware."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.camera import (
    DEFAULT_CAMERA_INDEX,
    CameraError,
    CameraUnavailableError,
    CapturedFrame,
    FrameCaptureError,
    capture_camera_frame,
    understand_camera,
)


class TestCaptureCameraFrame:
    """Tests for capture_camera_frame()."""

    def test_capture_camera_frame_success(self):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
            result = capture_camera_frame(device_index=0)

            mock_vc.assert_called_once_with(0, cv2.CAP_DSHOW)
            mock_cap.isOpened.assert_called_once()
            mock_cap.read.assert_called_once()
            mock_cap.release.assert_called_once()

            assert isinstance(result, CapturedFrame)
            assert result.width == 640
            assert result.height == 480
            assert result.channels == 3
            assert result.format == "BGR"
            assert np.array_equal(result.frame, fake_frame)

    def test_capture_camera_unavailable(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(CameraUnavailableError) as exc_info:
                capture_camera_frame(device_index=0)

            assert "could not be opened" in str(exc_info.value)
            mock_cap.release.assert_called_once()

    def test_capture_camera_read_failure(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(FrameCaptureError) as exc_info:
                capture_camera_frame(device_index=0)

            assert "Failed to read a frame" in str(exc_info.value)
            mock_cap.release.assert_called_once()

    def test_capture_camera_empty_frame(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.array([]))

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(FrameCaptureError) as exc_info:
                capture_camera_frame(device_index=0)

            assert "Failed to read a frame" in str(exc_info.value)
            mock_cap.release.assert_called_once()

    def test_camera_is_always_released_on_unexpected_exception(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = RuntimeError("Low-level driver crash")

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError) as exc_info:
                capture_camera_frame(device_index=0)

            assert "Low-level driver crash" in str(exc_info.value)
            mock_cap.release.assert_called_once()


class TestUnderstandCamera:
    """Tests for understand_camera()."""

    def test_understand_camera_success(self, monkeypatch):
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        captured = CapturedFrame(
            frame=fake_frame,
            width=1280,
            height=720,
            channels=3,
            format="BGR",
        )
        monkeypatch.setattr("backend.camera.capture_camera_frame", lambda **kwargs: captured)
        monkeypatch.setattr(
            "backend.vision.analyze_visual_frame",
            lambda frame: {
                "status": "success",
                "visual_context": "Captured 1 camera frame (1280x720, 3 channels) locally in memory.",
                "response": "I observed a well-lit camera view.",
                "note": None,
                "objects": [{"label": "bottle", "confidence": 0.89}],
                "frame_info": {
                    "width": 1280,
                    "height": 720,
                    "channels": 3,
                    "format": "BGR",
                    "objects": [{"label": "bottle", "confidence": 0.89}],
                },
            },
        )

        result = understand_camera()

        assert result["status"] == "success"
        assert "1280x720" in result["visual_context"]
        assert result["response"] == "I observed a well-lit camera view."
        assert result["objects"] == [{"label": "bottle", "confidence": 0.89}]
        assert result["note"] is None
        assert result["frame_info"]["width"] == 1280


    def test_understand_camera_vision_failure_fallback(self, monkeypatch):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        captured = CapturedFrame(
            frame=fake_frame,
            width=640,
            height=480,
            channels=3,
            format="BGR",
        )
        monkeypatch.setattr("backend.camera.capture_camera_frame", lambda **kwargs: captured)

        def mock_vision_fail(frame):
            raise RuntimeError("Vision inference model failure")

        monkeypatch.setattr("backend.vision.analyze_visual_frame", mock_vision_fail)

        result = understand_camera()

        assert result["status"] == "partial"
        assert "Vision inference model failure" in result["note"]
        assert result["frame_info"]["width"] == 640

    def test_understand_camera_unavailable_error(self, monkeypatch):
        def mock_capture(**kwargs):
            raise CameraUnavailableError("Camera device is busy")

        monkeypatch.setattr("backend.camera.capture_camera_frame", mock_capture)

        result = understand_camera()

        assert result["status"] == "error"
        assert result["frame_info"] is None
        assert result["note"] == "Camera device is busy"
        assert "Could not access or capture" in result["response"]

    def test_understand_camera_read_failure_error(self, monkeypatch):
        def mock_capture(**kwargs):
            raise FrameCaptureError("Failed to read a frame")

        monkeypatch.setattr("backend.camera.capture_camera_frame", mock_capture)

        result = understand_camera()

        assert result["status"] == "error"
        assert result["frame_info"] is None
        assert result["note"] == "Failed to read a frame"
        assert "Could not access or capture" in result["response"]

    def test_understand_camera_unexpected_exception(self, monkeypatch):
        def mock_capture(**kwargs):
            raise Exception("Unexpected system error")

        monkeypatch.setattr("backend.camera.capture_camera_frame", mock_capture)

        result = understand_camera()

        assert result["status"] == "error"
        assert result["frame_info"] is None
        assert "Unexpected system error" in result["note"]
        assert "encountered an internal error" in result["response"]
