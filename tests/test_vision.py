"""Unit tests for backend/vision.py with mocked vision hardware and models."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import pytesseract

from backend.vision import (
    COCO_CLASSES,
    PROVIDER_PRIORITY,
    InvalidFrameError,
    VisionInferenceError,
    VisionModelLoadError,
    VisionModelUnavailableError,
    analyze_lighting,
    analyze_sharpness,
    analyze_visual_frame,
    detect_objects,
    extract_camera_text,
    get_supported_execution_providers,
    get_vision_session,
    preprocess_frame,
)


def create_mock_yolo_output(detections: list[tuple[int, float, list[float]]]) -> np.ndarray:
    """Helper to generate a mock YOLOv8 output tensor of shape (1, 84, 8400).

    Args:
        detections: list of (class_id, confidence, [cx, cy, w, h]) in 640x640 space.
    """
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    for i, (class_id, conf, (cx, cy, w, h)) in enumerate(detections):
        if i >= 8400:
            break
        output[0, 0, i] = cx
        output[0, 1, i] = cy
        output[0, 2, i] = w
        output[0, 3, i] = h
        output[0, 4 + class_id, i] = conf
    return output


class TestLightingAndSharpness:
    """Tests for lighting and focus analysis."""

    def test_low_light_detection(self):
        dark_frame = np.full((100, 100), 20, dtype=np.uint8)
        label, score = analyze_lighting(dark_frame)
        assert label == "low-light / underexposed"
        assert score == 20.0

    def test_well_lit_detection(self):
        normal_frame = np.full((100, 100), 140, dtype=np.uint8)
        label, score = analyze_lighting(normal_frame)
        assert label == "well-lit"
        assert score == 140.0

    def test_sharpness_analysis(self):
        blurry_frame = np.full((100, 100), 128, dtype=np.uint8)
        label, score = analyze_sharpness(blurry_frame)
        assert label == "blurry / out-of-focus"
        assert score == 0.0

        sharp_frame = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(sharp_frame, (50, 50), (150, 150), 255, -1)
        label, score = analyze_sharpness(sharp_frame)
        assert label == "sharp / in-focus"
        assert score > 100.0


class TestModelSessionManagement:
    """Tests for model loading and error handling."""

    def test_model_unavailable_error(self, tmp_path):
        missing_file = tmp_path / "non_existent.onnx"
        with pytest.raises(VisionModelUnavailableError) as exc_info:
            get_vision_session(model_path=missing_file)
        assert "not found" in str(exc_info.value)

    def test_model_load_failure(self, tmp_path):
        corrupt_file = tmp_path / "corrupt.onnx"
        corrupt_file.write_text("not a real onnx")
        with pytest.raises(VisionModelLoadError) as exc_info:
            get_vision_session(model_path=corrupt_file)
        assert "Failed to load vision model" in str(exc_info.value)

    def test_get_supported_execution_providers_priority(self):
        # When only standard CPU and Azure are present
        assert get_supported_execution_providers(["AzureExecutionProvider", "CPUExecutionProvider"]) == [
            "CPUExecutionProvider"
        ]

        # When QNN is present
        assert get_supported_execution_providers(
            ["QNNExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
        ) == ["QNNExecutionProvider", "CPUExecutionProvider"]

        # When DML is present
        assert get_supported_execution_providers(
            ["DmlExecutionProvider", "CPUExecutionProvider"]
        ) == ["DmlExecutionProvider", "CPUExecutionProvider"]

        # When both QNN and DML are present, priority QNN > DML > CPU is preserved
        assert get_supported_execution_providers(
            ["DmlExecutionProvider", "QNNExecutionProvider", "CPUExecutionProvider"]
        ) == ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

        # When list is empty or CPU missing, CPU fallback is always guaranteed
        assert get_supported_execution_providers([]) == ["CPUExecutionProvider"]
        assert get_supported_execution_providers(["SomeOtherProvider"]) == ["CPUExecutionProvider"]

    def test_get_supported_execution_providers_current_machine(self):
        # On this development machine, only CPUExecutionProvider should be active
        providers = get_supported_execution_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_accelerator_initialization_failure_falls_back_to_cpu(self, tmp_path, monkeypatch):
        # Create a dummy model file
        dummy_model = tmp_path / "model.onnx"
        dummy_model.write_text("dummy")

        # Mock providers to return QNN and CPU
        monkeypatch.setattr(
            "backend.vision.get_supported_execution_providers",
            lambda: ["QNNExecutionProvider", "CPUExecutionProvider"],
        )

        mock_cpu_session = MagicMock()
        calls = []

        def mock_init(model_str, sess_options=None, providers=None):
            calls.append(providers)
            if providers == ["QNNExecutionProvider", "CPUExecutionProvider"]:
                raise RuntimeError("Failed to load QNN HTP backend DLL")
            return mock_cpu_session

        monkeypatch.setattr("onnxruntime.InferenceSession", mock_init)

        # Clear cached session to force loading
        import backend.vision as bv
        bv._cached_session = None
        bv._cached_session_path = None

        session = get_vision_session(model_path=dummy_model)
        assert session is mock_cpu_session
        assert len(calls) == 2
        assert calls[0] == ["QNNExecutionProvider", "CPUExecutionProvider"]
        assert calls[1] == ["CPUExecutionProvider"]

        # Cleanup cached session
        bv._cached_session = None
        bv._cached_session_path = None


class TestObjectDetection:
    """Tests for detect_objects()."""

    def test_detect_single_object(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        # Bottle is class 39
        mock_output = create_mock_yolo_output([(39, 0.89, [320.0, 320.0, 100.0, 200.0])])
        mock_session.run.return_value = [mock_output]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detect_objects(frame, conf_threshold=0.35, session=mock_session)

        assert len(results) == 1
        assert results[0]["label"] == "bottle"
        assert results[0]["confidence"] == 0.89
        assert "box" in results[0]

    def test_detect_multiple_objects(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        # Laptop (63) and Cell Phone (67)
        mock_output = create_mock_yolo_output([
            (63, 0.85, [200.0, 200.0, 150.0, 150.0]),
            (67, 0.78, [450.0, 300.0, 60.0, 120.0]),
        ])
        mock_session.run.return_value = [mock_output]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detect_objects(frame, conf_threshold=0.35, session=mock_session)

        assert len(results) == 2
        labels = [r["label"] for r in results]
        assert "laptop" in labels
        assert "cell phone" in labels
        assert results[0]["confidence"] >= results[1]["confidence"]

    def test_confidence_threshold_filtering(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        # Bottle (39) with low confidence (0.20) below threshold 0.35
        mock_output = create_mock_yolo_output([(39, 0.20, [320.0, 320.0, 100.0, 200.0])])
        mock_session.run.return_value = [mock_output]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detect_objects(frame, conf_threshold=0.35, session=mock_session)

        assert len(results) == 0

    def test_no_objects_detected(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_output = np.zeros((1, 84, 8400), dtype=np.float32)
        mock_session.run.return_value = [mock_output]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detect_objects(frame, conf_threshold=0.35, session=mock_session)

        assert len(results) == 0

    def test_inference_failure(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.side_effect = RuntimeError("GPU/CPU crash")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(VisionInferenceError):
            detect_objects(frame, session=mock_session)

    def test_vectorized_postprocessing_box_conversion(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        # Cup is class 41
        # Place detection at center of 640x640 space: cx=320, cy=320, w=100, h=100
        mock_output = create_mock_yolo_output([(41, 0.95, [320.0, 320.0, 100.0, 100.0])])
        mock_session.run.return_value = [mock_output]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detect_objects(frame, conf_threshold=0.35, session=mock_session)

        assert len(results) == 1
        assert results[0]["label"] == "cup"
        assert results[0]["confidence"] == 0.95
        box = results[0]["box"]
        assert len(box) == 4
        # Verify box coordinates are standard ints
        assert all(isinstance(coord, int) for coord in box)


class TestCameraTextExtraction:
    """Tests for extract_camera_text()."""

    def test_extract_text_success(self):
        fake_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        with patch("pytesseract.image_to_string", return_value="SnapSight Notes"):
            text, note = extract_camera_text(fake_frame)
            assert text == "SnapSight Notes"
            assert note is None

    def test_extract_text_tesseract_not_found(self):
        fake_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        with patch("pytesseract.image_to_string", side_effect=pytesseract.TesseractNotFoundError()):
            text, note = extract_camera_text(fake_frame)
            assert text == ""
            assert "Local Tesseract OCR not found" in note


class TestAnalyzeVisualFrame:
    """Tests for analyze_visual_frame()."""

    def test_analyze_frame_with_single_detected_object(self, monkeypatch):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)

        mock_detected = [{"label": "bottle", "confidence": 0.89, "box": [100, 100, 50, 150]}]
        monkeypatch.setattr("backend.vision.detect_objects", lambda *args, **kwargs: mock_detected)
        monkeypatch.setattr("backend.vision.extract_camera_text", lambda *args, **kwargs: ("", None))

        result = analyze_visual_frame(frame)

        assert result["status"] == "success"
        assert result["objects"] == [{"label": "bottle", "confidence": 0.89}]
        assert "bottle (89%)" in result["response"]
        assert "bottle" in result["visual_context"]
        assert result["frame_info"]["objects"] == [{"label": "bottle", "confidence": 0.89}]
        assert result["frame_info"]["detected_text"] is None

    def test_analyze_frame_no_objects(self, monkeypatch):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)

        monkeypatch.setattr("backend.vision.detect_objects", lambda *args, **kwargs: [])
        monkeypatch.setattr("backend.vision.extract_camera_text", lambda *args, **kwargs: ("", None))

        result = analyze_visual_frame(frame)

        assert result["status"] == "success"
        assert result["objects"] == []
        assert "No common objects or readable text were recognized" in result["response"]
        assert result["frame_info"]["objects"] == []

    def test_analyze_frame_with_ocr_and_objects(self, monkeypatch):
        frame = np.full((480, 640, 3), 140, dtype=np.uint8)

        mock_detected = [{"label": "cell phone", "confidence": 0.92, "box": [200, 200, 80, 160]}]
        monkeypatch.setattr("backend.vision.detect_objects", lambda *args, **kwargs: mock_detected)
        monkeypatch.setattr("backend.vision.extract_camera_text", lambda *args, **kwargs: ("Meeting 2pm", None))

        result = analyze_visual_frame(frame)

        assert result["status"] == "success"
        assert "cell phone (92%)" in result["response"]
        assert 'Readable text detected: "Meeting 2pm"' in result["response"]
        assert result["frame_info"]["detected_text"] == "Meeting 2pm"

    def test_analyze_frame_model_unavailable_fallback(self, monkeypatch):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)

        def mock_unavailable(*args, **kwargs):
            raise VisionModelUnavailableError("Model missing")

        monkeypatch.setattr("backend.vision.detect_objects", mock_unavailable)
        monkeypatch.setattr("backend.vision.extract_camera_text", lambda *args, **kwargs: ("", None))

        result = analyze_visual_frame(frame)

        assert result["status"] == "success"
        assert result["objects"] == []
        assert "Local object detection model is unavailable" in result["note"]

    def test_invalid_frame_none_or_empty(self):
        with pytest.raises(InvalidFrameError):
            analyze_visual_frame(None)

        with pytest.raises(InvalidFrameError):
            analyze_visual_frame(np.array([]))
