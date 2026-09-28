"""Local vision understanding foundation for SnapSight.

This module provides in-memory visual scene and content understanding for
captured camera frames. It performs lightweight, local object detection using
a quantized/nano ONNX model (YOLOv8n / COCO 80 classes) alongside genuine
image-derived lighting, sharpness analysis, and optional local Tesseract OCR.
All processing is 100% local with zero disk writes and zero external network calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"

COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

_session_lock = threading.Lock()
_cached_session: ort.InferenceSession | None = None
_cached_session_path: str | None = None


class VisionError(RuntimeError):
    """Base exception for vision processing errors."""


class InvalidFrameError(VisionError):
    """Raised when an empty or invalid frame buffer is provided."""


class VisionModelUnavailableError(VisionError):
    """Raised when the object detection model file cannot be found."""


class VisionModelLoadError(VisionError):
    """Raised when loading the vision model fails."""


class VisionInferenceError(VisionError):
    """Raised when visual analysis or inference encounters an error."""


def resolve_tesseract_cmd() -> str:
    """Resolve the Tesseract binary path: check system PATH first, then fallback to default."""
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path
    return DEFAULT_TESSERACT_PATH


def get_vision_session(model_path: str | Path | None = None) -> ort.InferenceSession:
    """Get or load a cached ONNX Runtime session for object detection.

    Thread-safe and lazy-loaded to prevent startup delays.
    """
    global _cached_session, _cached_session_path

    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    with _session_lock:
        if _cached_session is not None and _cached_session_path == str(target_path):
            return _cached_session

        if not target_path.exists():
            raise VisionModelUnavailableError(
                f"Object detection model not found at {target_path}."
            )

        try:
            logger.info("Loading local ONNX object detection model from %s", target_path)
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(target_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            _cached_session = session
            _cached_session_path = str(target_path)
            return session
        except Exception as error:
            logger.exception("Failed to load ONNX vision model from %s", target_path)
            raise VisionModelLoadError(f"Failed to load vision model: {error}") from error


def analyze_lighting(gray_frame: np.ndarray) -> tuple[str, float]:
    """Assess the lighting condition of a grayscale frame.

    Returns:
        tuple[str, float]: (lighting_label, mean_luminance)
    """
    mean_lum = float(np.mean(gray_frame))
    if mean_lum < 40.0:
        label = "low-light / underexposed"
    elif mean_lum < 85.0:
        label = "dimly lit"
    elif mean_lum > 220.0:
        label = "overexposed / glare"
    else:
        label = "well-lit"
    return label, round(mean_lum, 1)


def analyze_sharpness(gray_frame: np.ndarray) -> tuple[str, float]:
    """Assess the focus and sharpness of a grayscale frame using Laplacian variance.

    Returns:
        tuple[str, float]: (sharpness_label, variance_score)
    """
    laplacian_var = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
    if laplacian_var < 30.0:
        label = "blurry / out-of-focus"
    elif laplacian_var < 100.0:
        label = "moderate focus"
    else:
        label = "sharp / in-focus"
    return label, round(laplacian_var, 1)


def extract_camera_text(bgr_frame: np.ndarray) -> tuple[str, str | None]:
    """Attempt in-memory OCR on the camera frame using local Tesseract.

    Returns:
        tuple[str, str | None]: (extracted_text, note)
    """
    try:
        import pytesseract
    except ImportError:
        return "", "pytesseract is not installed; text extraction skipped."

    tesseract_cmd = resolve_tesseract_cmd()
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        text = pytesseract.image_to_string(pil_img).strip()
        return text, None
    except pytesseract.TesseractNotFoundError:
        return "", f"Local Tesseract OCR not found at {tesseract_cmd}."
    except Exception as error:
        logger.warning("Camera OCR encountered an error: %s", error)
        return "", f"OCR processing error: {error}"


def preprocess_frame(
    frame: np.ndarray, target_size: int = 640
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox resize an in-memory BGR frame and prepare an NCHW tensor for YOLO.

    Returns:
        tuple: (blob_tensor, scale_ratio, (pad_dx, pad_dy))
    """
    h, w = frame.shape[:2]
    scale = min(target_size / h, target_size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    padded[dy : dy + nh, dx : dx + nw] = resized

    # Convert BGR -> RGB, normalize to [0, 1], transpose to (1, 3, target_size, target_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)

    return blob, scale, (dx, dy)


def detect_objects(
    frame: np.ndarray,
    conf_threshold: float = 0.35,
    nms_threshold: float = 0.45,
    session: ort.InferenceSession | None = None,
) -> list[dict[str, Any]]:
    """Run local object detection on an in-memory BGR frame.

    Args:
        frame: BGR image as NumPy array.
        conf_threshold: Minimum confidence score to retain detection (default 0.35).
        nms_threshold: IoU threshold for Non-Maximum Suppression (default 0.45).
        session: Optional pre-loaded InferenceSession (uses cached if None).

    Returns:
        List of dicts: [{"label": str, "confidence": float, "box": [x, y, w, h]}]
    """
    if session is None:
        session = get_vision_session()

    blob, scale, (dx, dy) = preprocess_frame(frame, target_size=640)
    input_name = session.get_inputs()[0].name

    try:
        outputs = session.run(None, {input_name: blob})[0]
    except Exception as error:
        raise VisionInferenceError(f"Inference execution failed: {error}") from error

    # Output shape is [1, 84, 8400] for standard YOLOv8
    predictions = outputs[0].T  # shape (8400, 84)

    boxes: list[list[int]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for row in predictions:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])

        if conf >= conf_threshold and class_id < len(COCO_CLASSES):
            cx, cy, bw, bh = row[0:4]
            # Convert letterbox coordinates back to original frame coordinates
            orig_cx = (cx - dx) / scale
            orig_cy = (cy - dy) / scale
            orig_w = bw / scale
            orig_h = bh / scale
            x = int(round(orig_cx - orig_w / 2))
            y = int(round(orig_cy - orig_h / 2))

            boxes.append([x, y, int(round(orig_w)), int(round(orig_h))])
            confidences.append(conf)
            class_ids.append(class_id)

    if not boxes:
        return []

    # Non-Maximum Suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []

    detected_objects: list[dict[str, Any]] = []
    for i in np.array(indices).flatten():
        idx = int(i)
        label = COCO_CLASSES[class_ids[idx]]
        conf = round(confidences[idx], 2)
        detected_objects.append({
            "label": label,
            "confidence": conf,
            "box": boxes[idx],
        })


    # Sort detections by confidence descending
    detected_objects.sort(key=lambda o: o["confidence"], reverse=True)
    return detected_objects


def analyze_visual_frame(
    frame: np.ndarray,
    conf_threshold: float = 0.35,
    session: ort.InferenceSession | None = None,
) -> dict[str, Any]:
    """Perform genuine local visual understanding on an in-memory BGR camera frame.

    Extracts genuine COCO object labels with confidence scores, evaluates
    lighting/sharpness metrics from pixels, and attempts in-memory OCR.

    Args:
        frame: NumPy array (H, W, 3) in BGR format.
        conf_threshold: Confidence threshold for object detection.
        session: Optional custom or mocked ONNX session.

    Returns:
        Structured dictionary compatible with CameraResult schema.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise InvalidFrameError("The provided frame buffer is empty or not a valid NumPy array.")

    if frame.ndim not in (2, 3):
        raise InvalidFrameError(f"Expected 2D or 3D frame array, got shape {frame.shape}.")

    h, w = frame.shape[:2]
    channels = frame.shape[2] if frame.ndim > 2 else 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if channels == 3 else frame
    lighting_label, mean_lum = analyze_lighting(gray)
    sharpness_label, lap_var = analyze_sharpness(gray)

    # 1. Real Object Detection
    objects: list[dict[str, Any]] = []
    model_note: str | None = None
    try:
        objects = detect_objects(frame, conf_threshold=conf_threshold, session=session)
    except VisionModelUnavailableError as error:
        logger.warning("Object detection model unavailable: %s", error)
        model_note = "Local object detection model is unavailable; visual understanding ran without object recognition."
    except Exception as error:
        logger.exception("Object detection failed during analysis: %s", error)
        model_note = f"Object detection encountered an issue: {error}"

    # 2. Text / OCR Extraction
    text, ocr_note = extract_camera_text(frame)

    # 3. Truthful response synthesis
    simplified_objects = [{"label": obj["label"], "confidence": obj["confidence"]} for obj in objects]

    if objects:
        summary_items = [f"{obj['label']} ({int(obj['confidence'] * 100)}%)" for obj in objects]
        object_summary = ", ".join(summary_items)
        response = (
            f"I observed a {lighting_label} camera view and detected: {object_summary}."
        )
        visual_context = (
            f"Captured 1 camera frame ({w}x{h}). Detected {len(objects)} object(s): "
            f"{', '.join(obj['label'] for obj in objects)}. Scene: {lighting_label}, {sharpness_label}."
        )
    else:
        if text:
            response = (
                f"I observed a {lighting_label} camera view in {sharpness_label}. "
                "No common objects were recognized, but readable text was detected."
            )
        else:
            response = (
                f"I observed a {lighting_label} camera view in {sharpness_label}. "
                "No common objects or readable text were recognized in the frame."
            )
        visual_context = (
            f"Captured 1 camera frame ({w}x{h}). No objects detected above confidence {conf_threshold}. "
            f"Scene: {lighting_label}, {sharpness_label}."
        )

    if text:
        visual_context += f' Visible text observed: "{text}".'
        if objects:
            response += f' Readable text detected: "{text}".'
        else:
            response += f' Text: "{text}".'


    # Gather informative notes
    notes: list[str] = []
    if model_note:
        notes.append(model_note)
    if "low-light" in lighting_label or "dim" in lighting_label:
        notes.append("Ambient lighting is dim; improving lighting may help recognize objects.")
    if "blurry" in sharpness_label:
        notes.append("Camera view appears blurry; hold the camera steady or adjust focus.")
    if ocr_note and not text:
        pass

    note = " ".join(notes) if notes else None

    frame_info = {
        "width": w,
        "height": h,
        "channels": channels,
        "format": "BGR",
        "lighting": lighting_label,
        "mean_luminance": mean_lum,
        "sharpness": sharpness_label,
        "sharpness_score": lap_var,
        "objects": simplified_objects,
        "detected_text": text if text else None,
    }

    return {
        "status": "success",
        "visual_context": visual_context,
        "response": response,
        "note": note,
        "objects": simplified_objects,
        "frame_info": frame_info,
    }
