"""Camera frame capture foundation for SnapSight.

This module owns Windows camera capture. Exactly one frame is captured directly
into memory as a NumPy array using OpenCV with the DirectShow backend, ensuring
zero disk writes and strict adherence to SnapSight's privacy-first design.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_INDEX: int = 0


class CameraError(RuntimeError):
    """Base exception for camera capture failures."""


class CameraUnavailableError(CameraError):
    """Raised when the camera device cannot be opened or is busy."""


class FrameCaptureError(CameraError):
    """Raised when reading a frame from the opened camera fails."""


@dataclass(frozen=True)
class CapturedFrame:
    """In-memory representation of a single captured camera frame."""

    frame: np.ndarray
    width: int
    height: int
    channels: int = 3
    format: str = "BGR"


def capture_camera_frame(
    device_index: int = DEFAULT_CAMERA_INDEX,
) -> CapturedFrame:
    """Capture exactly one frame in-memory from the specified camera device.

    Uses the Windows DirectShow backend (cv2.CAP_DSHOW) for low-latency
    and reliable hardware access. The capture handle is guaranteed to be
    released in a finally block to immediately free the hardware lock.

    Args:
        device_index: Camera device index (defaults to 0 for default webcam).

    Returns:
        CapturedFrame containing the in-memory frame and dimensions.

    Raises:
        CameraUnavailableError: If the camera cannot be opened or accessed.
        FrameCaptureError: If reading a frame from the camera fails.
    """
    logger.info("Opening camera device %d using DirectShow backend", device_index)
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise CameraUnavailableError(
                f"Camera device (index {device_index}) could not be opened. "
                "The device may be disconnected, disabled in Windows privacy settings, "
                "or currently in use by another application."
            )

        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            raise FrameCaptureError(
                f"Failed to read a frame from camera device (index {device_index})."
            )

        height, width = frame.shape[:2]
        channels = frame.shape[2] if frame.ndim > 2 else 1

        logger.info(
            "Captured 1 camera frame in-memory (%dx%d, %d channels)",
            width,
            height,
            channels,
        )

        return CapturedFrame(
            frame=frame,
            width=width,
            height=height,
            channels=channels,
            format="BGR",
        )
    finally:
        cap.release()
        logger.debug("Camera device %d released", device_index)


def understand_camera(
    device_index: int = DEFAULT_CAMERA_INDEX,
) -> dict[str, Any]:
    """Capture a single camera frame locally and return structured metadata.

    Conservative MVP implementation: captures one frame into memory and
    returns truthful metadata without running visual AI models yet.

    Returns:
        Structured dictionary matching CameraResult schema.
    """
    try:
        captured = capture_camera_frame(device_index=device_index)
    except CameraError as error:
        logger.warning("Camera capture failed: %s", error)
        return {
            "status": "error",
            "visual_context": "Camera capture failed before image processing could run.",
            "response": "Could not access or capture from the camera.",
            "note": str(error),
            "frame_info": None,
        }
    except Exception as error:
        logger.exception("Unexpected error during camera capture: %s", error)
        return {
            "status": "error",
            "visual_context": "An unexpected error occurred during camera processing.",
            "response": "Camera processing encountered an internal error.",
            "note": str(error),
            "frame_info": None,
        }

    frame_info = {
        "width": captured.width,
        "height": captured.height,
        "channels": captured.channels,
        "format": captured.format,
    }
    visual_context = (
        f"Captured 1 camera frame ({captured.width}x{captured.height}, "
        f"{captured.channels} channels) locally in memory."
    )
    response = (
        f"Successfully captured a {captured.width}x{captured.height} camera frame locally. "
        "The frame is stored in memory and ready for visual understanding."
    )

    return {
        "status": "success",
        "visual_context": visual_context,
        "response": response,
        "note": None,
        "frame_info": frame_info,
    }
