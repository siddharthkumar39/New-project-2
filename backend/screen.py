"""Screen capture and local, optional OCR helpers.

This file intentionally knows nothing about HTTP.  Keeping capture/analysis here
makes it easy to replace OCR with a verified Qualcomm-compatible model later.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

from PIL import Image

from backend.windows_graphics_capture import (
    WindowsGraphicsCaptureError,
    capture_primary_display,
)

logger = logging.getLogger(__name__)

# Default Windows Tesseract installation path
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_PATH = DEFAULT_TESSERACT_PATH


def resolve_tesseract_cmd() -> str:
    """Resolve the Tesseract binary path: check system PATH first, then fallback to default."""
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path
    return DEFAULT_TESSERACT_PATH


def capture_screen() -> Image.Image:
    """Capture the primary display through the verified Windows WinRT backend."""
    try:
        image = capture_primary_display()
    except WindowsGraphicsCaptureError as error:
        logger.exception("Screen capture failed")
        raise RuntimeError(
            "Windows screen capture failed before OCR could run. "
            f"Capture backend: Windows.Graphics.Capture. Windows error: {error}"
        ) from error

    if image.width == 0 or image.height == 0:
        raise RuntimeError("The captured screen image was empty.")

    return image


def extract_text(image: Any) -> tuple[str, str | None]:
    """Extract text from the captured image using local Tesseract OCR."""
    try:
        import pytesseract
    except ImportError:
        return "", (
            "OCR is not installed yet. "
            "Install the project dependencies to enable text reading."
        )

    tesseract_cmd = resolve_tesseract_cmd()
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        text = pytesseract.image_to_string(image).strip()

    except pytesseract.TesseractNotFoundError:
        return "", (
            "Tesseract OCR could not be found at: "
            f"{tesseract_cmd}"
        )

    except Exception:
        logger.exception("OCR failed")
        return "", (
            "The screenshot was captured, but local OCR could not read it."
        )

    return text, None


def understand_screen() -> dict[str, str | None]:
    """Capture one screen image, inspect it locally, and create a truthful response."""
    image = capture_screen()

    text, note = extract_text(image)

    visual_context = (
        f"Captured one {image.width} x {image.height} "
        "screen image locally."
    )

    if text:
        preview = " ".join(text.split())[:500]

        response = (
            "I found readable text on your screen. "
            "Here is the visible text preview: "
            f"{preview}"
        )

        status = "success"

    else:
        response = (
            "I captured your screen, but I do not have enough readable "
            "local text to explain what is on it yet."
        )

        status = "partial"

    return {
        "status": status,
        "screen_text": text,
        "visual_context": visual_context,
        "response": response,
        "note": note,
    }