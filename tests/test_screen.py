import asyncio

from PIL import Image
import pytest
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap

from backend import screen
from backend.windows_graphics_capture import _bitmap_to_image


def test_understand_screen_returns_truthful_result(monkeypatch):
    monkeypatch.setattr(screen, "capture_screen", lambda: Image.new("RGB", (800, 600)))
    monkeypatch.setattr(screen, "extract_text", lambda _image: ("Hello SnapSight", None))

    result = screen.understand_screen()

    assert result["status"] == "success"
    assert result["screen_text"] == "Hello SnapSight"
    assert "800 x 600" in result["visual_context"]


def test_capture_screen_preserves_the_winrt_capture_error(monkeypatch):
    def fail_capture():
        raise screen.WindowsGraphicsCaptureError("capture session failed")

    monkeypatch.setattr(screen, "capture_primary_display", fail_capture)

    with pytest.raises(RuntimeError, match="Windows.Graphics.Capture.*capture session failed"):
        screen.capture_screen()


def test_winrt_bitmap_is_converted_to_an_in_memory_pillow_image():
    bitmap = SoftwareBitmap(BitmapPixelFormat.BGRA8, 4, 3)
    try:
        image = asyncio.run(_bitmap_to_image(bitmap))
    finally:
        bitmap.close()

    assert image.mode == "RGB"
    assert image.size == (4, 3)


def test_resolve_tesseract_cmd_prefers_path(monkeypatch):
    monkeypatch.setattr(screen.shutil, "which", lambda cmd: r"C:\custom\bin\tesseract.exe" if cmd == "tesseract" else None)
    assert screen.resolve_tesseract_cmd() == r"C:\custom\bin\tesseract.exe"


def test_resolve_tesseract_cmd_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(screen.shutil, "which", lambda cmd: None)
    assert screen.resolve_tesseract_cmd() == screen.DEFAULT_TESSERACT_PATH


def test_extract_text_handles_missing_tesseract(monkeypatch):
    import pytesseract

    monkeypatch.setattr(screen, "resolve_tesseract_cmd", lambda: r"C:\nonexistent\tesseract.exe")

    def mock_image_to_string(_image):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", mock_image_to_string)

    image = Image.new("RGB", (100, 100))
    text, note = screen.extract_text(image)

    assert text == ""
    assert note is not None
    assert "Tesseract OCR could not be found at: C:\\nonexistent\\tesseract.exe" in note


def test_understand_screen_handles_missing_tesseract(monkeypatch):
    monkeypatch.setattr(screen, "capture_screen", lambda: Image.new("RGB", (800, 600)))
    monkeypatch.setattr(
        screen,
        "extract_text",
        lambda _image: ("", "Tesseract OCR could not be found at: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"),
    )

    result = screen.understand_screen()

    assert result["status"] == "partial"
    assert result["screen_text"] == ""
    assert "could not be found" in result["note"]
    assert "do not have enough readable local text" in result["response"]

