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
