"""Windows.Graphics.Capture implementation for SnapSight's screen module.

This module owns Windows-specific resources.  The rest of the application only
receives a normal Pillow Image, so OCR and the API remain independent of WinRT.
"""

from __future__ import annotations

import asyncio
import ctypes
import threading
import uuid
from io import BytesIO

from PIL import Image
from winrt.windows.graphics.capture import Direct3D11CaptureFramePool
from winrt.windows.graphics.capture import interop as capture_interop
from winrt.windows.graphics.directx import DirectXPixelFormat
from winrt.windows.graphics.directx.direct3d11 import interop as d3d_interop
from winrt.windows.graphics.imaging import BitmapEncoder, SoftwareBitmap
from winrt.windows.storage.streams import DataReader, InMemoryRandomAccessStream


class WindowsGraphicsCaptureError(RuntimeError):
    """Raised when Windows.Graphics.Capture cannot provide a primary-display frame."""


WAIT_SECONDS = 8
D3D_DRIVER_TYPE_HARDWARE = 1
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
D3D11_SDK_VERSION = 7
MONITOR_DEFAULTTOPRIMARY = 1
IID_IDXGI_DEVICE = "{54ec77fa-1377-44e6-8c32-88fd5f44c84c}"


class GUID(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_ulong),
        ("data2", ctypes.c_ushort),
        ("data3", ctypes.c_ushort),
        ("data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "GUID":
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _check_hresult(result: int, operation: str) -> None:
    if result < 0:
        raise WindowsGraphicsCaptureError(
            f"{operation} failed with HRESULT 0x{result & 0xFFFFFFFF:08X}."
        )


def _release_com(pointer: ctypes.c_void_p | None) -> None:
    if not pointer or not pointer.value:
        return
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
    release(pointer)


def _create_winrt_d3d_device():
    """Create a D3D11 device and expose it as WinRT's IDirect3DDevice."""
    d3d_device = ctypes.c_void_p()
    context = ctypes.c_void_p()
    feature_level = ctypes.c_uint()
    create_device = ctypes.windll.d3d11.D3D11CreateDevice
    create_device.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    create_device.restype = ctypes.c_long
    _check_hresult(
        create_device(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            None,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            0,
            D3D11_SDK_VERSION,
            ctypes.byref(d3d_device),
            ctypes.byref(feature_level),
            ctypes.byref(context),
        ),
        "D3D11CreateDevice",
    )

    dxgi_device = ctypes.c_void_p()
    iid = GUID.from_string(IID_IDXGI_DEVICE)
    vtable = ctypes.cast(d3d_device, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    query_interface = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)
    )(vtable[0])
    try:
        _check_hresult(
            query_interface(d3d_device, ctypes.byref(iid), ctypes.byref(dxgi_device)),
            "ID3D11Device.QueryInterface(IDXGIDevice)",
        )
        return d3d_interop.create_direct3d11_device_from_dxgi_device(dxgi_device.value)
    finally:
        _release_com(dxgi_device)
        _release_com(context)
        _release_com(d3d_device)


def _primary_monitor_handle() -> int:
    user32 = ctypes.windll.user32
    user32.MonitorFromPoint.argtypes = [POINT, ctypes.c_uint]
    user32.MonitorFromPoint.restype = ctypes.c_void_p
    monitor = user32.MonitorFromPoint(POINT(0, 0), MONITOR_DEFAULTTOPRIMARY)
    if not monitor:
        raise WindowsGraphicsCaptureError("Windows could not find the primary display.")
    return monitor


async def _bitmap_to_image(bitmap: SoftwareBitmap) -> Image.Image:
    """Encode a WinRT bitmap to PNG in memory, then return a caller-owned image."""
    stream = InMemoryRandomAccessStream()
    reader = None
    input_stream = None
    try:
        encoder = await BitmapEncoder.create_async(BitmapEncoder.png_encoder_id, stream)
        encoder.set_software_bitmap(bitmap)
        await encoder.flush_async()

        input_stream = stream.get_input_stream_at(0)
        reader = DataReader(input_stream)
        await reader.load_async(stream.size)
        encoded_png = bytearray(int(stream.size))
        reader.read_bytes(encoded_png)
        with Image.open(BytesIO(encoded_png)) as image:
            return image.convert("RGB").copy()
    finally:
        if reader is not None:
            reader.close()
        if input_stream is not None:
            input_stream.close()
        stream.close()


async def _surface_to_image(surface) -> Image.Image:
    """Copy a GPU surface and convert it to an in-memory Pillow Image."""
    bitmap = await SoftwareBitmap.create_copy_from_surface_async(surface)
    try:
        return await _bitmap_to_image(bitmap)
    finally:
        bitmap.close()


def capture_primary_display() -> Image.Image:
    """Capture one real primary-display frame with Windows.Graphics.Capture.

    The returned Pillow Image is in memory and independent of the WinRT frame,
    allowing all WinRT resources to be released before OCR starts.
    """
    frame_ready = threading.Event()
    captured: list[object] = []
    callback_errors: list[Exception] = []
    frame_pool = None
    session = None
    frame = None
    event_token = None

    try:
        item = capture_interop.create_for_monitor(_primary_monitor_handle())
        device = _create_winrt_d3d_device()
        frame_pool = Direct3D11CaptureFramePool.create_free_threaded(
            device,
            DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED,
            1,
            item.size,
        )

        def on_frame_arrived(sender, _args) -> None:
            try:
                next_frame = sender.try_get_next_frame()
                if next_frame is not None and not captured:
                    captured.append(next_frame)
                    frame_ready.set()
            except Exception as error:  # Event callbacks cannot raise into WinRT.
                callback_errors.append(error)
                frame_ready.set()

        event_token = frame_pool.add_frame_arrived(on_frame_arrived)
        session = frame_pool.create_capture_session(item)
        session.start_capture()

        if not frame_ready.wait(WAIT_SECONDS):
            raise WindowsGraphicsCaptureError(
                "Windows.Graphics.Capture did not deliver a frame before timeout."
            )
        if callback_errors:
            raise WindowsGraphicsCaptureError("Windows.Graphics.Capture frame callback failed.") from callback_errors[0]
        if not captured:
            raise WindowsGraphicsCaptureError("Windows.Graphics.Capture returned no frame.")

        frame = captured[0]
        image = asyncio.run(_surface_to_image(frame.surface))
        if image.width == 0 or image.height == 0:
            raise WindowsGraphicsCaptureError("Windows.Graphics.Capture returned an empty image.")
        return image
    except WindowsGraphicsCaptureError:
        raise
    except Exception as error:
        raise WindowsGraphicsCaptureError(f"Windows.Graphics.Capture failed: {error}") from error
    finally:
        if frame is not None:
            frame.close()
        if frame_pool is not None and event_token is not None:
            frame_pool.remove_frame_arrived(event_token)
        if session is not None:
            session.close()
        if frame_pool is not None:
            frame_pool.close()
