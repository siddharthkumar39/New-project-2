"""Isolated Windows Graphics Capture proof-of-concept.

This script deliberately does not import anything from SnapSight. It captures one
frame from the primary display through Windows.Graphics.Capture, saves it as
tools/capture_test.png, and prints diagnostics suitable for debugging.
"""

from __future__ import annotations

import asyncio
import ctypes
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

from winrt.windows.graphics.capture import Direct3D11CaptureFramePool
from winrt.windows.graphics.capture import interop as capture_interop
from winrt.windows.graphics.directx import DirectXPixelFormat
from winrt.windows.graphics.directx.direct3d11 import interop as d3d_interop
from winrt.windows.graphics.imaging import BitmapEncoder, SoftwareBitmap
from winrt.windows.storage import CreationCollisionOption, FileAccessMode, StorageFolder


TOOLS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = TOOLS_DIR / "capture_test.png"
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
        import uuid

        raw = uuid.UUID(value).bytes_le
        return cls.from_buffer_copy(raw)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def check_hresult(result: int, operation: str) -> None:
    if result < 0:
        raise OSError(f"{operation} failed with HRESULT 0x{result & 0xFFFFFFFF:08X}")


def release_com(pointer: ctypes.c_void_p | None) -> None:
    if not pointer or not pointer.value:
        return
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
    release(pointer)


def create_winrt_d3d_device():
    """Create a D3D11 device then expose it as the WinRT IDirect3DDevice type."""
    d3d_device = ctypes.c_void_p()
    context = ctypes.c_void_p()
    feature_level = ctypes.c_uint()
    d3d11_create_device = ctypes.windll.d3d11.D3D11CreateDevice
    d3d11_create_device.argtypes = [
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
    d3d11_create_device.restype = ctypes.c_long
    check_hresult(
        d3d11_create_device(
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
        check_hresult(
            query_interface(d3d_device, ctypes.byref(iid), ctypes.byref(dxgi_device)),
            "ID3D11Device.QueryInterface(IDXGIDevice)",
        )
        return d3d_interop.create_direct3d11_device_from_dxgi_device(dxgi_device.value)
    finally:
        release_com(dxgi_device)
        release_com(context)
        release_com(d3d_device)


def primary_monitor_handle() -> int:
    user32 = ctypes.windll.user32
    user32.MonitorFromPoint.argtypes = [POINT, ctypes.c_uint]
    user32.MonitorFromPoint.restype = ctypes.c_void_p
    monitor = user32.MonitorFromPoint(POINT(0, 0), MONITOR_DEFAULTTOPRIMARY)
    if not monitor:
        raise OSError("MonitorFromPoint could not find the primary display.")
    return monitor


async def save_png(frame, output_path: Path) -> None:
    bitmap = await SoftwareBitmap.create_copy_from_surface_async(frame.surface)
    folder = await StorageFolder.get_folder_from_path_async(str(output_path.parent))
    storage_file = await folder.create_file_async(
        output_path.name, CreationCollisionOption.REPLACE_EXISTING
    )
    stream = await storage_file.open_async(FileAccessMode.READ_WRITE)
    try:
        encoder = await BitmapEncoder.create_async(BitmapEncoder.png_encoder_id, stream)
        encoder.set_software_bitmap(bitmap)
        await encoder.flush_async()
    finally:
        stream.close()
        bitmap.close()


def main() -> int:
    print(f"Windows version: {platform.platform()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Output path: {OUTPUT_PATH}")

    frame_ready = threading.Event()
    captured: list[object] = []
    frame_pool = None
    session = None

    try:
        item = capture_interop.create_for_monitor(primary_monitor_handle())
        print(f"Display resolution: {item.size.width}x{item.size.height}")
        print("Capture API initialization: primary monitor capture item created")

        device = create_winrt_d3d_device()
        print("Capture API initialization: D3D11/WinRT device created")
        frame_pool = Direct3D11CaptureFramePool.create_free_threaded(
            device,
            DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED,
            1,
            item.size,
        )

        def on_frame_arrived(sender, _args) -> None:
            frame = sender.try_get_next_frame()
            if frame is not None and not captured:
                captured.append(frame)
                frame_ready.set()

        frame_pool.add_frame_arrived(on_frame_arrived)
        session = frame_pool.create_capture_session(item)
        session.start_capture()
        print(f"Capture API initialization: session started; waiting up to {WAIT_SECONDS} seconds")

        if not frame_ready.wait(WAIT_SECONDS):
            raise TimeoutError("Windows.Graphics.Capture did not deliver a frame before timeout.")

        frame = captured[0]
        print(f"Frame dimensions: {frame.content_size.width}x{frame.content_size.height}")
        asyncio.run(save_png(frame, OUTPUT_PATH))
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
            raise RuntimeError("PNG encoding completed, but the output file is missing or empty.")

        print(f"PASS: saved real Windows.Graphics.Capture frame ({OUTPUT_PATH.stat().st_size} bytes)")
        return 0
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        traceback.print_exc()
        return 1
    finally:
        if session is not None:
            session.close()
        if frame_pool is not None:
            frame_pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
