"""FastAPI entry point for the first SnapSight module: Understand Screen."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import STATIC_DIR
from backend.schemas import CameraResult, ScreenResult, VoiceResult
import ctranslate2  # Pre-load modern C++ runtime before winrt loads bundled MSVCP140.dll
from backend.camera import understand_camera
from backend.screen import understand_screen
from backend.speech import understand_voice

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SnapSight", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "module": "screen"}


@app.post("/api/screen", response_model=ScreenResult)
def analyze_screen() -> dict[str, str | None]:
    """Run one capture only; this keeps the MVP private and easy to test."""
    try:
        return understand_screen()
    except RuntimeError as error:
        logger.warning("Screen module request failed: %s", error)
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/voice", response_model=VoiceResult)
def analyze_voice() -> dict[str, str | None]:
    """Capture a short microphone clip and transcribe it locally."""
    try:
        return understand_voice()
    except Exception as error:
        logger.exception("Voice module request failed unexpectedly: %s", error)
        return {
            "status": "error",
            "transcript": "",
            "audio_context": "An unexpected error occurred during voice processing.",
            "response": "Voice processing encountered an internal error.",
            "note": str(error),
        }


@app.post("/api/camera", response_model=CameraResult)
def analyze_camera() -> dict[str, Any]:
    """Capture a single frame from the camera and report frame metadata."""
    try:
        return understand_camera()
    except Exception as error:
        logger.exception("Camera module request failed unexpectedly: %s", error)
        return {
            "status": "error",
            "visual_context": "An unexpected error occurred during camera processing.",
            "response": "Camera processing encountered an internal error.",
            "note": str(error),
            "frame_info": None,
        }

