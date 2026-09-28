from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ScreenResult(BaseModel):
    """The small, predictable response sent from the screen module to the UI."""

    status: str = Field(description="success, partial, or error")
    screen_text: str = Field(description="Text extracted from the screenshot, when available")
    visual_context: str = Field(description="A conservative description based on observed input")
    response: str = Field(description="A helpful answer that never assumes unseen content")
    note: str | None = Field(default=None, description="Setup or limitation information")


class VoiceResult(BaseModel):
    """The structured response sent from the voice module to the UI."""

    status: str = Field(description="success, partial, or error")
    transcript: str = Field(description="Text transcribed from the audio, when available")
    audio_context: str = Field(description="A conservative description based on recorded input")
    response: str = Field(description="A helpful answer that never assumes unseen content")
    note: str | None = Field(default=None, description="Setup or limitation information")


class CameraResult(BaseModel):
    """The structured response sent from the camera module to the UI."""

    status: str = Field(description="success, partial, or error")
    visual_context: str = Field(description="A conservative description based on captured frame")
    response: str = Field(description="A helpful answer that never assumes unseen content")
    note: str | None = Field(default=None, description="Setup or limitation information")
    objects: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detected visual objects with labels and confidence scores",
    )
    frame_info: dict[str, Any] | None = Field(
        default=None, description="Metadata describing the captured frame (dimensions, channels, format)"
    )


