"""Microphone audio capture foundation for SnapSight.

This module owns Windows microphone recording. Audio is captured directly
into memory as a 1D NumPy array (16kHz float32), ensuring zero disk writes
and strict adherence to SnapSight's privacy-first design.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE: int = 16000  # Standard rate for local Whisper models
DEFAULT_RECORD_SECONDS: float = 5.0
DEFAULT_CHANNELS: int = 1
MAX_RECORD_SECONDS: float = 60.0

DEFAULT_WHISPER_MODEL: str = "tiny.en"
DEFAULT_DEVICE: str = "cpu"
DEFAULT_COMPUTE_TYPE: str = "int8"

_model_lock = threading.Lock()
_cached_whisper_model: Any = None


class AudioRecordingError(RuntimeError):
    """Raised when audio capture fails."""


class MicrophoneNotFoundError(AudioRecordingError):
    """Raised when no functional microphone input device is found."""


class InvalidAudioConfigError(ValueError):
    """Raised when audio recording parameters are invalid."""


@dataclass(frozen=True)
class RecordedAudio:
    """In-memory representation of captured microphone audio."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    channels: int = 1

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def actual_duration(self) -> float:
        return self.num_samples / self.sample_rate if self.sample_rate > 0 else 0.0


def validate_audio_config(
    duration_seconds: float,
    sample_rate: int,
    channels: int = DEFAULT_CHANNELS,
) -> None:
    """Validate audio capture parameters."""
    if duration_seconds <= 0 or duration_seconds > MAX_RECORD_SECONDS:
        raise InvalidAudioConfigError(
            f"Recording duration must be between 0 and {MAX_RECORD_SECONDS} seconds, "
            f"got {duration_seconds}."
        )

    if sample_rate <= 0 or sample_rate > 192000:
        raise InvalidAudioConfigError(
            f"Sample rate must be a positive rate up to 192000 Hz, got {sample_rate}."
        )

    if channels != 1:
        raise InvalidAudioConfigError(
            f"Only mono audio (channels=1) is supported for voice processing, got {channels}."
        )


def get_default_input_device() -> dict[str, Any]:
    """Query the default audio input device on Windows.

    Raises:
        MicrophoneNotFoundError: If no input device is detected or accessible.
    """
    try:
        import sounddevice as sd
    except ImportError as error:
        raise AudioRecordingError(
            "sounddevice is not installed. Install project dependencies to enable audio."
        ) from error

    try:
        device_info = sd.query_devices(kind="input")
        if not device_info:
            raise MicrophoneNotFoundError("No default audio input device found on this system.")
        return device_info
    except Exception as error:
        logger.warning("Failed to query input audio devices: %s", error)
        raise MicrophoneNotFoundError(
            f"Microphone input device is unavailable or access was denied. Windows error: {error}"
        ) from error


def record_audio(
    duration_seconds: float = DEFAULT_RECORD_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> RecordedAudio:
    """Record audio from the primary microphone into memory.

    Args:
        duration_seconds: Time to record in seconds (default 5.0).
        sample_rate: Audio sampling frequency in Hz (default 16000).

    Returns:
        RecordedAudio dataclass holding the in-memory float32 samples.

    Raises:
        InvalidAudioConfigError: If arguments are out of bounds.
        MicrophoneNotFoundError: If no microphone device is found.
        AudioRecordingError: If recording fails during capture.
    """
    validate_audio_config(duration_seconds=duration_seconds, sample_rate=sample_rate)

    try:
        import sounddevice as sd
    except ImportError as error:
        raise AudioRecordingError(
            "sounddevice is not installed. Install project dependencies to enable audio."
        ) from error

    # Verify input device exists before starting recording
    get_default_input_device()

    num_frames = int(round(duration_seconds * sample_rate))
    logger.info(
        "Starting in-memory audio recording: %.1f seconds @ %d Hz (mono)",
        duration_seconds,
        sample_rate,
    )

    try:
        raw_samples = sd.rec(
            frames=num_frames,
            samplerate=sample_rate,
            channels=DEFAULT_CHANNELS,
            dtype="float32",
        )
        sd.wait()
    except Exception as error:
        logger.exception("Microphone recording failed")
        raise AudioRecordingError(
            f"Microphone recording failed during capture: {error}"
        ) from error

    if raw_samples is None or raw_samples.size == 0:
        raise AudioRecordingError("Audio recording completed but returned no samples.")

    # Flatten (N, 1) array to 1D array of shape (N,)
    samples_1d = raw_samples.flatten()

    return RecordedAudio(
        samples=samples_1d,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        channels=DEFAULT_CHANNELS,
    )


def get_whisper_model(
    model_size_or_path: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> Any:
    """Lazily load and cache the faster-whisper WhisperModel singleton.

    Thread-safe and reused across backend requests to avoid reloading overhead.
    """
    global _cached_whisper_model
    if _cached_whisper_model is not None:
        return _cached_whisper_model

    with _model_lock:
        if _cached_whisper_model is not None:
            return _cached_whisper_model

        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install project dependencies to enable speech-to-text."
            ) from error

        try:
            logger.info(
                "Initializing faster-whisper model '%s' (device=%s, compute_type=%s)...",
                model_size_or_path,
                device,
                compute_type,
            )
            _cached_whisper_model = WhisperModel(
                model_size_or_path=model_size_or_path,
                device=device,
                compute_type=compute_type,
            )
            return _cached_whisper_model
        except Exception as error:
            logger.exception("Failed to load Whisper model")
            raise RuntimeError(
                f"Failed to load Whisper model '{model_size_or_path}': {error}"
            ) from error


def transcribe_audio(audio: RecordedAudio) -> tuple[str, str | None]:
    """Transcribe in-memory audio using local faster-whisper.

    Args:
        audio: In-memory RecordedAudio instance with float32 samples.

    Returns:
        tuple[str, str | None]: (transcript, note).
        If transcription is successful, transcript contains text and note is None.
        If empty or an error occurs, transcript is "" and note describes the issue.
    """
    if audio is None or audio.num_samples == 0:
        return "", "No audio samples were provided for transcription."

    try:
        model = get_whisper_model()
    except RuntimeError as error:
        return "", str(error)

    try:
        # faster-whisper accepts 1D float32 numpy array directly
        segments, _info = model.transcribe(audio.samples, beam_size=1)
        transcript = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    except Exception as error:
        logger.exception("Speech transcription failed during inference")
        return "", f"Speech transcription failed during inference: {error}"

    if not transcript:
        return "", "No spoken words were detected in the audio recording."

    return transcript, None


def understand_voice(
    duration_seconds: float = DEFAULT_RECORD_SECONDS,
) -> dict[str, str | None]:
    """Record audio from the microphone and transcribe it locally.

    Returns a structured dictionary consistent with SnapSight module patterns.
    """
    try:
        audio = record_audio(duration_seconds=duration_seconds)
    except AudioRecordingError as error:
        logger.warning("Voice recording failed: %s", error)
        return {
            "status": "error",
            "transcript": "",
            "audio_context": "Audio recording failed before transcription could run.",
            "response": "Could not access or record from the microphone.",
            "note": str(error),
        }

    audio_context = (
        f"Recorded {audio.actual_duration:.1f} seconds "
        f"({audio.sample_rate} Hz mono) locally in memory."
    )

    transcript, note = transcribe_audio(audio)

    if transcript:
        response = f'I heard your voice input: "{transcript}"'
        status = "success"
    else:
        response = (
            "I recorded your audio, but I could not detect clear "
            "spoken speech to transcribe."
        )
        status = "partial"

    return {
        "status": status,
        "transcript": transcript,
        "audio_context": audio_context,
        "response": response,
        "note": note,
    }

