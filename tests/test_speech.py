import numpy as np
import pytest

from backend import speech
from backend.speech import (
    AudioRecordingError,
    InvalidAudioConfigError,
    MicrophoneNotFoundError,
    RecordedAudio,
    get_default_input_device,
    record_audio,
    validate_audio_config,
)


def test_validate_audio_config_valid():
    # Should not raise
    validate_audio_config(duration_seconds=5.0, sample_rate=16000, channels=1)
    validate_audio_config(duration_seconds=0.5, sample_rate=44100, channels=1)


@pytest.mark.parametrize(
    "duration,sample_rate,channels",
    [
        (0.0, 16000, 1),
        (-1.0, 16000, 1),
        (65.0, 16000, 1),
        (5.0, 0, 1),
        (5.0, -16000, 1),
        (5.0, 200000, 1),
        (5.0, 16000, 2),
    ],
)
def test_validate_audio_config_invalid(duration, sample_rate, channels):
    with pytest.raises(InvalidAudioConfigError):
        validate_audio_config(
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
        )


def test_recorded_audio_properties():
    fake_samples = np.zeros(32000, dtype=np.float32)
    audio = RecordedAudio(
        samples=fake_samples,
        sample_rate=16000,
        duration_seconds=2.0,
        channels=1,
    )
    assert audio.num_samples == 32000
    assert audio.actual_duration == 2.0
    assert audio.channels == 1


def test_get_default_input_device_success(monkeypatch):
    import sounddevice as sd

    fake_device = {
        "name": "Microphone (Realtek(R) Audio)",
        "max_input_channels": 2,
        "default_samplerate": 44100.0,
    }
    monkeypatch.setattr(sd, "query_devices", lambda kind: fake_device if kind == "input" else None)

    device = get_default_input_device()
    assert device["name"] == "Microphone (Realtek(R) Audio)"


def test_get_default_input_device_missing_or_error(monkeypatch):
    import sounddevice as sd

    def mock_query_error(kind):
        raise RuntimeError("No audio endpoint found")

    monkeypatch.setattr(sd, "query_devices", mock_query_error)

    with pytest.raises(MicrophoneNotFoundError, match="unavailable or access was denied"):
        get_default_input_device()


def test_record_audio_success(monkeypatch):
    import sounddevice as sd

    monkeypatch.setattr(speech, "get_default_input_device", lambda: {"name": "Mock Mic"})

    expected_frames = 16000  # 1.0s @ 16kHz
    mock_samples = np.full((expected_frames, 1), 0.05, dtype=np.float32)

    recorded_calls = []

    def mock_rec(frames, samplerate, channels, dtype):
        recorded_calls.append((frames, samplerate, channels, dtype))
        return mock_samples

    monkeypatch.setattr(sd, "rec", mock_rec)
    monkeypatch.setattr(sd, "wait", lambda: None)

    audio = record_audio(duration_seconds=1.0, sample_rate=16000)

    assert isinstance(audio, RecordedAudio)
    assert audio.sample_rate == 16000
    assert audio.duration_seconds == 1.0
    assert audio.num_samples == 16000
    assert audio.samples.ndim == 1  # Flattened to 1D
    assert np.allclose(audio.samples, 0.05)
    assert recorded_calls == [(16000, 16000, 1, "float32")]


def test_record_audio_device_missing_raises_error(monkeypatch):
    def mock_no_device():
        raise MicrophoneNotFoundError("Mock: No device")

    monkeypatch.setattr(speech, "get_default_input_device", mock_no_device)

    with pytest.raises(MicrophoneNotFoundError, match="Mock: No device"):
        record_audio(duration_seconds=1.0, sample_rate=16000)


def test_record_audio_failure_during_rec(monkeypatch):
    import sounddevice as sd

    monkeypatch.setattr(speech, "get_default_input_device", lambda: {"name": "Mock Mic"})

    def mock_rec_fail(*args, **kwargs):
        raise RuntimeError("PortAudio device disconnected")

    monkeypatch.setattr(sd, "rec", mock_rec_fail)

    with pytest.raises(AudioRecordingError, match="Microphone recording failed during capture"):
        record_audio(duration_seconds=1.0, sample_rate=16000)


def test_record_audio_empty_samples(monkeypatch):
    import sounddevice as sd

    monkeypatch.setattr(speech, "get_default_input_device", lambda: {"name": "Mock Mic"})
    monkeypatch.setattr(sd, "rec", lambda *args, **kwargs: np.array([], dtype=np.float32))
    monkeypatch.setattr(sd, "wait", lambda: None)

    with pytest.raises(AudioRecordingError, match="returned no samples"):
        record_audio(duration_seconds=1.0, sample_rate=16000)


def test_get_whisper_model_reuses_cached(monkeypatch):
    fake_model = object()
    monkeypatch.setattr(speech, "_cached_whisper_model", fake_model)
    assert speech.get_whisper_model() is fake_model


def test_get_whisper_model_initialization(monkeypatch):
    monkeypatch.setattr(speech, "_cached_whisper_model", None)
    mock_calls = []

    class MockWhisperModel:
        def __init__(self, model_size_or_path, device, compute_type):
            mock_calls.append((model_size_or_path, device, compute_type))

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", MockWhisperModel)

    model = speech.get_whisper_model()
    assert isinstance(model, MockWhisperModel)
    assert mock_calls == [("tiny.en", "cpu", "int8")]


def test_get_whisper_model_load_failure(monkeypatch):
    monkeypatch.setattr(speech, "_cached_whisper_model", None)

    def mock_fail(*args, **kwargs):
        raise RuntimeError("Corrupt model weights")

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", mock_fail)

    with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
        speech.get_whisper_model()


def test_transcribe_audio_empty_audio():
    text, note = speech.transcribe_audio(None)
    assert text == ""
    assert "No audio samples" in note

    empty_audio = RecordedAudio(
        samples=np.array([], dtype=np.float32),
        sample_rate=16000,
        duration_seconds=0.0,
        channels=1,
    )
    text2, note2 = speech.transcribe_audio(empty_audio)
    assert text2 == ""
    assert "No audio samples" in note2


def test_transcribe_audio_success(monkeypatch):
    from collections import namedtuple

    Segment = namedtuple("Segment", ["text"])
    fake_segments = [Segment("  What is on "), Segment(" my screen? ")]

    class MockModel:
        def transcribe(self, samples, beam_size=1):
            return fake_segments, None

    monkeypatch.setattr(speech, "get_whisper_model", lambda: MockModel())

    audio = RecordedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        channels=1,
    )

    transcript, note = speech.transcribe_audio(audio)
    assert transcript == "What is on my screen?"
    assert note is None


def test_transcribe_audio_empty_transcript(monkeypatch):
    class MockModel:
        def transcribe(self, samples, beam_size=1):
            return [], None

    monkeypatch.setattr(speech, "get_whisper_model", lambda: MockModel())

    audio = RecordedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        channels=1,
    )

    transcript, note = speech.transcribe_audio(audio)
    assert transcript == ""
    assert "No spoken words were detected" in note


def test_transcribe_audio_inference_error(monkeypatch):
    class MockModel:
        def transcribe(self, samples, beam_size=1):
            raise RuntimeError("Engine failure")

    monkeypatch.setattr(speech, "get_whisper_model", lambda: MockModel())

    audio = RecordedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        channels=1,
    )

    transcript, note = speech.transcribe_audio(audio)
    assert transcript == ""
    assert "Speech transcription failed during inference: Engine failure" in note


def test_transcribe_audio_model_load_error(monkeypatch):
    def mock_fail_load():
        raise RuntimeError("Model download failed")

    monkeypatch.setattr(speech, "get_whisper_model", mock_fail_load)

    audio = RecordedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        channels=1,
    )

    transcript, note = speech.transcribe_audio(audio)
    assert transcript == ""
    assert "Model download failed" in note


def test_understand_voice_success(monkeypatch):
    fake_audio = RecordedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        channels=1,
    )
    monkeypatch.setattr(speech, "record_audio", lambda duration_seconds: fake_audio)
    monkeypatch.setattr(speech, "transcribe_audio", lambda audio: ("Explain this function", None))

    result = speech.understand_voice(duration_seconds=1.0)
    assert result["status"] == "success"
    assert result["transcript"] == "Explain this function"
    assert 'Explain this function' in result["response"]
    assert "16000 Hz mono" in result["audio_context"]
    assert result["note"] is None


def test_understand_voice_partial_when_no_speech(monkeypatch):
    fake_audio = RecordedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        channels=1,
    )
    monkeypatch.setattr(speech, "record_audio", lambda duration_seconds: fake_audio)
    monkeypatch.setattr(
        speech,
        "transcribe_audio",
        lambda audio: ("", "No spoken words were detected in the audio recording."),
    )

    result = speech.understand_voice(duration_seconds=1.0)
    assert result["status"] == "partial"
    assert result["transcript"] == ""
    assert "could not detect clear spoken speech" in result["response"]
    assert "No spoken words were detected" in result["note"]


def test_understand_voice_recording_error(monkeypatch):
    def mock_record_fail(duration_seconds):
        raise AudioRecordingError("Microphone disconnected")

    monkeypatch.setattr(speech, "record_audio", mock_record_fail)

    result = speech.understand_voice(duration_seconds=1.0)
    assert result["status"] == "error"
    assert result["transcript"] == ""
    assert "Could not access or record from the microphone." in result["response"]
    assert "Microphone disconnected" in result["note"]

