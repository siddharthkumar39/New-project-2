"""Integration tests for FastAPI endpoints in backend/main.py."""

from fastapi.testclient import TestClient
import pytest

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "module": "screen"}


def test_api_screen_still_works(client, monkeypatch):
    from backend import main

    monkeypatch.setattr(
        main,
        "understand_screen",
        lambda: {
            "status": "success",
            "screen_text": "Sample Screen Text",
            "visual_context": "Captured one 800 x 600 screen image locally.",
            "response": "Found readable text on screen.",
            "note": None,
        },
    )

    response = client.post("/api/screen")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["screen_text"] == "Sample Screen Text"
    assert "800 x 600" in data["visual_context"]
    assert data["note"] is None


def test_api_voice_success(client, monkeypatch):
    from backend import main

    monkeypatch.setattr(
        main,
        "understand_voice",
        lambda: {
            "status": "success",
            "transcript": "Explain this function",
            "audio_context": "Recorded 5.0 seconds (16000 Hz mono) locally in memory.",
            "response": 'I heard your voice input: "Explain this function"',
            "note": None,
        },
    )

    response = client.post("/api/voice")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["transcript"] == "Explain this function"
    assert "16000 Hz mono" in data["audio_context"]
    assert "Explain this function" in data["response"]
    assert data["note"] is None


def test_api_voice_partial_silence(client, monkeypatch):
    from backend import main

    monkeypatch.setattr(
        main,
        "understand_voice",
        lambda: {
            "status": "partial",
            "transcript": "",
            "audio_context": "Recorded 5.0 seconds (16000 Hz mono) locally in memory.",
            "response": "I recorded your audio, but I could not detect clear spoken speech to transcribe.",
            "note": "No spoken words were detected in the audio recording.",
        },
    )

    response = client.post("/api/voice")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "partial"
    assert data["transcript"] == ""
    assert "could not detect clear spoken speech" in data["response"]
    assert "No spoken words were detected" in data["note"]


def test_api_voice_recording_error(client, monkeypatch):
    from backend import main

    monkeypatch.setattr(
        main,
        "understand_voice",
        lambda: {
            "status": "error",
            "transcript": "",
            "audio_context": "Audio recording failed before transcription could run.",
            "response": "Could not access or record from the microphone.",
            "note": "Microphone input device is unavailable or access was denied.",
        },
    )

    response = client.post("/api/voice")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["transcript"] == ""
    assert "Could not access or record" in data["response"]
    assert "unavailable or access was denied" in data["note"]


def test_api_voice_unexpected_exception_handled_gracefully(client, monkeypatch):
    from backend import main

    def mock_crash():
        raise RuntimeError("Unexpected audio driver crash")

    monkeypatch.setattr(main, "understand_voice", mock_crash)

    response = client.post("/api/voice")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["transcript"] == ""
    assert "Unexpected audio driver crash" in data["note"]
    assert "encountered an internal error" in data["response"]
