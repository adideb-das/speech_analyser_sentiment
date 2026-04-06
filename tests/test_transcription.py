"""Unit tests for the Transcriber module."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_transcriber():
    with patch("modules.transcription.whisper") as mock_whisper:
        model = MagicMock()
        model.transcribe.return_value = {
            "text": "hello world",
            "language": "en",
            "segments": [],
        }
        mock_whisper.load_model.return_value = model
        from modules.transcription import Transcriber
        return Transcriber()


def test_transcribe_returns_text(mock_transcriber):
    audio = np.zeros(16000, dtype=np.float32)
    result = mock_transcriber.transcribe(audio)
    assert "text" in result
    assert isinstance(result["text"], str)


def test_transcribe_empty_audio(mock_transcriber):
    audio = np.zeros(16000, dtype=np.float32)
    result = mock_transcriber.transcribe(audio)
    assert result["language"] == "en"
