"""Unit tests for SpeakerDiarizer."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from modules.speaker_diarizer import SpeakerDiarizer


def test_diarize_skips_when_disabled():
    """When diarization is disabled, should return empty list."""
    diarizer = SpeakerDiarizer()  # disabled by default in test env
    audio = np.zeros(16000, dtype=np.float32)
    result = diarizer.diarize(audio)
    assert result == []
