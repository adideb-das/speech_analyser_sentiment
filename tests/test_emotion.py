"""Unit tests for the EmotionAnalyzer module."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_analyzer():
    with patch("modules.emotion_analyzer.pipeline") as mock_pipe:
        mock_instance = MagicMock()
        mock_instance.return_value = [[
            {"label": "joy", "score": 0.85},
            {"label": "neutral", "score": 0.10},
        ]]
        mock_pipe.return_value = mock_instance
        from modules.emotion_analyzer import EmotionAnalyzer
        return EmotionAnalyzer(top_k=2)


def test_analyze_returns_list(mock_analyzer):
    result = mock_analyzer.analyze("I am very happy today!")
    assert isinstance(result, list)
    assert len(result) > 0


def test_analyze_empty_text(mock_analyzer):
    result = mock_analyzer.analyze("")
    assert result == []


def test_dominant_returns_top(mock_analyzer):
    result = mock_analyzer.dominant("Great news!")
    assert result is not None
    assert "label" in result and "score" in result
