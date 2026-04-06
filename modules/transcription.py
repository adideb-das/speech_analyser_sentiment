"""
Speech-to-text using OpenAI Whisper.
"""
import numpy as np
import whisper
from config.settings import settings
from utils.logger import logger


class Transcriber:
    def __init__(self):
        logger.info(f"Loading Whisper model: {settings.whisper_model}")
        self.model = whisper.load_model(settings.whisper_model)

    def transcribe(self, audio: np.ndarray, language: str = "en") -> dict:
        """
        Args:
            audio: float32 array, normalized to [-1, 1]
            language: ISO 639-1 code, or None for auto-detect
        Returns:
            {"text": str, "language": str, "segments": list}
        """
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=False,
            verbose=False,
        )
        logger.debug(f"Transcript: {result['text'][:80]!r}")
        return {
            "text": result["text"].strip(),
            "language": result.get("language", language),
            "segments": result.get("segments", []),
        }
