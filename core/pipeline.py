"""
SpeechPipeline orchestrates all analysis modules for each audio chunk.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

from config.settings import settings
from utils.audio_utils import normalize_audio, is_silent, save_segment
from utils.file_utils import save_result
from utils.logger import logger
from modules.transcription import Transcriber
from modules.keyword_detector import KeywordDetector
from modules.emotion_analyzer import EmotionAnalyzer
from modules.voice_emotion import VoiceEmotionAnalyzer
from modules.speaker_diarizer import SpeakerDiarizer


@dataclass
class AnalysisResult:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    transcript: str = ""
    language: str = ""
    keywords_found: list[str] = field(default_factory=list)
    text_emotions: list[dict] = field(default_factory=list)
    voice_emotion: dict = field(default_factory=dict)
    speakers: list[dict] = field(default_factory=list)
    is_silent: bool = False
    audio_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "transcript": self.transcript,
            "language": self.language,
            "keywords_found": self.keywords_found,
            "text_emotions": self.text_emotions,
            "voice_emotion": self.voice_emotion,
            "speakers": self.speakers,
            "is_silent": self.is_silent,
            "audio_path": self.audio_path,
        }


class SpeechPipeline:
    def __init__(self):
        logger.info("Initialising SpeechPipeline …")
        self.transcriber = Transcriber()
        self.keyword_detector = KeywordDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.voice_emotion = VoiceEmotionAnalyzer()
        self.diarizer = SpeakerDiarizer()
        logger.info("SpeechPipeline ready.")

    def process(self, audio_int16: np.ndarray) -> AnalysisResult:
        """
        Full analysis pipeline for one audio chunk.

        Args:
            audio_int16: raw int16 PCM from microphone
        Returns:
            AnalysisResult dataclass
        """
        result = AnalysisResult()

        # ── Silence check ────────────────────────────────────────────────
        audio_f32 = normalize_audio(audio_int16)
        if is_silent(audio_f32):
            result.is_silent = True
            logger.debug("Chunk is silent, skipping analysis.")
            return result

        # ── Optionally save the segment ──────────────────────────────────
        if settings.save_audio_segments:
            path = save_segment(audio_f32)
            result.audio_path = str(path)

        # ── Transcription ────────────────────────────────────────────────
        transcription = self.transcriber.transcribe(audio_f32)
        result.transcript = transcription["text"]
        result.language = transcription["language"]

        # ── Keyword detection ────────────────────────────────────────────
        result.keywords_found = self.keyword_detector.detect_in_text(result.transcript)

        # ── Text emotion ─────────────────────────────────────────────────
        result.text_emotions = self.emotion_analyzer.analyze(result.transcript)

        # ── Voice emotion (acoustic) ─────────────────────────────────────
        result.voice_emotion = self.voice_emotion.analyze(audio_f32)

        # ── Speaker diarization ──────────────────────────────────────────
        result.speakers = self.diarizer.diarize(audio_f32)

        return result
