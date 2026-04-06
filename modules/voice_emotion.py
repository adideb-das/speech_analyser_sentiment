"""
Acoustic (audio-based) emotion recognition using SpeechBrain.
Model: speechbrain/emotion-recognition-wav2vec2-IEMOCAP
Labels: anger, happiness, sadness, neutral
"""
import torch
import numpy as np
from pathlib import Path
from utils.logger import logger
from utils.audio_utils import save_segment
from config.settings import settings


class VoiceEmotionAnalyzer:
    MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

    def __init__(self):
        logger.info("Loading SpeechBrain voice emotion model…")
        try:
            from speechbrain.pretrained.interfaces import foreign_class
            self._classifier = foreign_class(
                source=self.MODEL,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir="models/voice_emotion",
            )
            self._available = True
        except Exception as e:
            logger.warning(f"VoiceEmotionAnalyzer unavailable: {e}")
            self._available = False

    def analyze(self, audio: np.ndarray) -> dict:
        """
        Args:
            audio: float32, shape (N,), sample_rate == 16000
        Returns:
            {"label": str, "score": float}
        """
        if not self._available:
            return {"label": "unavailable", "score": 0.0}

        tmp = save_segment(audio, prefix="voice_emo_tmp")
        out_prob, score, index, label = self._classifier.classify_file(str(tmp))
        tmp.unlink(missing_ok=True)

        result = {"label": label[0], "score": float(score)}
        logger.debug(f"Voice emotion: {result}")
        return result
