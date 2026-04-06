"""
Speaker diarization using pyannote.audio.
Identifies WHO spoke WHEN in an audio segment.

Requires:
  - HuggingFace token (HUGGINGFACE_TOKEN in .env)
  - Model agreement at https://hf.co/pyannote/speaker-diarization-3.1
"""
import numpy as np
from pathlib import Path
from utils.logger import logger
from utils.audio_utils import save_segment
from config.settings import settings


class SpeakerDiarizer:
    def __init__(self):
        self._pipeline = None
        if settings.enable_diarization and settings.huggingface_token:
            self._init_pipeline()

    def _init_pipeline(self):
        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.huggingface_token,
            )
            logger.info("Pyannote diarization pipeline loaded.")
        except Exception as e:
            logger.warning(f"Diarization unavailable: {e}")

    def diarize(self, audio: np.ndarray) -> list[dict]:
        """
        Args:
            audio: float32, shape (N,), sample_rate == 16000
        Returns:
            List of {"speaker": str, "start": float, "end": float}
        """
        if not self._pipeline:
            logger.debug("Diarization skipped (not configured).")
            return []

        tmp = save_segment(audio, prefix="diarize_tmp")
        diarization = self._pipeline(str(tmp))
        tmp.unlink(missing_ok=True)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
            })

        logger.debug(f"Diarization segments: {segments}")
        return segments


class SpeakerVerifier:
    """
    One-shot speaker verification: are two audio clips the same speaker?
    Uses SpeechBrain ECAPA-TDNN embeddings.
    """
    def __init__(self):
        try:
            from speechbrain.pretrained import SpeakerRecognition
            self._model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speaker_verification",
            )
            self._available = True
            logger.info("Speaker verifier loaded.")
        except Exception as e:
            logger.warning(f"SpeakerVerifier unavailable: {e}")
            self._available = False

    def verify(self, wav1: str | Path, wav2: str | Path) -> dict:
        """
        Returns {"same_speaker": bool, "score": float}
        """
        if not self._available:
            return {"same_speaker": None, "score": 0.0}

        score, decision = self._model.verify_files(str(wav1), str(wav2))
        return {"same_speaker": bool(decision), "score": float(score)}
