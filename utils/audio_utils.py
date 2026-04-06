"""
Audio helper functions: normalization, VAD, saving.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from config.settings import settings


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1, 1]."""
    return audio.astype(np.float32) / 32768.0


def is_silent(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True if the audio segment is below the silence threshold (RMS)."""
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return rms < threshold


def save_segment(audio: np.ndarray, prefix: str = "segment") -> Path:
    """Save an audio segment to disk with a timestamp filename."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = settings.output_dir / f"{prefix}_{ts}.wav"
    sf.write(str(path), audio, settings.sample_rate)
    return path


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a wav file. Returns (audio_array, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    return audio, sr
