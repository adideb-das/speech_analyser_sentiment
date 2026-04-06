"""
Two-mode keyword detection:
  1. Simple string matching (no API key needed)
  2. Porcupine wake-word engine (accurate, low-latency, requires key)
"""
from config.settings import settings
from utils.logger import logger


class KeywordDetector:
    def __init__(self):
        self.keywords = [kw.lower() for kw in settings.keywords]
        self._porcupine = None

        if settings.enable_wake_word and settings.picovoice_api_key:
            self._init_porcupine()

    def _init_porcupine(self):
        try:
            import pvporcupine
            self._porcupine = pvporcupine.create(
                access_key=settings.picovoice_api_key,
                keywords=["hey siri"],   # replace with custom .ppn files
            )
            logger.info("Porcupine wake-word engine loaded.")
        except Exception as e:
            logger.warning(f"Porcupine init failed: {e}. Falling back to string match.")

    # ── Text-based detection ────────────────────────────────────────────────
    def detect_in_text(self, text: str) -> list[str]:
        """Find configured keywords in a transcript string."""
        lower = text.lower()
        found = [kw for kw in self.keywords if kw in lower]
        if found:
            logger.info(f"Keywords detected: {found}")
        return found

    # ── Audio-frame based (Porcupine) ───────────────────────────────────────
    def detect_in_frame(self, pcm_frame) -> bool:
        """
        Pass a single PCM frame (int16, length == porcupine.frame_length).
        Returns True if the wake word is detected.
        Requires Porcupine to be initialised.
        """
        if not self._porcupine:
            return False
        result = self._porcupine.process(pcm_frame)
        return result >= 0

    @property
    def frame_length(self) -> int | None:
        return self._porcupine.frame_length if self._porcupine else None

    def cleanup(self):
        if self._porcupine:
            self._porcupine.delete()
