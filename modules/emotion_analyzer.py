"""
Text-based emotion/sentiment analysis using HuggingFace Transformers.
Model: j-hartmann/emotion-english-distilroberta-base
Labels: anger, disgust, fear, joy, neutral, sadness, surprise
"""
from transformers import pipeline
from utils.logger import logger


class EmotionAnalyzer:
    MODEL = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self, top_k: int = 3):
        logger.info(f"Loading emotion model: {self.MODEL}")
        self._pipe = pipeline(
            "text-classification",
            model=self.MODEL,
            top_k=None,
        )
        self.top_k = top_k

    def analyze(self, text: str) -> list[dict]:
        """
        Returns top-k emotions sorted by score.
        Example: [{"label": "joy", "score": 0.87}, ...]
        """
        if not text.strip():
            return []

        results = self._pipe(text)[0]
        ranked = sorted(results, key=lambda x: x["score"], reverse=True)
        top = ranked[: self.top_k]
        logger.debug(f"Emotions: {top}")
        return top

    def dominant(self, text: str) -> dict | None:
        emotions = self.analyze(text)
        return emotions[0] if emotions else None
