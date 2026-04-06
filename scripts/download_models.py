"""
Pre-downloads all models so the first run is fast.
Run: python scripts/download_models.py
"""
import whisper
from config.settings import settings
from utils.logger import logger


def download_whisper():
    logger.info(f"Downloading Whisper '{settings.whisper_model}'…")
    whisper.load_model(settings.whisper_model)
    logger.info("Whisper ready.")


def download_emotion():
    logger.info("Downloading emotion model…")
    from transformers import pipeline
    pipeline("text-classification",
             model="j-hartmann/emotion-english-distilroberta-base")
    logger.info("Emotion model ready.")


def download_speaker_verifier():
    logger.info("Downloading SpeechBrain speaker verifier…")
    try:
        from speechbrain.pretrained import SpeakerRecognition
        SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speaker_verification",
        )
        logger.info("Speaker verifier ready.")
    except Exception as e:
        logger.warning(f"Skipping: {e}")


if __name__ == "__main__":
    download_whisper()
    download_emotion()
    download_speaker_verifier()
    print("\n✅ All models downloaded.")
