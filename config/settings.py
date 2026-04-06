"""
Centralized configuration using pydantic + .env
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    # API Keys
    picovoice_api_key: str = ""
    huggingface_token: str = ""

    # Whisper
    whisper_model: str = "base"          # tiny | base | small | medium | large

    # Audio
    sample_rate: int = 16000
    chunk_duration: int = 5              # seconds per analysis window
    channels: int = 1
    block_size: int = 1024

    # Wake word / keywords
    keywords: List[str] = ["help", "stop", "emergency", "hello", "alert"]
    wake_word: str = "hey speech"        # Porcupine wake word

    # Paths
    output_dir: Path = Path("data/output")
    sample_dir: Path = Path("data/samples")

    # Logging
    log_level: str = "INFO"

    # Feature flags
    enable_wake_word: bool = False       # set True when Porcupine key available
    enable_diarization: bool = False     # set True when HF token available
    save_audio_segments: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
