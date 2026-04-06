from setuptools import setup, find_packages

setup(
    name="speech-analyzer",
    version="0.1.0",
    description="Live speech analysis: transcription, keyword detection, emotion, diarization",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "sounddevice", "numpy", "scipy", "openai-whisper",
        "librosa", "soundfile", "transformers", "torch",
        "speechbrain", "python-dotenv", "rich", "pydantic",
        "pydantic-settings", "loguru",
    ],
    extras_require={
        "diarization": ["pyannote.audio"],
        "wakeword": ["pvporcupine"],
        "dev": ["pytest", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": [
            "speech-analyzer=main:main",
        ]
    },
)
