# 🎙️ Speech Analyzer

Live microphone speech analysis pipeline in Python.

## Features
| Module | Library | What it does |
|--------|---------|-------------|
| **Transcription** | Whisper | Speech → text |
| **Keyword Detection** | string match / Porcupine | Finds trigger words |
| **Text Emotion** | distilroberta | Emotion from transcript |
| **Voice Emotion** | SpeechBrain wav2vec2 | Emotion from audio |
| **Speaker Diarization** | pyannote.audio | Who spoke when |

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env (whisper model, API keys, keywords)

# 3. Pre-download models
python scripts/download_models.py

# 4. Run live
python main.py

# 5. Or analyse a file
python scripts/run_demo.py data/samples/test.wav
```

## Project Structure

```
speech_analyzer/
├── main.py                  # Entry point
├── config/
│   └── settings.py          # All config via pydantic + .env
├── core/
│   ├── audio_capture.py     # Mic stream (sounddevice)
│   └── pipeline.py          # Orchestrates all modules
├── modules/
│   ├── transcription.py     # Whisper STT
│   ├── keyword_detector.py  # String match + Porcupine
│   ├── emotion_analyzer.py  # Text emotion (transformers)
│   ├── voice_emotion.py     # Acoustic emotion (SpeechBrain)
│   └── speaker_diarizer.py  # Diarization + verification
├── utils/
│   ├── logger.py            # loguru logger
│   ├── audio_utils.py       # normalize, VAD, save
│   └── file_utils.py        # JSON persistence
├── scripts/
│   ├── download_models.py   # Pre-download all models
│   └── run_demo.py          # Analyse a .wav file
├── tests/                   # pytest unit tests
├── data/
│   ├── samples/             # Test audio files
│   └── output/              # Saved segments + JSON results
└── requirements.txt
```

## Optional Integrations

- **Wake word**: Set `PICOVOICE_API_KEY` and `ENABLE_WAKE_WORD=true`
- **Diarization**: Set `HUGGINGFACE_TOKEN` and `ENABLE_DIARIZATION=true`
  - Requires model agreement at https://hf.co/pyannote/speaker-diarization-3.1

## Running Tests

```bash
pytest tests/ -v
```

## Web Dashboard (New)

A real-time browser dashboard streams every analysis result over WebSocket.

```bash
# Install server deps
pip install fastapi uvicorn[standard]

# Start the server (runs mic pipeline in background)
uvicorn server:app --reload --port 8000

# Open browser
open http://localhost:8000
```

### Dashboard features
| Panel | What it shows |
|-------|--------------|
| **Live transcript** | Every chunk, coloured keyword highlights, emotion badges |
| **Emotion over time** | Multi-line Chart.js plot, one line per emotion |
| **Current scores** | Animated spark bars for all 7 emotions |
| **Speaker timeline** | Colour-coded speaker turns from pyannote |
| **Keyword alerts** | Chips for every detected keyword, running count |

### Architecture

```
Browser ←── WebSocket (/ws) ←── ConnectionManager
                                      ↑
                              asyncio broadcast
                                      ↑
                           background thread
                           AudioCapture → SpeechPipeline
```
