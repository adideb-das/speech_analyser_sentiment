"""
Offline demo: analyse a .wav file instead of live mic.
Run: python scripts/run_demo.py path/to/audio.wav
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.audio_utils import load_audio, normalize_audio
from core.pipeline import SpeechPipeline
from rich import print as rprint


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_demo.py <audio.wav>")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        sys.exit(1)

    print(f"\n🎧 Analysing: [cyan]{wav_path}[/cyan]\n")
    audio, sr = load_audio(wav_path)

    pipeline = SpeechPipeline()
    result = pipeline.process((audio * 32768).astype("int16"))

    rprint(result.to_dict())


if __name__ == "__main__":
    main()
