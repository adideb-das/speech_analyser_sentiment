"""
Entry point — live speech analysis loop.
Run: python main.py
"""
import signal
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.audio_capture import AudioCapture
from core.pipeline import SpeechPipeline, AnalysisResult
from utils.file_utils import save_result
from utils.logger import logger
from config.settings import settings

console = Console()


def render_result(result: AnalysisResult) -> None:
    """Pretty-print one analysis result to the terminal."""
    if result.is_silent:
        console.print("[dim]── silence ──[/dim]")
        return

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold cyan", width=18)
    table.add_column("Value", style="white")

    table.add_row("📝 Transcript", result.transcript or "[dim]empty[/dim]")
    table.add_row("🌍 Language", result.language)

    if result.keywords_found:
        table.add_row(
            "🔑 Keywords",
            ", ".join(f"[yellow]{k}[/yellow]" for k in result.keywords_found),
        )

    if result.text_emotions:
        top = result.text_emotions[0]
        bar = "█" * int(top["score"] * 20)
        table.add_row(
            "😶 Text Emotion",
            f"[green]{top['label']}[/green] {bar} {top['score']:.0%}",
        )

    if result.voice_emotion.get("label") not in ("unavailable", "", None):
        ve = result.voice_emotion
        table.add_row("🎙️ Voice Emotion", f"{ve['label']} ({ve['score']:.2f})")

    if result.speakers:
        spk_str = " | ".join(
            f"{s['speaker']} {s['start']}s–{s['end']}s" for s in result.speakers
        )
        table.add_row("👥 Speakers", spk_str)

    console.print(Panel(table, title=f"[bold]{result.timestamp}[/bold]", border_style="blue"))


def main():
    console.rule("[bold blue]🎙️ Live Speech Analyzer[/bold blue]")
    console.print(
        f"  Model: [cyan]{settings.whisper_model}[/cyan]  |  "
        f"Chunk: [cyan]{settings.chunk_duration}s[/cyan]  |  "
        f"Keywords: [cyan]{settings.keywords}[/cyan]\n"
    )

    pipeline = SpeechPipeline()
    stop_event = [False]

    def handle_sigint(*_):
        console.print("\n[bold red]Stopping…[/bold red]")
        stop_event[0] = True

    signal.signal(signal.SIGINT, handle_sigint)

    with AudioCapture() as mic:
        console.print("[bold green]Listening… (Ctrl+C to stop)[/bold green]\n")
        while not stop_event[0]:
            try:
                audio = mic.read_chunk()
                result = pipeline.process(audio)
                render_result(result)

                if not result.is_silent:
                    save_result(result.to_dict(), prefix="analysis")

            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)

    console.rule("[bold blue]Session ended[/bold blue]")


if __name__ == "__main__":
    main()
