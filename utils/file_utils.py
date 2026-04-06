"""
JSON result persistence helpers.
"""
import json
from pathlib import Path
from datetime import datetime
from config.settings import settings


def save_result(result: dict, prefix: str = "result") -> Path:
    """Persist an analysis result dict as JSON."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = settings.output_dir / f"{prefix}_{ts}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def load_result(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
