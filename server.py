"""
FastAPI WebSocket server.
Streams live AnalysisResult events to connected browsers.

Run: uvicorn server:app --reload --port 8000
Then open: http://localhost:8000
"""
import asyncio
import json
import threading
from pathlib import Path
from typing import Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.audio_capture import AudioCapture
from core.pipeline import SpeechPipeline
from utils.logger import logger

app = FastAPI(title="Speech Analyzer")

# ── Static dashboard files ────────────────────────────────────────────────────
DASHBOARD = Path(__file__).parent / "dashboard"
app.mount("/static", StaticFiles(directory=str(DASHBOARD)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (DASHBOARD / "index.html").read_text()


# ── Connection manager ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info(f"WS client connected ({len(self.active)} total)")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info(f"WS client disconnected ({len(self.active)} remaining)")

    async def broadcast(self, data: dict):
        if not self.active:
            return
        payload = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self.active -= dead


manager = ConnectionManager()

# ── Background pipeline thread ────────────────────────────────────────────────
pipeline: SpeechPipeline | None = None
_loop: asyncio.AbstractEventLoop | None = None
_stop_event = threading.Event()


def _run_pipeline():
    """Runs in a background thread; sends results to the asyncio event loop."""
    global pipeline
    pipeline = SpeechPipeline()

    with AudioCapture() as mic:
        logger.info("Background pipeline started.")
        while not _stop_event.is_set():
            try:
                audio = mic.read_chunk()
                result = pipeline.process(audio)
                payload = result.to_dict()

                # Enqueue broadcast back on the asyncio loop
                if _loop and not _loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast(payload), _loop
                    )
            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)


@app.on_event("startup")
async def startup():
    global _loop
    _loop = asyncio.get_running_loop()
    t = threading.Thread(target=_run_pipeline, daemon=True)
    t.start()
    logger.info("Pipeline thread launched.")


@app.on_event("shutdown")
async def shutdown():
    _stop_event.set()


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # keep connection alive; client sends pings
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
