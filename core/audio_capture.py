"""
Continuous microphone capture using sounddevice.
Fills a thread-safe queue; consumers call .read_chunk().
"""
import queue
import numpy as np
import sounddevice as sd
from config.settings import settings
from utils.logger import logger


class AudioCapture:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            logger.warning(f"Audio stream status: {status}")
        self._q.put(indata.copy())

    def start(self) -> None:
        logger.info(
            f"Opening mic stream: {settings.sample_rate}Hz, "
            f"block={settings.block_size}"
        )
        self._stream = sd.InputStream(
            samplerate=settings.sample_rate,
            channels=settings.channels,
            dtype="int16",
            blocksize=settings.block_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Mic stream closed.")

    def read_chunk(self, duration: float | None = None) -> np.ndarray:
        """
        Block until `duration` seconds of audio have been collected,
        then return them as a flat int16 array.
        """
        dur = duration or settings.chunk_duration
        num_blocks = int(settings.sample_rate * dur / settings.block_size)
        frames = [self._q.get() for _ in range(num_blocks)]
        return np.concatenate(frames, axis=0).flatten()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
