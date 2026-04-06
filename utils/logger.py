"""
Structured logger using loguru + rich.
"""
import sys
from loguru import logger
from config.settings import settings

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, format=LOG_FORMAT, level=settings.log_level, colorize=True)
logger.add(
    settings.output_dir / "speech_analyzer.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
)

__all__ = ["logger"]
