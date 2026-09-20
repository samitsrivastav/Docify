"""Application-wide logging setup."""
import logging
import sys

from app.config.settings import settings

_configured = False


def setup_logging() -> None:
    """Configure root logging once (safe to call multiple times, e.g. on Streamlit reruns)."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(settings.LOG_DIR / "docify.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _configured = True
