"""Centralized logging configuration. Import setup_logging() once at app startup."""

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(level=logging.INFO):
    """Configure root logger to write to both file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any previous log file so each session starts fresh
    if LOG_FILE.exists():
        LOG_FILE.write_text("")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler — captures everything
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — warnings and above only
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    # Avoid duplicate handlers on Streamlit re-runs
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('app.log') for h in root.handlers):
        root.addHandler(fh)
        root.addHandler(ch)
    root.setLevel(level)

    # Suppress noisy third-party loggers
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.getLogger("app").info("=== Session started ===")
