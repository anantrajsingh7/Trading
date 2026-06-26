"""
Structured logging setup.
Uses rich for beautiful console output + file logging.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_console = Console()
_configured = False


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler] = [
        RichHandler(console=_console, rich_tracebacks=True, show_path=False)
    ]
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
