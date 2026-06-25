"""Shared utilities: config loading, logging, date helpers."""
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent


def get_root() -> Path:
    return _ROOT


def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        lf = _ROOT / log_file
        lf.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(lf))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)
    return logging.getLogger("swing")


def ensure_dirs() -> None:
    for d in ["data/raw", "data/processed", "reports"]:
        (_ROOT / d).mkdir(parents=True, exist_ok=True)


def trading_days_between(start, end, calendar=None) -> int:
    """Approximate trading-day count using pandas bdate_range."""
    import pandas as pd
    return len(pd.bdate_range(start=start, end=end))
