"""
Configuration management.
Single source of truth for all settings.
Loads config.yaml, merges with environment variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_root() -> Path:
    return _ROOT


def get_db_path() -> Path:
    cfg = get_config()
    return _ROOT / cfg["portfolio"]["db_path"]


def get_reports_dir() -> Path:
    cfg = get_config()
    p = _ROOT / cfg["reporting"]["output_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cache_dir() -> Path:
    cfg = get_config()
    p = _ROOT / cfg["data"]["cache_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs() -> None:
    for d in ["db", "logs", "reports"]:
        (_ROOT / d).mkdir(parents=True, exist_ok=True)


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)
