"""Configuration loading.

YAML files hold *parameters*; the environment holds *secrets*. The two never
mix: :class:`Credentials` reads only from ``os.environ`` and its ``__repr__``
never shows the values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .logging_utils import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        path = candidate if candidate.exists() else path
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


@dataclass(frozen=True)
class Credentials:
    """Bitvavo credentials, loaded from environment variables only.

    ``None`` for both fields is a fully supported state: all research and
    backtesting runs on public endpoints.
    """

    api_key: str | None = None
    api_secret: str | None = None
    access_window_ms: int = 10_000

    @classmethod
    def from_env(cls) -> Credentials:
        key = os.environ.get("BITVAVO_API_KEY") or None
        secret = os.environ.get("BITVAVO_API_SECRET") or None
        try:
            window = int(os.environ.get("BITVAVO_ACCESS_WINDOW_MS") or 10_000)
        except ValueError:
            window = 10_000
        if bool(key) != bool(secret):
            log.warning(
                "Only one of BITVAVO_API_KEY / BITVAVO_API_SECRET is set; "
                "falling back to unauthenticated public access."
            )
            key = secret = None
        return cls(api_key=key, api_secret=secret, access_window_ms=window)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def __repr__(self) -> str:  # never leak secrets through a traceback
        state = "authenticated" if self.is_authenticated else "public-only"
        return f"Credentials({state})"

    __str__ = __repr__


@dataclass
class Config:
    """Merged view of the three YAML configuration files."""

    research: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    paper: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    @classmethod
    def load(
        cls,
        research_path: str | Path = "config/research.yaml",
        risk_path: str | Path = "config/risk.yaml",
        paper_path: str | Path = "config/paper_trading.yaml",
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        cfg = cls(
            research=load_yaml(research_path),
            risk=load_yaml(risk_path),
            paper=load_yaml(paper_path),
        )
        if overrides:
            for section, value in overrides.items():
                current = getattr(cfg, section, None)
                if isinstance(current, dict) and isinstance(value, dict):
                    setattr(cfg, section, _deep_merge(current, value))
        return cfg

    # -- convenience accessors -------------------------------------------------
    def get(self, section: str, *keys: str, default: Any = None) -> Any:
        node: Any = getattr(self, section)
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, key: str) -> Path:
        """Resolve a configured data path against the project root."""
        raw = self.get("research", "paths", key, default=f"data/{key.replace('_dir', '')}")
        env_root = os.environ.get("BMA_DATA_DIR")
        p = Path(raw)
        if env_root and not p.is_absolute():
            # Allow relocating the whole data tree with one variable.
            parts = p.parts
            p = Path(env_root).joinpath(*parts[1:]) if parts and parts[0] == "data" else Path(env_root) / p
        if not p.is_absolute():
            p = self.root / p
        return p

    def ensure_dirs(self) -> None:
        for key in ("raw_dir", "processed_dir", "results_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)

    # -- safety ---------------------------------------------------------------
    def live_trading_allowed(self, cli_ack: bool = False) -> bool:
        """Three independent switches must agree before any order can be sent."""
        cfg_flag = bool(self.get("paper", "live_trading", "enabled", default=False))
        env_flag = os.environ.get("ALLOW_LIVE_TRADING", "false").strip().lower() == "true"
        allowed = cfg_flag and env_flag and bool(cli_ack)
        if not allowed:
            log.debug(
                "Live trading blocked (config=%s env=%s cli_ack=%s)", cfg_flag, env_flag, cli_ack
            )
        return allowed


def load_dotenv_if_present(path: str | Path = ".env") -> bool:
    """Load ``.env`` if python-dotenv is installed and the file exists."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        log.warning("python-dotenv not installed; skipping %s", p)
        return False
    load_dotenv(p, override=False)
    log.info("Loaded environment overrides from %s (values not logged)", p.name)
    return True
