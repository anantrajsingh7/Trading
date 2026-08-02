"""Logging setup with credential redaction.

Every log record passes through :class:`RedactSecretsFilter`, which strips
anything that looks like an API key or signature. Secrets must never reach a log
file, and a defensive filter is cheaper than auditing every call site.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

_SECRET_ENV_VARS = (
    "BITVAVO_API_KEY",
    "BITVAVO_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_WEBHOOK_URL",
    "ALERT_SMTP_PASSWORD",
)

# Header/field names whose values must never be logged.
_SENSITIVE_PATTERNS = [
    re.compile(r"(bitvavo-access-key\s*[:=]\s*)(\S+)", re.IGNORECASE),
    re.compile(r"(bitvavo-access-signature\s*[:=]\s*)(\S+)", re.IGNORECASE),
    re.compile(r"(api[_-]?secret\s*[:=]\s*)(\S+)", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*)(\S+)", re.IGNORECASE),
    re.compile(r"(bot\d+:)([A-Za-z0-9_\-]{20,})"),
    re.compile(r"(https://discord\.com/api/webhooks/)(\S+)"),
]

REDACTED = "***REDACTED***"


class RedactSecretsFilter(logging.Filter):
    """Remove secret material from log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - logging API
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - never let logging explode
            return True
        cleaned = self.redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True

    @staticmethod
    def redact(text: str) -> str:
        for pattern in _SENSITIVE_PATTERNS:
            text = pattern.sub(lambda m: m.group(1) + REDACTED, text)
        for var in _SECRET_ENV_VARS:
            value = os.environ.get(var)
            if value and len(value) >= 8 and value in text:
                text = text.replace(value, REDACTED)
        return text


_CONFIGURED = False


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the package root logger once."""
    global _CONFIGURED
    root = logging.getLogger("bitvavo_momentum")
    if _CONFIGURED and not force:
        return root

    root.setLevel(logging.getLevelName(level) if isinstance(level, str) else level)
    root.handlers.clear()
    root.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    redactor = RedactSecretsFilter()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    stream.addFilter(redactor)
    root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Child logger under the package root."""
    if not _CONFIGURED:
        setup_logging(os.environ.get("BMA_LOG_LEVEL", "INFO"))
    short = name.split(".")[-1]
    return logging.getLogger(f"bitvavo_momentum.{short}")
