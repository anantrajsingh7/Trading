"""Safety invariants: no live trading, no leaked secrets, locked test set."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from bitvavo_momentum.config import Config, Credentials
from bitvavo_momentum.logging_utils import REDACTED, RedactSecretsFilter, setup_logging
from bitvavo_momentum.paper_trader import LiveTradingDisabled, assert_live_trading_allowed
from bitvavo_momentum.walk_forward import TestSetLocked, require_test_set, unlock_test_set

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# live trading
# --------------------------------------------------------------------------- #
def test_live_trading_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    config = Config.load()
    assert not config.live_trading_allowed(cli_ack=True)


def test_live_trading_needs_all_three_switches(monkeypatch):
    config = Config.load()
    config.paper.setdefault("live_trading", {})["enabled"] = True

    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    assert not config.live_trading_allowed(cli_ack=True), "env switch missing must block"

    monkeypatch.setenv("ALLOW_LIVE_TRADING", "true")
    assert not config.live_trading_allowed(cli_ack=False), "CLI acknowledgement missing must block"
    assert config.live_trading_allowed(cli_ack=True)


def test_assert_live_trading_always_raises(monkeypatch):
    """Even with every switch set there is no order-placement path."""
    config = Config.load()
    with pytest.raises(LiveTradingDisabled):
        assert_live_trading_allowed(config, cli_ack=False)

    config.paper.setdefault("live_trading", {})["enabled"] = True
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "true")
    with pytest.raises(LiveTradingDisabled):
        assert_live_trading_allowed(config, cli_ack=True)


def test_client_exposes_no_order_placement_method():
    from bitvavo_momentum.bitvavo_client import BitvavoClient

    forbidden = [name for name in dir(BitvavoClient)
                 if any(word in name.lower() for word in ("place", "order", "withdraw", "cancel"))]
    assert not forbidden, f"client must not expose trading methods, found: {forbidden}"


# --------------------------------------------------------------------------- #
# secrets
# --------------------------------------------------------------------------- #
def test_credentials_repr_never_shows_the_key(monkeypatch):
    monkeypatch.setenv("BITVAVO_API_KEY", "super-secret-key-value-123456")
    monkeypatch.setenv("BITVAVO_API_SECRET", "super-secret-secret-value-123456")
    credentials = Credentials.from_env()
    assert credentials.is_authenticated
    assert "super-secret" not in repr(credentials)
    assert "super-secret" not in str(credentials)


def test_half_configured_credentials_fall_back_to_public(monkeypatch):
    monkeypatch.setenv("BITVAVO_API_KEY", "only-the-key")
    monkeypatch.delenv("BITVAVO_API_SECRET", raising=False)
    assert not Credentials.from_env().is_authenticated


def test_log_filter_redacts_headers_and_env_values(monkeypatch):
    monkeypatch.setenv("BITVAVO_API_SECRET", "abcdef1234567890abcdef")
    text = (
        "bitvavo-access-key: MYKEY123456789 "
        "bitvavo-access-signature: deadbeefcafe "
        "leaked=abcdef1234567890abcdef"
    )
    cleaned = RedactSecretsFilter.redact(text)
    assert "MYKEY123456789" not in cleaned
    assert "deadbeefcafe" not in cleaned
    assert "abcdef1234567890abcdef" not in cleaned
    assert REDACTED in cleaned


def test_logger_applies_the_redaction_filter(monkeypatch, caplog):
    monkeypatch.setenv("BITVAVO_API_KEY", "PLAINTEXTKEY0123456789")
    setup_logging("DEBUG", force=True)
    logger = logging.getLogger("bitvavo_momentum.test")
    handler = logging.getLogger("bitvavo_momentum").handlers[0]
    record = logger.makeRecord(
        "bitvavo_momentum.test", logging.INFO, __file__, 1,
        "using key PLAINTEXTKEY0123456789", None, None,
    )
    for log_filter in handler.filters:
        log_filter.filter(record)
    assert "PLAINTEXTKEY0123456789" not in record.getMessage()


def test_gitignore_excludes_env_and_data():
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", "data/raw/", "data/processed/", "*.key"):
        assert pattern in content, f"{pattern} must be git-ignored"


def test_no_secrets_are_tracked_by_git():
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    tracked = result.stdout.splitlines()
    offenders = [f for f in tracked if f == ".env" or f.endswith(".key") or f.endswith(".pem")]
    assert not offenders, f"secret-looking files are tracked: {offenders}"


def test_no_hardcoded_api_keys_in_source():
    """Guard against a credential being pasted into a module during debugging."""
    suspicious: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "os.environ" in line or "_env(" in line:
                continue
            lowered = stripped.lower()
            if ("api_key" in lowered or "api_secret" in lowered) and "=" in stripped:
                value = stripped.split("=", 1)[1].strip()
                if value.startswith(('"', "'")) and len(value) > 12:
                    suspicious.append(f"{path.name}: {stripped[:80]}")
    assert not suspicious, f"possible hardcoded credentials: {suspicious}"


# --------------------------------------------------------------------------- #
# test-set lock
# --------------------------------------------------------------------------- #
def test_test_set_is_locked_by_default():
    config = Config.load()
    assert not unlock_test_set(config.research)
    with pytest.raises(TestSetLocked):
        require_test_set(config.research)


def test_test_set_can_be_unlocked_explicitly():
    config = Config.load()
    config.research.setdefault("splits", {})["unlock_test_set"] = True
    assert unlock_test_set(config.research)
    require_test_set(config.research)  # must not raise


# --------------------------------------------------------------------------- #
# scanner refuses unapproved parameters
# --------------------------------------------------------------------------- #
def test_scanner_refuses_without_an_approved_card(tmp_path):
    from bitvavo_momentum.scanner import StrategyCard, StrategyCardMissing

    with pytest.raises(StrategyCardMissing):
        StrategyCard.load(tmp_path / "does_not_exist.json")
