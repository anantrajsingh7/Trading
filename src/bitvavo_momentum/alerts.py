"""Phase 13: optional alert delivery.

Channels are opt-in through configuration; credentials come from environment
variables only. If a channel is enabled but its credentials are missing, the
alert is logged and skipped - it is never sent to a fallback destination, and the
missing credential is never echoed.

An alert is emitted only when every gate passes (:func:`should_alert`). A
separate, clearly labelled informational alert covers "momentum event detected,
but no valid entry", so an empty signal feed is distinguishable from a broken
scanner.
"""

from __future__ import annotations

import os
import smtplib
import time
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any

import requests

from .logging_utils import get_logger
from .scanner import Signal
from .timeutils import format_display, now_utc

log = get_logger(__name__)

ALERT_SIGNAL = "signal"
ALERT_INFO = "informational"
ALERT_SYSTEM = "system"


@dataclass
class AlertGateResult:
    allowed: bool
    reasons: list[str]

    def __bool__(self) -> bool:
        return self.allowed


def should_alert(
    signal: Signal,
    config: dict[str, Any],
    risk_snapshot: dict[str, Any],
    data_is_current: bool = True,
) -> AlertGateResult:
    """Every Phase 13 gate, evaluated explicitly."""
    reasons: list[str] = []
    quality = config.get("signal_quality", {})

    if signal.action != "valid":
        reasons.append(f"signal action is {signal.action}, not valid")
    if signal.action == "extended":
        reasons.append("setup is already extended beyond the tested entry zone")
    min_rr = float(quality.get("min_reward_to_risk", 1.5))
    if signal.reward_to_risk < min_rr:
        reasons.append(f"net reward:risk {signal.reward_to_risk:.2f} below minimum {min_rr:.2f}")
    max_spread = float(config.get("scanner", {}).get("max_spread_bps", 60))
    if signal.spread_bps > max_spread:
        reasons.append(f"spread {signal.spread_bps:.1f} bps above {max_spread:.0f} bps")
    if risk_snapshot.get("paused_until"):
        reasons.append(f"risk manager paused: {risk_snapshot.get('pause_reason')}")
    open_positions = int(risk_snapshot.get("open_positions", 0))
    if open_positions >= int(risk_snapshot.get("max_concurrent_positions", 3)):
        reasons.append("position limit reached")
    if not data_is_current:
        reasons.append("market data is stale or incomplete")
    return AlertGateResult(not reasons, reasons)


class AlertDispatcher:
    """Sends alerts to the configured channels, with throttling."""

    def __init__(self, config: dict[str, Any]):
        alerts_cfg = config.get("alerts", {}) if "alerts" in config else config
        self.enabled = bool(alerts_cfg.get("enabled", False))
        self.channels = [str(c).lower() for c in alerts_cfg.get("channels", [])]
        self.min_seconds_between = float(alerts_cfg.get("min_seconds_between_alerts", 60))
        self.send_informational = bool(alerts_cfg.get("send_informational_event_alerts", True))
        self._last_sent: float = 0.0
        self.sent_count = 0
        self.skipped_count = 0

    # -- credentials (environment only) ---------------------------------------
    @staticmethod
    def _env(name: str) -> str | None:
        value = os.environ.get(name)
        return value.strip() if value and value.strip() else None

    # -- transport -------------------------------------------------------------
    def _send_telegram(self, text: str) -> bool:
        token = self._env("TELEGRAM_BOT_TOKEN")
        chat_id = self._env("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            log.warning("Telegram alerts enabled but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are not set")
            return False
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
                timeout=15,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.warning("Telegram alert failed: %s", type(exc).__name__)
            return False

    def _send_discord(self, text: str) -> bool:
        url = self._env("DISCORD_WEBHOOK_URL")
        if not url:
            log.warning("Discord alerts enabled but DISCORD_WEBHOOK_URL is not set")
            return False
        try:
            response = requests.post(url, json={"content": text[:1900]}, timeout=15)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.warning("Discord alert failed: %s", type(exc).__name__)
            return False

    def _send_email(self, subject: str, text: str) -> bool:
        host = self._env("ALERT_SMTP_HOST")
        user = self._env("ALERT_SMTP_USER")
        password = self._env("ALERT_SMTP_PASSWORD")
        sender = self._env("ALERT_EMAIL_FROM") or user
        recipient = self._env("ALERT_EMAIL_TO")
        if not all([host, sender, recipient]):
            log.warning("Email alerts enabled but SMTP settings are incomplete")
            return False
        port = int(self._env("ALERT_SMTP_PORT") or 587)
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = recipient
        message.set_content(text)
        try:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(message)
            return True
        except Exception as exc:
            log.warning("Email alert failed: %s", type(exc).__name__)
            return False

    # -- public API ------------------------------------------------------------
    def send(self, subject: str, text: str, kind: str = ALERT_SIGNAL) -> bool:
        if not self.enabled or not self.channels:
            log.debug("Alerts disabled; would have sent: %s", subject)
            return False
        if kind == ALERT_INFO and not self.send_informational:
            return False
        elapsed = time.time() - self._last_sent
        if elapsed < self.min_seconds_between:
            self.skipped_count += 1
            log.debug("Alert throttled (%.0fs since last)", elapsed)
            return False

        delivered = False
        for channel in self.channels:
            if channel == "telegram":
                delivered |= self._send_telegram(f"{subject}\n\n{text}")
            elif channel == "discord":
                delivered |= self._send_discord(f"**{subject}**\n```\n{text}\n```")
            elif channel == "email":
                delivered |= self._send_email(subject, text)
            else:
                log.warning("Unknown alert channel %r; skipping", channel)
        if delivered:
            self._last_sent = time.time()
            self.sent_count += 1
        return delivered

    def send_signal(self, signal: Signal) -> bool:
        subject = (
            f"[{signal.confidence.upper()}] {signal.market} "
            f"+{signal.lookback_return:.1%}/{signal.lookback_minutes}m - {signal.strategy}"
        )
        return self.send(subject, signal.to_text(), kind=ALERT_SIGNAL)

    def send_informational(self, market: str, detail: str) -> bool:
        subject = f"[INFO] momentum event detected on {market}, no valid entry"
        body = (
            f"{format_display(now_utc())}\n{detail}\n\n"
            "This is informational only: the event was detected but at least one "
            "validated filter did not pass, so no entry is suggested."
        )
        return self.send(subject, body, kind=ALERT_INFO)

    def send_system(self, message: str) -> bool:
        return self.send("[SYSTEM] bitvavo-momentum-agent", message, kind=ALERT_SYSTEM)
