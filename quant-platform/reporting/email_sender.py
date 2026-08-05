"""
Email delivery for daily dashboard reports.

Two backends — whichever credentials are present in .env is used:

1. SendGrid HTTP API (works everywhere, including remote containers)
   Requires: SENDGRID_API_KEY, ALERT_EMAIL_FROM, ALERT_EMAIL_TO

2. Gmail SMTP (works when running locally, SMTP not blocked)
   Requires: GMAIL_SENDER, GMAIL_APP_PASSWORD, ALERT_EMAIL_TO

SendGrid is preferred when both are configured.
"""
from __future__ import annotations

import json
import os
import smtplib
import socket
import ssl
import urllib.request
import urllib.error
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from core.logging import get_logger

log = get_logger("reporting.email")


def send_dashboard(html_path: Path, subject: str | None = None) -> bool:
    """
    Send the HTML dashboard as email.
    Tries SendGrid first, falls back to Gmail SMTP.
    Returns True on success, False on failure.
    """
    if not html_path.exists():
        log.error("Dashboard file not found: %s", html_path)
        return False

    today = date.today()
    subject = subject or f"Quant Platform Dashboard — {today.strftime('%A, %B %d, %Y')}"
    html_body = html_path.read_text(encoding="utf-8")
    recipient = os.getenv("ALERT_EMAIL_TO", "anantrajsingh7@gmail.com").strip()

    # Try SendGrid first (works in remote containers)
    sg_key = os.getenv("SENDGRID_API_KEY", "").strip()
    if sg_key:
        return _send_sendgrid(sg_key, subject, html_body, recipient)

    # Fall back to Gmail SMTP (works locally)
    sender   = os.getenv("GMAIL_SENDER", "").strip()
    password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    if sender and password:
        return _send_gmail(sender, password, recipient, subject, html_body)

    log.warning(
        "No email credentials found. Add either:\n"
        "  SENDGRID_API_KEY (recommended for remote use)\n"
        "  GMAIL_SENDER + GMAIL_APP_PASSWORD (for local use)"
    )
    return False


def _send_sendgrid(api_key: str, subject: str, html: str, recipient: str) -> bool:
    sender = os.getenv("ALERT_EMAIL_FROM", "anantrajsingh7@gmail.com").strip()
    payload = json.dumps({
        "personalizations": [{"to": [{"email": recipient}]}],
        "from": {"email": sender, "name": "Quant Platform"},
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": "Open in HTML email client to view dashboard."},
            {"type": "text/html",  "value": html},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status in (200, 202):
                log.info("Dashboard emailed via SendGrid to %s", recipient)
                return True
            log.error("SendGrid returned HTTP %s", resp.status)
            return False
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        log.error("SendGrid error %s: %s", e.code, body)
        return False
    except Exception as e:
        log.error("SendGrid send failed: %s", e)
        return False


def _send_gmail(sender: str, password: str, recipient: str,
                subject: str, html: str) -> bool:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Quant Platform <{sender}>"
    msg["To"]      = recipient
    msg.attach(MIMEText("Open in HTML email client to view dashboard.", "plain"))
    msg.attach(MIMEText(html, "html"))

    # Always connect by HOSTNAME. Connecting to a resolved IP breaks TLS
    # verification: Gmail's certificate is issued for smtp.gmail.com, so an
    # IP target raises CERTIFICATE_VERIFY_FAILED (IP address mismatch).
    ctx = ssl.create_default_context()

    # Try STARTTLS on 587 first, then implicit TLS on 465 as a fallback
    # (some networks block one or the other).
    attempts = (
        ("starttls", "smtp.gmail.com", 587),
        ("ssl",      "smtp.gmail.com", 465),
    )
    last_err = None
    for mode, host, port in attempts:
        try:
            if mode == "starttls":
                with smtplib.SMTP(host, port, timeout=30) as smtp:
                    smtp.ehlo()
                    smtp.starttls(context=ctx)
                    smtp.login(sender, password)
                    smtp.sendmail(sender, recipient, msg.as_string())
            else:
                with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as smtp:
                    smtp.login(sender, password)
                    smtp.sendmail(sender, recipient, msg.as_string())
            log.info("Dashboard emailed via Gmail SMTP (%s:%d) to %s", host, port, recipient)
            return True
        except smtplib.SMTPAuthenticationError:
            log.error(
                "Gmail auth rejected — GMAIL_APP_PASSWORD must be a 16-character "
                "App Password (not your normal Gmail password), and GMAIL_SENDER "
                "must be the account that generated it."
            )
            return False        # auth won't improve on the next port
        except Exception as e:
            last_err = e
            log.warning("Gmail SMTP %s:%d failed: %s", host, port, e)

    log.error("Gmail SMTP failed on all endpoints: %s", last_err)
    return False


def test_connection() -> bool:
    """Send a short test email using whichever backend is configured."""
    recipient = os.getenv("ALERT_EMAIL_TO", "anantrajsingh7@gmail.com").strip()
    sg_key    = os.getenv("SENDGRID_API_KEY", "").strip()
    sender    = os.getenv("GMAIL_SENDER", "").strip()
    password  = os.getenv("GMAIL_APP_PASSWORD", "").strip()

    subject = "Quant Platform — Test Email"
    html    = "<h2>Quant Platform</h2><p>Email delivery is working correctly.</p>"

    if sg_key:
        print(f"Testing SendGrid → {recipient}")
        ok = _send_sendgrid(sg_key, subject, html, recipient)
    elif sender and password:
        print(f"Testing Gmail SMTP → {recipient}")
        ok = _send_gmail(sender, password, recipient, subject, html)
    else:
        print("ERROR: No email credentials in .env")
        return False

    print("SUCCESS" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    test_connection()
