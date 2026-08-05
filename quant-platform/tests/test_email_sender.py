"""
Regression tests for email delivery.

The bug these guard against: _send_gmail resolved smtp.gmail.com to an IP
and connected to that IP, so TLS verification failed with
"certificate is not valid for <ip>". SMTP must always be given the
HOSTNAME so the certificate matches.
"""
from unittest.mock import MagicMock, patch

from reporting import email_sender


def _capture_smtp_host(mock_smtp):
    """First positional arg of the SMTP constructor call."""
    assert mock_smtp.called, "SMTP was never constructed"
    return mock_smtp.call_args[0][0]


class TestGmailSMTP:
    @patch("smtplib.SMTP")
    def test_connects_by_hostname_not_ip(self, mock_smtp):
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        ok = email_sender._send_gmail(
            "me@gmail.com", "abcd efgh ijkl mnop", "me@gmail.com",
            "subject", "<html>body</html>")
        host = _capture_smtp_host(mock_smtp)
        assert host == "smtp.gmail.com", f"must connect by hostname, got {host!r}"
        assert not host.replace(".", "").isdigit(), "must not connect to a raw IP"
        assert ok

    @patch("smtplib.SMTP")
    def test_starttls_uses_verifying_context(self, mock_smtp):
        smtp_obj = MagicMock()
        mock_smtp.return_value.__enter__.return_value = smtp_obj
        email_sender._send_gmail("me@gmail.com", "pw", "me@gmail.com", "s", "<p>x</p>")
        assert smtp_obj.starttls.called
        ctx = smtp_obj.starttls.call_args.kwargs.get("context")
        assert ctx is not None and ctx.check_hostname, "hostname checking must stay on"

    @patch("smtplib.SMTP_SSL")
    @patch("smtplib.SMTP")
    def test_falls_back_to_port_465(self, mock_smtp, mock_ssl):
        mock_smtp.side_effect = OSError("587 blocked")
        mock_ssl.return_value.__enter__.return_value = MagicMock()
        ok = email_sender._send_gmail("me@gmail.com", "pw", "me@gmail.com", "s", "<p>x</p>")
        assert ok, "should succeed via the 465 fallback"
        assert mock_ssl.call_args[0][0] == "smtp.gmail.com"
        assert mock_ssl.call_args[0][1] == 465

    @patch("smtplib.SMTP")
    def test_auth_error_does_not_retry(self, mock_smtp):
        import smtplib
        mock_smtp.side_effect = smtplib.SMTPAuthenticationError(535, b"bad creds")
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            ok = email_sender._send_gmail("me@gmail.com", "pw", "me@gmail.com", "s", "<p>x</p>")
            assert not ok
            assert not mock_ssl.called, "bad credentials must not retry on 465"
