"""Time handling.

One rule: **everything internal is timezone-aware UTC**. Europe/Amsterdam only
appears at the display boundary (dashboard, human-readable reports). Amsterdam
observes DST, so any arithmetic done in local time would be wrong twice a year;
converting only at the very end avoids that entirely.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

DISPLAY_TZ = ZoneInfo("Europe/Amsterdam")

# Interval label -> minutes. Mirrors the intervals Bitvavo exposes on /v2/candles.
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


def interval_to_minutes(interval: str) -> int:
    """Return the number of minutes in a Bitvavo interval label."""
    try:
        return INTERVAL_MINUTES[interval]
    except KeyError as exc:  # pragma: no cover - guard
        raise ValueError(f"Unsupported interval {interval!r}; known: {sorted(INTERVAL_MINUTES)}") from exc


def interval_to_timedelta(interval: str) -> timedelta:
    return timedelta(minutes=interval_to_minutes(interval))


def now_utc() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def to_utc(value: datetime | str | int | float | pd.Timestamp) -> pd.Timestamp:
    """Coerce a value to a timezone-aware UTC ``pd.Timestamp``.

    Integers/floats are interpreted as **milliseconds** since the epoch, which is
    what the Bitvavo REST API returns. Naive datetimes are assumed to be UTC
    rather than local time - assuming local time is a classic source of silent
    one-hour errors in Amsterdam.
    """
    if isinstance(value, int | float):
        return pd.Timestamp(int(value), unit="ms", tz="UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_epoch_ms(value: datetime | str | pd.Timestamp) -> int:
    """Milliseconds since epoch, as required by Bitvavo's ``start``/``end`` args."""
    ts = to_utc(value)
    return int(ts.value // 1_000_000)


def to_display(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    """Convert a UTC instant to Europe/Amsterdam for display purposes only."""
    return to_utc(value).tz_convert(DISPLAY_TZ)


def format_display(value: datetime | str | pd.Timestamp, with_tz: bool = True) -> str:
    ts = to_display(value)
    fmt = "%Y-%m-%d %H:%M:%S %Z" if with_tz else "%Y-%m-%d %H:%M:%S"
    return ts.strftime(fmt)


def floor_to_interval(value: datetime | str | pd.Timestamp, interval: str) -> pd.Timestamp:
    """Floor a timestamp to the start of its interval bucket (UTC)."""
    return to_utc(value).floor(f"{interval_to_minutes(interval)}min")


def utc_index(values) -> pd.DatetimeIndex:
    """Build a UTC ``DatetimeIndex`` from anything pandas understands."""
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx


def expected_index(
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    interval: str,
) -> pd.DatetimeIndex:
    """The complete set of bar-open timestamps expected between start and end."""
    minutes = interval_to_minutes(interval)
    return pd.date_range(
        start=floor_to_interval(start, interval),
        end=floor_to_interval(end, interval),
        freq=f"{minutes}min",
        tz="UTC",
    )


def session_bucket(ts: pd.Timestamp) -> str:
    """Coarse UTC time-of-day bucket used in the time-of-day breakdown."""
    hour = to_utc(ts).hour
    if 0 <= hour < 6:
        return "00-06_UTC"
    if 6 <= hour < 12:
        return "06-12_UTC"
    if 12 <= hour < 18:
        return "12-18_UTC"
    return "18-24_UTC"
