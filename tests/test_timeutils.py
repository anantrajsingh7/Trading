from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from bitvavo_momentum.timeutils import (
    DISPLAY_TZ,
    expected_index,
    floor_to_interval,
    interval_to_minutes,
    to_display,
    to_epoch_ms,
    to_utc,
)


def test_integer_input_is_interpreted_as_milliseconds():
    # Bitvavo returns epoch milliseconds; treating them as seconds would put every
    # candle in 1970 and would still "work" silently.
    assert to_utc(1_700_000_000_000) == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")


def test_naive_datetime_is_assumed_utc_not_local():
    naive = datetime(2024, 6, 1, 12, 0, 0)
    assert to_utc(naive).hour == 12
    assert str(to_utc(naive).tz) == "UTC"


def test_roundtrip_epoch_ms():
    ts = pd.Timestamp("2024-03-15 08:45:00", tz="UTC")
    assert to_utc(to_epoch_ms(ts)) == ts


def test_display_conversion_handles_dst():
    # Amsterdam is UTC+1 in January and UTC+2 in July. Doing arithmetic in local
    # time would silently shift results across the DST boundary.
    winter = to_display("2024-01-15T12:00:00Z")
    summer = to_display("2024-07-15T12:00:00Z")
    assert winter.hour == 13
    assert summer.hour == 14
    assert str(winter.tz) == str(DISPLAY_TZ)


def test_expected_index_is_complete_and_utc():
    index = expected_index("2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z", "1m")
    assert len(index) == 11
    assert str(index.tz) == "UTC"


def test_floor_to_interval():
    assert floor_to_interval("2024-01-01T00:07:33Z", "5m") == pd.Timestamp("2024-01-01 00:05:00", tz="UTC")


def test_unknown_interval_raises():
    with pytest.raises(ValueError):
        interval_to_minutes("3m")
