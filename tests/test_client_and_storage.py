from __future__ import annotations

import gzip
import json
import time

import pandas as pd
import pytest
import requests

from bitvavo_momentum.bitvavo_client import (
    BitvavoAPIError,
    BitvavoClient,
    BitvavoNetworkError,
    BitvavoRateLimitError,
)
from bitvavo_momentum.config import Credentials
from bitvavo_momentum.data_downloader import candles_to_frame, resample_candles
from bitvavo_momentum.market_universe import MarketRules
from bitvavo_momentum.storage import DatasetRecord, ParquetStore, RawStore
from bitvavo_momentum.walk_forward import chronological_splits, walk_forward_windows


class _Response:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else []
        self.headers = headers or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Scripted transport so retry/rate-limit behaviour can be asserted offline."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, headers=None, data=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers or {}})
        item = self.responses.pop(0) if self.responses else _Response()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def _client(session, **kwargs) -> BitvavoClient:
    return BitvavoClient(Credentials(), session=session, backoff_base=1.0, backoff_cap=0.01, **kwargs)


def test_signature_is_deterministic_and_matches_the_documented_scheme():
    credentials = Credentials(api_key="key", api_secret="secret")
    client = BitvavoClient(credentials, session=_FakeSession([]))
    signature = client._sign(1700000000000, "GET", "/account", "")
    assert signature == client._sign(1700000000000, "GET", "/account", "")
    assert len(signature) == 64  # hex sha256

    import hashlib
    import hmac

    expected = hmac.new(b"secret", b"1700000000000GET/v2/account", hashlib.sha256).hexdigest()
    assert signature == expected


def test_authenticated_endpoint_requires_credentials():
    client = _client(_FakeSession([]))
    with pytest.raises(Exception, match="requires credentials"):
        client.get_account()


def test_retries_on_server_error_then_succeeds():
    session = _FakeSession([_Response(503, text="boom"), _Response(200, {"time": 1})])
    client = _client(session)
    assert client.get_time() == 1
    assert len(session.calls) == 2
    assert client.retry_count == 1


def test_gives_up_after_max_retries():
    session = _FakeSession([requests.ConnectionError("down")] * 3)
    client = _client(session, max_retries=3)
    with pytest.raises(BitvavoNetworkError):
        client.get_time()
    assert len(session.calls) == 3


def test_client_error_is_not_retried():
    session = _FakeSession([_Response(400, {"errorCode": 203, "error": "bad market"})])
    client = _client(session)
    with pytest.raises(BitvavoAPIError):
        client.get_candles("NOPE-EUR")
    assert len(session.calls) == 1, "4xx must not burn the rate-limit budget on retries"


def test_rate_limit_headers_update_the_local_budget():
    session = _FakeSession([
        _Response(200, {"time": 1}, headers={
            "bitvavo-ratelimit-remaining": "42",
            "bitvavo-ratelimit-resetat": str(int(time.time() * 1000) + 1000),
        })
    ])
    client = _client(session)
    client.get_time()
    assert client.rate_limit.remaining <= 42


def test_error_code_105_is_a_rate_limit_error():
    session = _FakeSession([_Response(200, {"errorCode": 105, "error": "rate limited"})])
    with pytest.raises(BitvavoRateLimitError):
        _client(session).get_time()


def test_raw_sink_receives_the_unmodified_payload(tmp_path):
    payload = [[1700000000000, "1", "2", "0.5", "1.5", "10"]]
    store = RawStore(tmp_path)
    session = _FakeSession([_Response(200, payload)])
    client = _client(session, raw_sink=store.as_sink())
    client.get_candles("BTC-EUR", "1m")

    files = list(tmp_path.rglob("*.json.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rb") as handle:
        captured = json.loads(handle.read().decode("utf-8"))
    assert captured["payload"] == payload
    assert captured["endpoint"] == "/BTC-EUR/candles"


def test_raw_sink_failure_does_not_break_the_download():
    def _explode(path, params, payload):
        raise RuntimeError("disk full")

    session = _FakeSession([_Response(200, {"time": 1})])
    client = _client(session, raw_sink=_explode)
    assert client.get_time() == 1  # capture failure is logged, not fatal


# --------------------------------------------------------------------------- #
# parsing / storage
# --------------------------------------------------------------------------- #
def test_candles_to_frame_sorts_and_deduplicates():
    rows = [
        [1700000120000, "3", "3", "3", "3", "3"],
        [1700000000000, "1", "1", "1", "1", "1"],
        [1700000060000, "2", "2", "2", "2", "2"],
        [1700000060000, "2", "2", "2", "2", "9"],  # duplicate, later value wins
    ]
    frame = candles_to_frame(rows)
    assert len(frame) == 3
    assert frame["timestamp"].is_monotonic_increasing
    assert frame.loc[frame["timestamp"] == pd.Timestamp(1700000060000, unit="ms", tz="UTC"), "volume"].iloc[0] == 9.0


def test_candles_to_frame_handles_empty():
    assert candles_to_frame([]).empty


def test_resample_aggregates_consistently():
    index = pd.date_range("2024-01-01", periods=60, freq="1min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": index,
        "open": range(60), "high": range(1, 61), "low": range(60), "close": range(60),
        "volume": [1.0] * 60,
    })
    resampled = resample_candles(frame, "15m")
    assert len(resampled) == 4
    assert resampled["volume"].iloc[0] == 15.0
    assert resampled["open"].iloc[0] == 0
    assert resampled["close"].iloc[0] == 14


def test_parquet_roundtrip_and_manifest(tmp_path):
    store = ParquetStore(tmp_path)
    index = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": index, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    store.write_candles("A-EUR", "1m", frame)
    read_back = store.read_candles("A-EUR", "1m")
    assert len(read_back) == 100
    assert str(read_back["timestamp"].dt.tz) == "UTC"

    record = DatasetRecord(
        market="A-EUR", interval="1m", first_timestamp=index[0], last_timestamp=index[-1],
        n_rows=100, n_missing_intervals=0, missing_fraction=0.0, n_duplicate_rows=0,
        n_zero_volume_rows=0, zero_volume_fraction=0.0,
        download_started_utc=index[0], download_finished_utc=index[-1],
        api_source="test", validation_status="PASS",
    )
    store.upsert_manifest(record)
    manifest = store.read_manifest()
    assert len(manifest) == 1
    assert store.usable_datasets("1m") == ["A-EUR"]

    record.validation_status = "FAIL"
    store.upsert_manifest(record)
    assert store.read_manifest().shape[0] == 1, "upsert must replace, not duplicate"
    assert store.usable_datasets("1m") == []


# --------------------------------------------------------------------------- #
# market rules
# --------------------------------------------------------------------------- #
def test_market_rules_round_prices_to_significant_digits():
    rules = MarketRules(market="X-EUR", base="X", quote="EUR", status="trading", price_precision=5)
    assert rules.round_price(1.234567) == pytest.approx(1.2346)
    assert rules.round_price(12345.67) == pytest.approx(12346.0)


def test_amount_rounding_never_rounds_up():
    rules = MarketRules(market="X-EUR", base="X", quote="EUR", status="trading", quantity_decimals=4)
    assert rules.round_amount(1.99999) == pytest.approx(1.9999)


def test_minimum_order_is_enforced():
    rules = MarketRules(market="X-EUR", base="X", quote="EUR", status="trading", min_order_in_quote=5.0)
    assert not rules.meets_minimum(0.01, 100.0)   # 1 EUR
    assert rules.meets_minimum(0.10, 100.0)       # 10 EUR


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def test_chronological_splits_do_not_overlap_and_have_an_embargo():
    events = pd.DataFrame({
        "event_time": pd.date_range("2023-01-01", periods=1000, freq="6h", tz="UTC")
    })
    splits = chronological_splits(events, 0.5, 0.25, embargo_minutes=2880)
    assert splits.train.end < splits.validation.start
    assert splits.validation.end < splits.test.start
    gap = (splits.validation.start - splits.train.end).total_seconds() / 60
    assert gap == pytest.approx(2880)


def test_walk_forward_windows_roll_forward():
    events = pd.DataFrame({
        "event_time": pd.date_range("2022-01-01", periods=2000, freq="12h", tz="UTC")
    })
    windows = walk_forward_windows(events, train_months=6, test_months=3, step_months=3)
    assert len(windows) >= 2
    for a, b in zip(windows, windows[1:], strict=False):
        assert b.train_start > a.train_start
        assert a.train_end <= a.test_start
