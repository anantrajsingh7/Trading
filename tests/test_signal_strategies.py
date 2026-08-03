"""Signal-strategy tests, causality first."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitvavo_momentum.signal_strategies import (
    CompressionBreakout,
    DonchianBreakout,
    TrendFollowing,
    compute_setup_features,
    default_signal_strategies,
    exhaustion_veto,
    generate_signals,
)


def _series(n: int = 1200, seed: int = 7, drift: float = 0.0002) -> pd.DataFrame:
    """Random walk with volatility clustering and volume that tracks it.

    Volume must correlate with absolute return, as it does in real markets. A
    constant-volume fixture silently disables every relative-volume filter, so
    the compression and breakout strategies would appear to work while never
    actually being exercised - which is exactly what an earlier version of this
    fixture did.
    """
    rng = np.random.default_rng(seed)

    # Volatility clustering: quiet stretches alternate with active ones, so the
    # compression percentile has something real to detect.
    vol_state = np.abs(rng.normal(1.0, 0.5, n))
    for i in range(1, n):
        vol_state[i] = 0.95 * vol_state[i - 1] + 0.05 * vol_state[i]
    steps = rng.normal(drift, 0.006, n) * vol_state

    close = 100.0 * np.exp(np.cumsum(steps))
    index = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="15min", tz="UTC")
    noise = np.abs(rng.normal(0, 0.002, n)) * close
    previous = np.concatenate([[100.0], close[:-1]])

    # Volume expands with the size of the move, as real order flow does.
    move = np.abs(steps) / (np.abs(steps).mean() + 1e-12)
    volume = np.abs(rng.normal(1000, 200, n)) * (0.5 + move)

    return pd.DataFrame(
        {
            "timestamp": index,
            "open": previous,
            "high": np.maximum(previous, close) + noise,
            "low": np.minimum(previous, close) - noise,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture(scope="module")
def features():
    return compute_setup_features(_series(), "15m")


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #
def test_features_are_unchanged_when_future_data_is_removed():
    frame = _series()
    full = compute_setup_features(frame, "15m")
    cut = len(frame) // 2
    truncated = compute_setup_features(frame.iloc[:cut].copy(), "15m")

    common = truncated.index
    numeric = [c for c in full.columns if pd.api.types.is_float_dtype(full[c])]
    mismatched = []
    for column in numeric:
        a = full.loc[common, column].to_numpy(dtype="float64")
        b = truncated.loc[common, column].to_numpy(dtype="float64")
        both_nan = np.isnan(a) & np.isnan(b)
        if not np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True):
            mismatched.append(column)
    assert not mismatched, f"these features changed when future data was removed: {mismatched}"


def test_percentiles_are_expanding_not_full_sample(features):
    """A full-sample percentile would be defined from the very first bar."""
    pct = features["bb_width_pctile"]
    assert pct.iloc[:100].isna().all(), "percentile must warm up, not know the future distribution"
    valid = pct.dropna()
    assert ((valid >= 0) & (valid <= 1)).all()


@pytest.mark.parametrize("strategy", default_signal_strategies())
def test_signals_are_stable_under_truncation(strategy):
    frame = _series()
    full_features = compute_setup_features(frame, "15m")
    full = strategy.generate(full_features, "T-EUR")
    if full.empty:
        pytest.skip(f"{strategy.name} produced no signals on the fixture")

    cut_time = full["event_time"].iloc[len(full) // 2]
    truncated = strategy.generate(full_features[full_features.index <= cut_time], "T-EUR")

    earlier = set(full[full["event_time"] <= cut_time]["event_time"])
    assert earlier == set(truncated["event_time"]), (
        f"{strategy.name}: signal set changed when future bars were removed"
    )


def test_donchian_channel_excludes_the_current_bar(features):
    """A breakout compared against a channel including today is trivially true."""
    channel = features["donchian_high_24"]
    highs = features["high"]
    # The channel at bar i must equal the max of highs[i-24:i], not including i.
    i = 500
    expected = highs.iloc[i - 24 : i].max()
    assert channel.iloc[i] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# behaviour
# --------------------------------------------------------------------------- #
def test_trend_fires_far_less_often_in_a_downtrend():
    """Trend following must be strongly biased toward rising markets.

    Not *zero* in a downtrend: counter-trend bounces do briefly lift EMA20 above
    EMA50, and a rule that pretended otherwise would be lying about what it does
    live. That residual is exactly what the regime filter exists to remove.
    """
    strategy = TrendFollowing(name="t", family="trend")
    rising = strategy.generate(compute_setup_features(_series(drift=0.001, seed=11), "15m"), "T-EUR")
    falling = strategy.generate(compute_setup_features(_series(drift=-0.001, seed=11), "15m"), "T-EUR")
    assert len(falling) < len(rising), "a downtrend must produce fewer trend signals than an uptrend"
    assert len(falling) / 1200 < 0.02, "and they must remain rare in absolute terms"


def test_compression_requires_prior_quiet(features):
    strict = CompressionBreakout(name="c", family="compression", compression_max_pctile=0.05)
    loose = CompressionBreakout(name="c", family="compression", compression_max_pctile=0.90)
    assert len(strict.generate(features, "T-EUR")) <= len(loose.generate(features, "T-EUR"))


def test_longer_donchian_window_produces_fewer_signals(features):
    short = DonchianBreakout(name="d12", family="donchian", channel_window=12).generate(features, "T-EUR")
    long = DonchianBreakout(name="d96", family="donchian", channel_window=96).generate(features, "T-EUR")
    assert len(long) <= len(short)


def test_cooldown_collapses_a_run_of_bars(features):
    fast = DonchianBreakout(name="d", family="donchian", channel_window=12, cooldown_bars=1)
    slow = DonchianBreakout(name="d", family="donchian", channel_window=12, cooldown_bars=48)
    assert len(slow.generate(features, "T-EUR")) <= len(fast.generate(features, "T-EUR"))


def test_signal_table_matches_the_backtester_schema(features):
    signals = DonchianBreakout(name="d", family="donchian", channel_window=12).generate(features, "T-EUR")
    if signals.empty:
        pytest.skip("no signals")
    for column in ("market", "event_time", "event_spec", "close", "atr_60m",
                   "realised_vol_60m", "volume", "session_bucket"):
        assert column in signals.columns, f"backtester requires {column}"


def test_signals_can_be_backtested_end_to_end(features):
    """The whole point of the schema: reuse the existing engine unchanged."""
    from bitvavo_momentum.backtester import Backtester
    from bitvavo_momentum.execution_model import ExecutionModel, ExecutionScenario
    from bitvavo_momentum.risk_manager import RiskLimits, SizingConfig
    from bitvavo_momentum.strategies import ExitPolicy, ImmediateEntry

    signals = DonchianBreakout(name="d", family="donchian", channel_window=12).generate(features, "T-EUR")
    if signals.empty:
        pytest.skip("no signals")

    engine = Backtester(
        ExecutionModel(ExecutionScenario(name="t"), seed=1),
        ExitPolicy(take_profit_pct=0.05, stop_loss_pct=0.03, time_stop_minutes=48 * 60),
        SizingConfig(method="fixed_eur", fixed_eur_amount=500.0, max_position_pct_of_equity=1.0,
                     max_participation_of_recent_volume=1.0),
        RiskLimits(max_concurrent_positions=3),
        starting_equity=10_000.0, interval="15m", max_holding_minutes=48 * 60,
        min_order_quote_eur=1.0,
    )
    result = engine.run(signals, {"T-EUR": features}, ImmediateEntry())
    assert len(result.trades) == len(signals), "every signal must produce a row"


# --------------------------------------------------------------------------- #
# exhaustion veto
# --------------------------------------------------------------------------- #
def test_exhaustion_veto_rejects_a_vertical_move():
    frame = _series(n=400, seed=3)
    # Force a +25% single-bar spike near the end.
    frame.loc[350, ["open", "high", "low", "close"]] = [100.0, 130.0, 99.0, 128.0]
    frame.loc[351:, ["open", "high", "low", "close"]] *= 1.28
    features = compute_setup_features(frame, "15m")
    veto = exhaustion_veto(features)
    assert bool(veto.iloc[350]), "a vertical single-bar move must be vetoed"


def test_exhaustion_veto_leaves_calm_bars_alone(features):
    veto = exhaustion_veto(features)
    assert veto.mean() < 0.5, "the veto must not reject most of a normal series"


def test_generate_signals_across_markets(features):
    universe = {"A-EUR": features, "B-EUR": features}
    out = generate_signals(default_signal_strategies()[:3], universe)
    if out.empty:
        pytest.skip("no signals on the fixture")
    assert set(out["market"]) <= {"A-EUR", "B-EUR"}
    assert out["event_time"].is_monotonic_increasing


def test_veto_reduces_signal_count(features):
    universe = {"A-EUR": features}
    strategies = default_signal_strategies()[:5]
    without = generate_signals(strategies, universe, apply_exhaustion_veto=False)
    with_veto = generate_signals(strategies, universe, apply_exhaustion_veto=True)
    assert len(with_veto) <= len(without)
