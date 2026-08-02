from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitvavo_momentum.data_validator import reindex_to_grid  # noqa: E402
from bitvavo_momentum.features import build_features  # noqa: E402
from bitvavo_momentum.synthetic import SyntheticConfig, generate_market, generate_universe  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_config() -> SyntheticConfig:
    # Small but long enough for the 1440-bar volume baseline to warm up.
    return SyntheticConfig(n_minutes=60 * 24 * 12, seed=424242)


@pytest.fixture(scope="session")
def candles(synthetic_config):
    return generate_market("TEST-EUR", synthetic_config)


@pytest.fixture(scope="session")
def features(candles):
    return build_features(reindex_to_grid(candles, "1m"), "1m")


@pytest.fixture(scope="session")
def universe(synthetic_config):
    return generate_universe(3, synthetic_config)
