"""Orchestration shared by the scripts and the dashboard.

Keeps data loading, feature construction and event building in one place so that
``run_research.py``, ``run_backtest.py`` and the dashboard cannot drift apart and
report subtly different numbers from the same inputs.

Data sources
------------
``real``
    Parquet written by ``scripts/download_history.py`` from the Bitvavo API.

``synthetic``
    Artificial series from :mod:`synthetic`. Every artefact produced from this
    source is written under a ``synthetic/`` prefix and carries a banner. It
    exists to prove the pipeline runs, never to support a claim about markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import Config
from .data_validator import reindex_to_grid, validate_candles
from .event_detector import EventSpec, add_forward_returns, build_event_dataset
from .features import attach_context, build_features, build_market_context
from .logging_utils import get_logger
from .market_universe import eligibility_mask, research_universe
from .regimes import RegimeConfig, attach_regimes, classify_regimes
from .storage import ParquetStore, ResultStore
from .synthetic import SyntheticConfig, generate_universe

log = get_logger(__name__)


@dataclass
class Dataset:
    """Everything the research and backtest stages need."""

    candles: dict[str, pd.DataFrame] = field(default_factory=dict)
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    eligibility: dict[str, pd.Series] = field(default_factory=dict)
    context: pd.DataFrame = field(default_factory=pd.DataFrame)
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "none"
    is_synthetic: bool = False

    @property
    def markets(self) -> list[str]:
        return sorted(self.candles)

    @property
    def first_timestamp(self) -> pd.Timestamp | None:
        stamps = [f["timestamp"].min() for f in self.candles.values() if not f.empty]
        return min(stamps) if stamps else None

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        stamps = [f["timestamp"].max() for f in self.candles.values() if not f.empty]
        return max(stamps) if stamps else None

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "is_synthetic": self.is_synthetic,
            "n_markets": len(self.candles),
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "total_bars": int(sum(len(f) for f in self.candles.values())),
        }


def load_real_candles(
    config: Config,
    interval: str = "1m",
    markets: list[str] | None = None,
    max_markets: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load validated Parquet candles written by the downloader."""
    store = ParquetStore(config.path("processed_dir"))
    exclude = config.get("research", "universe", "exclude_bases", default=[])
    available = markets or research_universe(store, interval=interval, exclude_bases=exclude)
    references = config.get("research", "data", "reference_markets", default=["BTC-EUR", "ETH-EUR"])
    for reference in references:
        if reference not in available and store.candle_path(reference, interval).exists():
            available.append(reference)
    if max_markets is not None:
        # Keep references plus the most-covered markets, so a smoke run is
        # representative rather than alphabetical.
        manifest = store.read_manifest()
        if not manifest.empty:
            ranked = (
                manifest[manifest["interval"] == interval]
                .sort_values("n_rows", ascending=False)["market"]
                .tolist()
            )
            available = [m for m in ranked if m in available][:max_markets]
        else:
            available = available[:max_markets]
        for reference in references:
            if reference not in available and store.candle_path(reference, interval).exists():
                available.append(reference)

    out: dict[str, pd.DataFrame] = {}
    for market in available:
        frame = store.read_candles(market, interval)
        if not frame.empty:
            out[market] = frame
    log.info("Loaded %d markets from %s", len(out), store.candles_dir)
    return out


def build_dataset(
    config: Config,
    source: str = "real",
    interval: str = "1m",
    markets: list[str] | None = None,
    max_markets: int | None = None,
    synthetic_config: SyntheticConfig | None = None,
    n_synthetic_markets: int = 8,
) -> Dataset:
    """Load candles, validate, build features, context, eligibility and regimes."""
    if source == "synthetic":
        candles = generate_universe(n_synthetic_markets, synthetic_config)
        is_synthetic = True
        source_label = "SYNTHETIC (bitvavo_momentum.synthetic)"
    else:
        candles = load_real_candles(config, interval, markets, max_markets)
        is_synthetic = False
        source_label = "Bitvavo public candles (data/processed)"

    dataset = Dataset(candles=candles, source=source_label, is_synthetic=is_synthetic)
    if not candles:
        log.warning("No candle data available for source=%s", source)
        return dataset

    # -- validation ---------------------------------------------------------
    validations = []
    usable: dict[str, pd.DataFrame] = {}
    for market, frame in candles.items():
        result = validate_candles(frame, market, interval, min_rows=200)
        validations.append(result.to_dict())
        if result.is_usable:
            usable[market] = frame
        else:
            log.warning("Excluding %s: %s", market, "; ".join(result.notes))
    dataset.validation = pd.DataFrame(validations)
    dataset.candles = usable

    # -- features -----------------------------------------------------------
    for market, frame in usable.items():
        gridded = reindex_to_grid(frame, interval)
        features = build_features(gridded, interval)
        if not features.empty:
            dataset.features[market] = features

    dataset.context = build_market_context(usable, interval)
    if not dataset.context.empty:
        dataset.features = {
            market: attach_context(features, dataset.context)
            for market, features in dataset.features.items()
        }

    # -- eligibility (liquidity/age, evaluated causally) --------------------
    min_volume = float(config.get("research", "universe", "min_median_daily_quote_volume_eur", default=25000))
    min_days = int(config.get("research", "universe", "min_history_days", default=30))
    for market, frame in usable.items():
        dataset.eligibility[market] = eligibility_mask(frame, min_volume, min_days)

    # -- regimes ------------------------------------------------------------
    btc = usable.get("BTC-EUR")
    if btc is not None and not btc.empty:
        from .data_downloader import resample_candles

        daily = resample_candles(btc, "1d")
        breadth = None
        if not dataset.context.empty and "breadth_positive_1440m" in dataset.context.columns:
            breadth = dataset.context["breadth_positive_1440m"]
        dataset.regimes = classify_regimes(daily, RegimeConfig.from_config(config.research), breadth)
    else:
        log.warning("BTC-EUR missing - regime labels will be unavailable")

    log.info("Dataset ready: %s", dataset.summary())
    return dataset


def build_events(
    dataset: Dataset,
    specs: list[EventSpec],
    config: Config,
    interval: str = "1m",
    with_forward_returns: bool = True,
    exclude_reference_markets: bool = True,
) -> pd.DataFrame:
    """Detect events, attach forward returns (study only) and regime labels."""
    features = dataset.features
    if exclude_reference_markets:
        references = set(config.get("research", "data", "reference_markets", default=[]))
        features = {m: f for m, f in features.items() if m not in references}

    events = build_event_dataset(
        features,
        specs,
        eligibility_by_market=dataset.eligibility,
        interval=interval,
        max_missing_fraction_lookback=float(
            config.get("research", "events", "max_missing_candle_fraction_lookback", default=0.20)
        ),
        require_nonzero_volume_candles=int(
            config.get("research", "events", "require_nonzero_volume_candles", default=3)
        ),
    )
    if events.empty:
        log.warning("No events detected for %d specs across %d markets", len(specs), len(features))
        return events

    if with_forward_returns:
        horizons = tuple(config.get("research", "events", "forward_horizons", default=[15, 30, 60, 120, 240, 480, 1440]))
        events = add_forward_returns(events, dataset.candles, horizons, interval)
    if not dataset.regimes.empty:
        events = attach_regimes(events, dataset.regimes, time_column="event_time")

    log.info("Built %d events across %d markets", len(events), events["market"].nunique())
    return events


def result_store_for(config: Config, dataset: Dataset) -> ResultStore:
    """Synthetic runs are written to a separate subtree, never mixed with real ones."""
    root = config.path("results_dir")
    return ResultStore(root / "synthetic" if dataset.is_synthetic else root)
