"""
Strategy registry.

Centralised place to register and retrieve all strategies.
Add a new strategy here; the scanner picks it up automatically.
"""
from __future__ import annotations

from typing import Optional

from strategies.base           import BaseStrategy
from strategies.minervini      import MinerviniStrategy
from strategies.breakout       import BreakoutStrategy
from strategies.pullback       import EMApullbackStrategy
from strategies.momentum_strat import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.enhanced_ema   import EnhancedEMAPullbackStrategy

_REGISTRY: dict[str, BaseStrategy] = {}


def _register(s: BaseStrategy) -> None:
    _REGISTRY[s.name] = s


# Register all strategies at import time
_register(MinerviniStrategy())
_register(BreakoutStrategy())
_register(EMApullbackStrategy())
_register(MomentumStrategy())
_register(MeanReversionStrategy())
_register(EnhancedEMAPullbackStrategy())   # research only — not in production scan


def get_strategy(name: str) -> BaseStrategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def all_strategies() -> list[BaseStrategy]:
    return list(_REGISTRY.values())


def strategy_names() -> list[str]:
    return list(_REGISTRY.keys())
