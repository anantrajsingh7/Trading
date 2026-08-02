"""bitvavo-momentum-agent.

Research, backtesting and paper-trading system for short-horizon momentum
events on Bitvavo EUR markets.

Design rules enforced throughout this package:

* All internal timestamps are timezone-aware UTC. Europe/Amsterdam is used for
  display only (dashboard, reports).
* No function that generates an entry signal may read a bar that closes at or
  after the decision timestamp.
* Live order execution is disabled unless three independent switches are set
  (see ``config/paper_trading.yaml``).
* Credentials are read from the environment only.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
