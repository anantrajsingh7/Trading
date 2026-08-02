"""Phase 12: Streamlit dashboard.

Run with::

    streamlit run dashboard/app.py
    # or: python scripts/run_dashboard.py

All internal state is UTC; every timestamp shown here is converted to
Europe/Amsterdam at the display boundary and labelled with its timezone.

The dashboard is a *viewer*. It never recomputes a strategy, never re-selects
parameters and never writes to the research artefacts - if the numbers on screen
could drift from the numbers in the report, the report would stop being the
record of truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitvavo_momentum.config import Config, load_dotenv_if_present  # noqa: E402
from bitvavo_momentum.metrics import breakdown, compute_trade_metrics  # noqa: E402
from bitvavo_momentum.storage import ParquetStore, ResultStore  # noqa: E402
from bitvavo_momentum.timeutils import DISPLAY_TZ, format_display, now_utc, to_display  # noqa: E402

st.set_page_config(page_title="Bitvavo momentum agent", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60)
def _load_config() -> tuple[dict, dict, dict]:
    load_dotenv_if_present()
    config = Config.load()
    return config.research, config.risk, config.paper


def _config() -> Config:
    load_dotenv_if_present()
    return Config.load()


@st.cache_data(ttl=30)
def _read_frame(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    except Exception as exc:  # a corrupt artefact must not blank the page
        st.warning(f"Could not read {path.name}: {exc}")
    return pd.DataFrame()


def _localise(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert UTC columns to Europe/Amsterdam for display."""
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            try:
                out[column] = pd.to_datetime(out[column], utc=True).dt.tz_convert(DISPLAY_TZ)
            except Exception:
                pass
    return out


def _results_root(config: Config, synthetic: bool) -> Path:
    root = config.path("results_dir")
    return root / "synthetic" if synthetic else root


def _metric_or_dash(value, fmt: str = "{:.2%}") -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
config = _config()
st.sidebar.title("bitvavo-momentum-agent")
st.sidebar.caption(f"Now: {format_display(now_utc())}")

use_synthetic = st.sidebar.toggle(
    "Show synthetic-data run", value=False,
    help="Synthetic runs validate the pipeline. They are not research results.",
)
results_root = _results_root(config, use_synthetic)
store = ResultStore(results_root)

if use_synthetic:
    st.sidebar.error("SYNTHETIC MODE — figures below come from artificial data.")

page = st.sidebar.radio(
    "Page",
    ["Overview", "Live scanner", "Backtest results", "Event explorer", "Trade journal", "Data & status"],
)

live_enabled = config.get("paper", "live_trading", "enabled", default=False)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mode:** `{config.get('paper', 'mode', default='paper')}`")
st.sidebar.markdown(f"**Live trading:** {'⚠️ ENABLED IN CONFIG' if live_enabled else '✅ disabled'}")
st.sidebar.caption("Times displayed in Europe/Amsterdam; all internal state is UTC.")


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
if page == "Overview":
    st.title("Overview")
    if use_synthetic:
        st.error("**SYNTHETIC DATA — NOT A RESEARCH RESULT.**")

    paper_dir = Path(config.root) / str(config.get("paper", "paper", "state_dir", default="data/results/paper"))
    state_path = paper_dir / "paper_state.json"
    journal = _read_frame(str(paper_dir / "trade_journal.parquet"))
    signals = _read_frame(str(paper_dir / "signal_log.parquet"))

    starting_equity = float(config.get("risk", "portfolio", "starting_equity_eur", default=10000.0))
    equity = starting_equity
    open_positions = 0
    if state_path.exists():
        import json

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        equity = float(payload.get("equity", starting_equity))
        open_positions = len(payload.get("positions", []))

    realised = float(journal["net_pnl_eur"].sum()) if not journal.empty else 0.0
    peak = max(starting_equity, equity)
    drawdown = equity / peak - 1.0 if peak else 0.0

    columns = st.columns(5)
    columns[0].metric("Paper portfolio", f"€{equity:,.2f}", f"{(equity / starting_equity - 1):+.2%}")
    columns[1].metric("Realised P&L", f"€{realised:,.2f}")
    columns[2].metric("Current drawdown", _metric_or_dash(drawdown))
    columns[3].metric("Open positions", open_positions)
    columns[4].metric("Signals recorded", len(signals))

    st.subheader("Strategy status")
    card_path = Path(config.root) / str(
        config.get("paper", "scanner", "approved_strategy_card", default="data/results/approved_strategy.json")
    )
    if card_path.exists():
        import json

        card = json.loads(card_path.read_text(encoding="utf-8"))
        if int(card.get("historical_sample", 0)) <= 0:
            st.warning(
                "The approved strategy card reports a historical sample of 0. Every signal it "
                "produces is flagged **insufficient_evidence** and must not be acted on."
            )
        st.json(card)
    else:
        st.info(
            "No approved strategy card. The scanner will refuse to run until one exists — "
            "by design, so that live rules can only come from a completed evaluation."
        )

    st.subheader("Recent signals")
    if signals.empty:
        st.caption("No signals recorded yet.")
    else:
        display = _localise(signals, ["timestamp_utc", "recorded_utc", "expires_utc"])
        columns_to_show = [c for c in ["timestamp_utc", "market", "lookback_return", "action",
                                       "confidence", "outcome", "reward_to_risk", "position_size_eur"]
                           if c in display.columns]
        st.dataframe(display[columns_to_show].tail(50), width="stretch")


# --------------------------------------------------------------------------- #
# Live scanner
# --------------------------------------------------------------------------- #
elif page == "Live scanner":
    st.title("Live scanner")
    st.caption(
        "Populated by `python scripts/run_scanner.py`. The dashboard displays what the scanner "
        "recorded; it does not itself call the exchange."
    )
    scanner_dir = config.path("results_dir") / "scanner"
    files = sorted(scanner_dir.glob("signals_*.csv")) if scanner_dir.exists() else []
    if not files:
        st.info("No scanner output yet. Start the scanner to populate this page.")
    else:
        chosen = st.selectbox("Scan date", [f.name for f in files], index=len(files) - 1)
        frame = _read_frame(str(scanner_dir / chosen))
        if frame.empty:
            st.caption("File is empty.")
        else:
            display = _localise(frame, ["timestamp_utc", "expires_utc"])
            preferred = ["market", "current_price", "lookback_return", "volume_ratio", "spread_bps",
                         "quote_volume_24h_eur", "strategy", "entry_zone_low", "entry_zone_high",
                         "stop_loss", "target_1", "target_2", "reward_to_risk", "action",
                         "confidence", "timestamp_utc"]
            columns_to_show = [c for c in preferred if c in display.columns]
            st.dataframe(display[columns_to_show], width="stretch")
            st.caption(
                "action: watch / valid / extended / invalid / expired. "
                "Historical statistics attached to a signal are estimates from past data, not forecasts."
            )


# --------------------------------------------------------------------------- #
# Backtest results
# --------------------------------------------------------------------------- #
elif page == "Backtest results":
    st.title("Backtest results")
    if use_synthetic:
        st.error("**SYNTHETIC DATA — these charts validate the pipeline, nothing more.**")

    trades = _read_frame(str(results_root / "trade_log.parquet"))
    comparison = _read_frame(str(results_root / "strategy_comparison.csv"))
    walk_forward = _read_frame(str(results_root / "walk_forward_results.csv"))
    stability = _read_frame(str(results_root / "parameter_stability.csv"))

    if trades.empty and comparison.empty:
        st.info("No backtest artefacts found. Run `python scripts/run_backtest.py`.")
    else:
        if not comparison.empty:
            st.subheader("Strategy comparison (training data)")
            columns_to_show = [c for c in ["strategy", "exit_policy", "n_trades", "win_rate",
                                           "net_expectancy", "profit_factor", "max_drawdown", "sharpe"]
                               if c in comparison.columns]
            st.dataframe(comparison[columns_to_show], width="stretch")

        if not trades.empty:
            closed = trades[trades["status"] == "closed"] if "status" in trades.columns else trades
            strategies = sorted(closed["strategy"].dropna().unique()) if "strategy" in closed.columns else []
            if strategies:
                chosen = st.selectbox("Strategy", strategies)
                subset = closed[closed["strategy"] == chosen]
                policies = sorted(subset["exit_policy"].dropna().unique()) if "exit_policy" in subset.columns else []
                if policies:
                    chosen_policy = st.selectbox("Exit policy", policies)
                    subset = subset[subset["exit_policy"] == chosen_policy]

                stats = compute_trade_metrics(subset)
                columns = st.columns(6)
                columns[0].metric("Trades", stats["n_trades"])
                columns[1].metric("Win rate", _metric_or_dash(stats["win_rate"]))
                columns[2].metric("Net expectancy", _metric_or_dash(stats["net_expectancy"], "{:.3%}"))
                columns[3].metric("Profit factor", _metric_or_dash(stats["profit_factor"], "{:.2f}"))
                columns[4].metric("Max drawdown", _metric_or_dash(stats["max_drawdown"]))
                columns[5].metric("Fees paid", f"€{stats['total_fees_eur']:,.2f}")

                if not subset.empty:
                    import plotly.express as px

                    ordered = subset.sort_values("exit_time").copy()
                    ordered["equity"] = 10000.0 + ordered["net_pnl_eur"].cumsum()
                    ordered["exit_time_local"] = pd.to_datetime(
                        ordered["exit_time"], utc=True
                    ).dt.tz_convert(DISPLAY_TZ)

                    st.subheader("Equity curve")
                    st.plotly_chart(
                        px.line(ordered, x="exit_time_local", y="equity",
                                labels={"exit_time_local": "exit time (Europe/Amsterdam)", "equity": "equity (EUR)"}),
                        width="stretch",
                    )

                    ordered["peak"] = ordered["equity"].cummax()
                    ordered["drawdown"] = ordered["equity"] / ordered["peak"] - 1.0
                    st.subheader("Drawdown")
                    st.plotly_chart(
                        px.area(ordered, x="exit_time_local", y="drawdown"), width="stretch"
                    )

                    st.subheader("Trade distribution")
                    st.plotly_chart(
                        px.histogram(subset, x="net_return", nbins=60), width="stretch"
                    )

                    left, right = st.columns(2)
                    with left:
                        st.subheader("By coin")
                        st.dataframe(breakdown(subset, "market")[
                            ["market", "n_trades", "win_rate", "net_expectancy", "total_net_pnl_eur"]
                        ], width="stretch")
                    with right:
                        st.subheader("By month")
                        st.dataframe(breakdown(subset, "month")[
                            ["month", "n_trades", "win_rate", "net_expectancy", "total_net_pnl_eur"]
                        ], width="stretch")

                    st.subheader("Before and after costs")
                    costs = pd.DataFrame({
                        "component": ["gross P&L", "fees", "spread cost", "slippage cost", "net P&L"],
                        "eur": [
                            subset["gross_pnl_eur"].sum(),
                            -subset["fees_eur"].sum(),
                            -subset["spread_cost_eur"].sum(),
                            -subset["slippage_cost_eur"].sum(),
                            subset["net_pnl_eur"].sum(),
                        ],
                    })
                    st.dataframe(costs, width="stretch")

        if not walk_forward.empty:
            st.subheader("Walk-forward results")
            st.dataframe(walk_forward, width="stretch")
        if not stability.empty:
            st.subheader("Parameter stability")
            st.dataframe(stability, width="stretch")


# --------------------------------------------------------------------------- #
# Event explorer
# --------------------------------------------------------------------------- #
elif page == "Event explorer":
    st.title("Event explorer")
    events = _read_frame(str(results_root / "event_dataset.parquet"))
    if events.empty:
        st.info("No event dataset found. Run `python scripts/run_research.py`.")
    else:
        markets = sorted(events["market"].dropna().unique())
        chosen_market = st.selectbox("Market", markets)
        subset = events[events["market"] == chosen_market].copy()
        subset["label"] = subset["event_time"].astype(str) + "  (+" + (
            subset["event_lookback_return"] * 100
        ).round(1).astype(str) + "%)"
        chosen_label = st.selectbox("Event", subset["label"].tolist())
        event = subset[subset["label"] == chosen_label].iloc[0]

        st.markdown(
            f"**{chosen_market}** — event at {format_display(event['event_time'])} "
            f"(+{event['event_lookback_return']:.2%} over {int(event['event_lookback_minutes'])} minutes)"
        )

        interval = config.get("research", "data", "base_interval", default="1m")
        parquet = ParquetStore(config.path("processed_dir"))
        candles = parquet.read_candles(chosen_market, interval)
        if candles.empty:
            st.warning(
                "Underlying candles are not in data/processed — the chart needs the downloaded "
                "history, which the event table alone does not contain."
            )
        else:
            import plotly.graph_objects as go

            centre = pd.Timestamp(event["event_time"])
            window = candles[
                (candles["timestamp"] >= centre - pd.Timedelta(hours=6))
                & (candles["timestamp"] <= centre + pd.Timedelta(hours=12))
            ].copy()
            window["local_time"] = window["timestamp"].dt.tz_convert(DISPLAY_TZ)

            figure = go.Figure(
                data=[go.Candlestick(
                    x=window["local_time"], open=window["open"], high=window["high"],
                    low=window["low"], close=window["close"], name=chosen_market,
                )]
            )
            figure.add_vline(x=to_display(centre), line_dash="dash", line_color="orange")
            figure.update_layout(height=520, xaxis_title="Europe/Amsterdam", yaxis_title="price (EUR)")
            st.plotly_chart(figure, width="stretch")

            st.subheader("Volume")
            st.bar_chart(window.set_index("local_time")["volume"])

        st.subheader("Forward outcomes (gross, before costs)")
        forward = {k: v for k, v in event.items() if str(k).startswith("fwd_")}
        if forward:
            st.dataframe(pd.DataFrame([forward]).T.rename(columns={0: "value"}), width="stretch")
        st.subheader("Features at the event (all causally available)")
        skip = {"market", "event_time", "label"}
        feature_row = {k: v for k, v in event.items() if k not in skip and not str(k).startswith("fwd_")}
        st.dataframe(pd.DataFrame([feature_row]).T.rename(columns={0: "value"}), width="stretch")


# --------------------------------------------------------------------------- #
# Trade journal
# --------------------------------------------------------------------------- #
elif page == "Trade journal":
    st.title("Trade journal")
    paper_dir = Path(config.root) / str(config.get("paper", "paper", "state_dir", default="data/results/paper"))
    journal = _read_frame(str(paper_dir / "trade_journal.parquet"))
    signals = _read_frame(str(paper_dir / "signal_log.parquet"))

    tab_trades, tab_signals = st.tabs(["Paper trades", "All signals (including rejected)"])
    with tab_trades:
        if journal.empty:
            st.info("No paper trades yet.")
        else:
            st.dataframe(_localise(journal, ["entry_time_utc", "exit_time_utc"]), width="stretch")
            stats = journal["net_pnl_eur"].describe()
            st.write(stats)
    with tab_signals:
        if signals.empty:
            st.info("No signals recorded yet.")
        else:
            st.caption(
                "Every signal is recorded, including ones that were rejected or expired — that is "
                "what makes the paper record auditable rather than a highlight reel."
            )
            st.dataframe(_localise(signals, ["timestamp_utc", "recorded_utc"]), width="stretch")


# --------------------------------------------------------------------------- #
# Data & status
# --------------------------------------------------------------------------- #
elif page == "Data & status":
    st.title("Data and system status")
    parquet = ParquetStore(config.path("processed_dir"))
    manifest = parquet.read_manifest()

    if manifest.empty:
        st.warning("No dataset manifest. Run `python scripts/download_history.py` first.")
    else:
        columns = st.columns(4)
        columns[0].metric("Datasets", len(manifest))
        columns[1].metric("Failed validation", int((manifest["validation_status"] == "FAIL").sum()))
        columns[2].metric("Total bars", f"{int(manifest['n_rows'].sum()):,}")
        columns[3].metric("Markets", manifest["market"].nunique())
        st.dataframe(_localise(manifest, ["first_timestamp", "last_timestamp",
                                          "download_started_utc", "download_finished_utc"]),
                     width="stretch")

    st.subheader("Configuration in force")
    left, right = st.columns(2)
    with left:
        st.caption("Execution scenarios")
        st.json(config.get("risk", "execution_scenarios", default={}))
    with right:
        st.caption("Risk limits")
        st.json(config.get("risk", "limits", default={}))

    st.subheader("Safety")
    st.markdown(
        "- Live order execution requires **three** independent switches and no order-placement "
        "code exists in this repository.\n"
        "- Credentials are read from environment variables only and are redacted from all logs.\n"
        "- Withdrawal permission must never be granted to the API key used here."
    )

    report_path = results_root / "research_report.md"
    if report_path.exists():
        st.subheader("Latest research report")
        st.markdown(report_path.read_text(encoding="utf-8"))
