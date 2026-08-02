"""Phase 14: research outputs.

Generates the deliverables listed in the brief. Two rules are enforced here
rather than left to discipline:

1. **Provenance.** Every report carries a header stating the data source, the
   number of markets, the date range, the number of configurations evaluated and
   the execution scenario. A report generated from synthetic data carries an
   unmissable banner and the word SYNTHETIC in its title.
2. **No invented numbers.** Report functions render values that were passed to
   them; they never estimate, extrapolate or fill gaps. Where a statistic could
   not be computed the report says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .timeutils import format_display, now_utc

log = get_logger(__name__)

SYNTHETIC_WARNING = (
    "> **SYNTHETIC DATA - NOT A RESEARCH RESULT.**\n"
    "> Every number below was computed from artificially generated price series\n"
    "> produced by `bitvavo_momentum.synthetic`. They demonstrate that the pipeline\n"
    "> executes; they say nothing whatsoever about real Bitvavo markets.\n"
)

NO_DATA_NOTICE = (
    "> **NO MARKET DATA WAS AVAILABLE IN THIS ENVIRONMENT.**\n"
    "> The Bitvavo API was unreachable, so no historical candles were downloaded and\n"
    "> no empirical result is reported. Run `scripts/download_history.py` from a machine\n"
    "> with access to `api.bitvavo.com`, then re-run the research pipeline.\n"
)


@dataclass
class RunContext:
    """Everything needed to make a report auditable."""

    data_source: str = "not run"
    n_markets: int = 0
    first_timestamp: Any = None
    last_timestamp: Any = None
    n_events: int = 0
    n_configurations_evaluated: int = 0
    execution_scenario: str = "realistic"
    is_synthetic: bool = False
    has_real_data: bool = False
    code_version: str = "0.1.0"
    notes: list[str] = field(default_factory=list)

    def header_markdown(self) -> str:
        lines = [
            "| field | value |",
            "|---|---|",
            f"| generated (UTC) | {now_utc().isoformat()} |",
            f"| generated (Europe/Amsterdam) | {format_display(now_utc())} |",
            f"| data source | {self.data_source} |",
            f"| markets | {self.n_markets} |",
            f"| data range | {self.first_timestamp} .. {self.last_timestamp} |",
            f"| events detected | {self.n_events} |",
            f"| configurations evaluated | {self.n_configurations_evaluated} |",
            f"| headline execution scenario | {self.execution_scenario} |",
            f"| code version | {self.code_version} |",
        ]
        return "\n".join(lines)

    def banner(self) -> str:
        if self.is_synthetic:
            return SYNTHETIC_WARNING
        if not self.has_real_data:
            return NO_DATA_NOTICE
        return ""


def _fmt(value: Any, spec: str = ".4f", missing: str = "not available") -> str:
    if value is None:
        return missing
    if isinstance(value, float | int | np.floating | np.integer):
        if not np.isfinite(float(value)):
            return missing
        return format(float(value), spec)
    return str(value)


def _pct(value: Any, digits: int = 2) -> str:
    return _fmt(value, f".{digits}%")


def frame_to_markdown(frame: pd.DataFrame, max_rows: int = 25, float_format: str = "{:.4f}") -> str:
    if frame is None or frame.empty:
        return "_no rows_"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda v: float_format.format(v) if np.isfinite(v) else "")
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    divider = "|" + "|".join(["---"] * len(view.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()]
    suffix = f"\n\n_showing {max_rows} of {len(frame)} rows_" if len(frame) > max_rows else ""
    return "\n".join([header, divider, *rows]) + suffix


# --------------------------------------------------------------------------- #
# research_report.md
# --------------------------------------------------------------------------- #
def build_research_report(
    context: RunContext,
    event_study: pd.DataFrame | None = None,
    strategy_comparison: pd.DataFrame | None = None,
    headline_metrics: dict[str, Any] | None = None,
    stress_metrics: dict[str, Any] | None = None,
    split_comparison: pd.DataFrame | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
    robustness_summary: dict[str, Any] | None = None,
    concentration: dict[str, Any] | None = None,
    rejection_reasons: list[str] | None = None,
    benchmarks: pd.DataFrame | None = None,
    verdict: str | None = None,
) -> str:
    """Assemble ``research_report.md``, answering the brief's questions in order."""
    parts: list[str] = []
    title = "Bitvavo short-horizon momentum research report"
    if context.is_synthetic:
        title += " (SYNTHETIC PIPELINE VALIDATION - NOT A RESULT)"
    parts.append(f"# {title}\n")
    banner = context.banner()
    if banner:
        parts.append(banner + "\n")
    parts.append("## Run provenance\n")
    parts.append(context.header_markdown() + "\n")

    parts.append("## Hypothesis under test\n")
    parts.append(
        "A cryptocurrency that rises roughly 10-12% within two to three hours continues to rise "
        "often enough to produce a positive net expectancy after realistic fees, spread, slippage "
        "and failed fills.\n\n"
        "The null hypothesis - that the forward return distribution after such an event is "
        "indistinguishable from what a randomly-timed entry in the same market would have "
        "produced - is the one that must be rejected by evidence, not by narrative.\n"
    )

    if not context.has_real_data:
        parts.append("## Status\n")
        parts.append(
            "No empirical result is reported because no market data was available. "
            "Every section below that would contain numbers is marked accordingly. "
            "This is a deliberate outcome: reporting a figure without data behind it "
            "would be fabrication.\n"
        )

    parts.append("## 1. Does a 10-12% rise over two to three hours predict positive future returns?\n")
    if event_study is not None and not event_study.empty:
        parts.append(
            "Forward returns are measured from the **open of the bar after the event bar** - the "
            "first price a trader could realistically have transacted at. Costs are excluded here; "
            "this section is descriptive, not a strategy.\n"
        )
        parts.append(frame_to_markdown(event_study, max_rows=40) + "\n")
        parts.append(_interpret_event_study(event_study) + "\n")
    else:
        parts.append("_Not computed - no event study available._\n")

    parts.append("## 2. For how long does any continuation effect persist?\n")
    if event_study is not None and not event_study.empty and "horizon_minutes" in event_study.columns:
        parts.append(_interpret_persistence(event_study) + "\n")
    else:
        parts.append("_Not computed._\n")

    parts.append("## 3-4. Immediate entry versus pullback, breakout and retest entries\n")
    if strategy_comparison is not None and not strategy_comparison.empty:
        columns = [
            c for c in ["strategy", "n_trades", "win_rate", "net_expectancy", "net_expectancy_bps",
                        "profit_factor", "max_drawdown", "avg_holding_minutes"]
            if c in strategy_comparison.columns
        ]
        parts.append(frame_to_markdown(strategy_comparison[columns], max_rows=40) + "\n")
    else:
        parts.append("_Not computed._\n")

    parts.append("## 5. Which filters add genuine out-of-sample value?\n")
    if robustness_summary and "ablation" in robustness_summary:
        parts.append(frame_to_markdown(robustness_summary["ablation"], max_rows=20) + "\n")
    else:
        parts.append("_Not computed - requires a validated base strategy and an ablation run._\n")

    parts.append("## 6. Which coins and market regimes should be excluded?\n")
    if robustness_summary and "regime_performance" in robustness_summary:
        parts.append(frame_to_markdown(robustness_summary["regime_performance"], max_rows=20) + "\n")
    else:
        parts.append("_Not computed._\n")

    parts.append("## 7-11. The most robust strategy, its evidence and its sensitivity\n")
    if headline_metrics:
        parts.append(_headline_block(headline_metrics, stress_metrics, concentration) + "\n")
    else:
        parts.append("_No strategy reached the point of having headline metrics._\n")

    parts.append("## 12. Did it survive the untouched test set?\n")
    if split_comparison is not None and not split_comparison.empty:
        parts.append(frame_to_markdown(split_comparison, max_rows=10) + "\n")
    else:
        parts.append(
            "_Not evaluated. The test set stays locked (`splits.unlock_test_set: false`) until a "
            "candidate has been selected on train/validation data alone._\n"
        )

    if walk_forward_summary:
        parts.append("### Walk-forward record\n")
        parts.append(
            "\n".join(f"- **{k}**: {_fmt(v)}" for k, v in walk_forward_summary.items()) + "\n"
        )

    if benchmarks is not None and not benchmarks.empty:
        parts.append("## Benchmarks\n")
        parts.append(frame_to_markdown(benchmarks, max_rows=20) + "\n")

    parts.append("## 13. Verdict: is it suitable for paper trading?\n")
    if verdict:
        parts.append(verdict + "\n")
    elif rejection_reasons:
        parts.append(
            "**Rejected.** The following criteria were not met:\n\n"
            + "\n".join(f"- {r}" for r in rejection_reasons)
            + "\n"
        )
    elif not context.has_real_data:
        parts.append(
            "**Insufficient evidence.** No data was available, therefore no strategy is approved "
            "for paper trading. This is the correct conclusion given the inputs, and the brief "
            "explicitly prefers it to a manufactured one.\n"
        )
    else:
        parts.append("_No verdict recorded._\n")

    parts.append("## Reproduction\n")
    parts.append(
        "```bash\n"
        "python scripts/download_history.py --markets all --interval 1m\n"
        "python scripts/run_research.py\n"
        "python scripts/run_backtest.py\n"
        "python scripts/run_walk_forward.py\n"
        "```\n"
    )
    return "\n".join(parts)


def _interpret_event_study(study: pd.DataFrame) -> str:
    if "mean_return" not in study.columns:
        return ""
    rows = []
    for _, row in study.iterrows():
        horizon = row.get("horizon_minutes")
        mean = row.get("mean_return")
        n = row.get("n_events")
        hit = row.get("hit_rate")
        if not np.isfinite(float(mean or np.nan)):
            continue
        direction = "positive" if mean > 0 else "negative"
        rows.append(
            f"- at **{horizon} minutes**: mean forward return {mean:+.3%} ({direction}), "
            f"hit rate {hit:.1%}, n={int(n)}"
        )
    caveat = (
        "\n\nThese are *gross* figures. A strategy must clear roughly "
        "2 x (fee + half-spread + slippage) per round trip before any of this becomes tradeable, "
        "which for the realistic scenario is a materially higher bar than the raw means above. "
        "The naive t-statistics in the table assume independent events; overlapping events across "
        "correlated coins violate that, so the bootstrap results are the ones to trust."
    )
    return "\n".join(rows) + caveat


def _interpret_persistence(study: pd.DataFrame) -> str:
    grouped = study.groupby("horizon_minutes")["mean_return"].mean()
    if grouped.empty:
        return "_Not computed._"
    positive = grouped[grouped > 0]
    if positive.empty:
        return (
            "Mean forward returns are non-positive at every horizon tested: on this data the event "
            "shows no continuation effect at all."
        )
    peak = grouped.idxmax()
    last_positive = int(positive.index.max())
    return (
        f"Mean forward return peaks at **{peak} minutes** and remains positive out to "
        f"**{last_positive} minutes** on this data. Beyond that horizon the average effect is not "
        "positive, which bounds the useful holding period."
    )


def _headline_block(
    metrics: dict[str, Any],
    stress: dict[str, Any] | None,
    concentration: dict[str, Any] | None,
) -> str:
    lines = [
        "| metric | realistic scenario | stress scenario |",
        "|---|---|---|",
    ]
    keys = [
        ("n_trades", "trades", "{:.0f}"),
        ("win_rate", "win rate", "{:.1%}"),
        ("payoff_ratio", "payoff ratio", "{:.2f}"),
        ("net_expectancy", "net expectancy per trade", "{:.3%}"),
        ("profit_factor", "profit factor", "{:.2f}"),
        ("total_net_pnl_eur", "total net P&L (EUR)", "{:.2f}"),
        ("max_drawdown", "maximum drawdown", "{:.1%}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("avg_holding_minutes", "average holding (minutes)", "{:.0f}"),
        ("total_fees_eur", "total fees (EUR)", "{:.2f}"),
        ("ambiguous_exit_share", "intrabar-ambiguous exits", "{:.1%}"),
    ]
    def _render(value: Any, spec: str) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "not available"
        try:
            return spec.format(value)
        except (TypeError, ValueError):
            return str(value)

    for key, label, fmt in keys:
        a = _render(metrics.get(key), fmt)
        b = _render((stress or {}).get(key), fmt)
        lines.append(f"| {label} | {a} | {b} |")

    if concentration:
        lines.append("")
        lines.append(
            f"Profit concentration: the largest single coin contributed "
            f"{_pct(concentration.get('top_coin_profit_share'))} of gross profit "
            f"({concentration.get('top_coin')}), and the largest single month "
            f"{_pct(concentration.get('top_month_profit_share'))} ({concentration.get('top_month')})."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# other deliverables
# --------------------------------------------------------------------------- #
def build_rejected_strategies(rejections: list[dict[str, Any]], context: RunContext) -> str:
    """``rejected_strategies.md`` - what was tried and why it was discarded."""
    parts = ["# Rejected strategies\n"]
    banner = context.banner()
    if banner:
        parts.append(banner + "\n")
    parts.append(
        "Recording rejections is as important as recording the winner: it is the only way to "
        "know how many hypotheses were tested, which is what makes the surviving result "
        "interpretable rather than a survivor of an unrecorded search.\n"
    )
    if not rejections:
        parts.append("_No strategies have been evaluated yet._\n")
        return "\n".join(parts)
    for item in rejections:
        parts.append(f"## {item.get('name', 'unnamed')}\n")
        parts.append(f"- configuration: `{item.get('config', {})}`")
        parts.append(f"- trades: {item.get('n_trades', 'n/a')}")
        parts.append(f"- net expectancy: {_pct(item.get('net_expectancy'), 3)}")
        parts.append("- rejected because:")
        for reason in item.get("reasons", []):
            parts.append(f"  - {reason}")
        parts.append("")
    return "\n".join(parts)


def build_assumptions_document(
    context: RunContext,
    execution_description: dict[str, Any] | None = None,
    extra_limitations: list[str] | None = None,
) -> str:
    """``assumptions_and_limitations.md``."""
    parts = ["# Assumptions and limitations\n"]
    banner = context.banner()
    if banner:
        parts.append(banner + "\n")

    parts.append("## Data\n")
    parts.append(
        "- **Interval.** All research runs on 1-minute candles; coarser intervals are aggregated "
        "locally from the same 1-minute bars so timeframes cannot disagree.\n"
        "- **Missing candles.** A missing Bitvavo candle means *no trades occurred*, not "
        "*price unchanged*. Missing bars are kept as explicit gaps, counted in the manifest, and "
        "events whose look-back window is more than 20% gaps are discarded.\n"
        "- **Depth of history.** Bitvavo's public candle endpoint does not serve unlimited "
        "1-minute history. The true first available timestamp is probed per market and recorded; "
        "no assumption is made about how far back data goes.\n"
        "- **Survivorship.** Delisted and suspended markets are retained in the research universe "
        "wherever local history exists. Excluding them would remove exactly the coins that failed.\n"
        "- **Microstructure.** Historical candles contain no quotes. Bid-ask spread and order-book "
        "imbalance are therefore *proxies* (`spread_proxy_bps`, Amihud illiquidity), never "
        "measurements, and are named as such throughout.\n"
    )

    parts.append("## Execution\n")
    if execution_description:
        parts.append(
            "\n".join(f"- **{k}**: {_fmt(v)}" for k, v in execution_description.items()) + "\n"
        )
    parts.append(
        "- Entries never fill on the bar that generated the signal.\n"
        "- Stop and take-profit within the same bar resolve to the **stop** (conservative); such "
        "trades are flagged and their share is reported.\n"
        "- Stops that gap fill at the bar open plus extra slippage, not at the stop price.\n"
        "- Limit orders that are merely touched are not assumed filled.\n"
        "- Order size is capped by a share of the bar's volume; capacity is a constraint, not a fee.\n"
    )

    parts.append("## Statistics\n")
    parts.append(
        "- Momentum events cluster across coins. Trades are therefore **not independent**, and the "
        "clustered count (`independent_trade_count`) is the honest sample size, not the raw trade "
        "count.\n"
        "- Confidence intervals use a moving-block bootstrap to preserve that dependence.\n"
        "- The Sharpe ratio is deflated for the number of configurations evaluated.\n"
        "- Sharpe, Sortino and annualised return are suppressed below 20 trades rather than "
        "reported as noise.\n"
    )

    parts.append("## Scope\n")
    parts.append(
        "- Spot only. No short selling, no leverage, no margin, no derivatives.\n"
        "- Live order execution is disabled and no order-placement code exists in this repository.\n"
        "- Results describe the past behaviour of a rule on a specific dataset. They are not a "
        "prediction, and nothing here should be read as advice or as an expectation of profit.\n"
    )

    if extra_limitations:
        parts.append("## Environment-specific limitations\n")
        parts.extend(f"- {item}" for item in extra_limitations)
        parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def build_html_report(
    context: RunContext,
    figures: dict[str, Any] | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    title: str = "Bitvavo momentum research",
) -> str:
    """Self-contained HTML report with Plotly figures inlined."""
    figures = figures or {}
    tables = tables or {}
    banner_html = ""
    if context.is_synthetic:
        banner_html = (
            '<div class="banner">SYNTHETIC DATA - NOT A RESEARCH RESULT. '
            "Generated from artificial price series to validate the pipeline.</div>"
        )
    elif not context.has_real_data:
        banner_html = (
            '<div class="banner">NO MARKET DATA AVAILABLE - no empirical result is reported.</div>'
        )

    blocks: list[str] = []
    include_js = True
    for name, figure in figures.items():
        blocks.append(f"<h2>{name}</h2>")
        try:
            blocks.append(
                figure.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)
            )
            include_js = False
        except Exception:
            log.exception("Could not render figure %s", name)
            blocks.append("<p><em>figure could not be rendered</em></p>")

    for name, frame in tables.items():
        blocks.append(f"<h2>{name}</h2>")
        blocks.append(frame.head(200).to_html(index=False, classes="data", border=0))

    header_rows = "".join(
        f"<tr><th>{line.split('|')[1].strip()}</th><td>{line.split('|')[2].strip()}</td></tr>"
        for line in context.header_markdown().splitlines()[2:]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 2rem auto; max-width: 1100px; color: #1a1a1a; line-height: 1.5; }}
 h1 {{ border-bottom: 2px solid #ddd; padding-bottom: .4rem; }}
 h2 {{ margin-top: 2.5rem; }}
 .banner {{ background: #fff3cd; border: 2px solid #d39e00; padding: 1rem;
            font-weight: 700; margin-bottom: 1.5rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
 table.data td, table.data th {{ border: 1px solid #ddd; padding: .35rem .5rem; }}
 table.data tr:nth-child(even) {{ background: #fafafa; }}
 th {{ text-align: left; background: #f0f0f0; }}
 .meta td, .meta th {{ padding: .3rem .6rem; border-bottom: 1px solid #eee; }}
</style></head><body>
<h1>{title}</h1>
{banner_html}
<table class="meta">{header_rows}</table>
{"".join(blocks)}
<hr><p style="color:#666;font-size:.8rem">
Generated by bitvavo-momentum-agent {context.code_version} at {format_display(now_utc())}.
Past behaviour of a rule on historical data. Not advice, not a forecast, no promise of profit.
</p></body></html>"""


def write_all(
    result_store,
    context: RunContext,
    report_markdown: str,
    backtest_summary: pd.DataFrame | None = None,
    strategy_comparison: pd.DataFrame | None = None,
    trade_log: pd.DataFrame | None = None,
    event_dataset: pd.DataFrame | None = None,
    walk_forward_results: pd.DataFrame | None = None,
    parameter_stability: pd.DataFrame | None = None,
    rejected_markdown: str | None = None,
    assumptions_markdown: str | None = None,
    html_report: str | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    """Write every Phase 14 deliverable that has content."""
    written: dict[str, Path] = {}

    def _p(name: str) -> str:
        return f"{prefix}{name}"

    written["research_report.md"] = result_store.write_text(_p("research_report.md"), report_markdown)
    if backtest_summary is not None and not backtest_summary.empty:
        written["backtest_summary.csv"] = result_store.write_frame(_p("backtest_summary.csv"), backtest_summary)
    if strategy_comparison is not None and not strategy_comparison.empty:
        written["strategy_comparison.csv"] = result_store.write_frame(
            _p("strategy_comparison.csv"), strategy_comparison
        )
    if trade_log is not None and not trade_log.empty:
        written["trade_log.parquet"] = result_store.write_frame(_p("trade_log.parquet"), trade_log)
    if event_dataset is not None and not event_dataset.empty:
        written["event_dataset.parquet"] = result_store.write_frame(_p("event_dataset.parquet"), event_dataset)
    if walk_forward_results is not None and not walk_forward_results.empty:
        written["walk_forward_results.csv"] = result_store.write_frame(
            _p("walk_forward_results.csv"), walk_forward_results
        )
    if parameter_stability is not None and not parameter_stability.empty:
        written["parameter_stability.csv"] = result_store.write_frame(
            _p("parameter_stability.csv"), parameter_stability
        )
    if rejected_markdown:
        written["rejected_strategies.md"] = result_store.write_text(_p("rejected_strategies.md"), rejected_markdown)
    if assumptions_markdown:
        written["assumptions_and_limitations.md"] = result_store.write_text(
            _p("assumptions_and_limitations.md"), assumptions_markdown
        )
    if html_report:
        written["report.html"] = result_store.write_text(_p("report.html"), html_report)

    result_store.write_json(_p("run_context.json"), context.__dict__)
    log.info("Wrote %d research artefacts", len(written))
    return written


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def equity_curve_figure(equity: pd.DataFrame, title: str = "Equity curve"):
    import plotly.graph_objects as go

    figure = go.Figure()
    if equity is not None and not equity.empty:
        figure.add_trace(
            go.Scatter(x=equity["timestamp"], y=equity["equity"], name="equity", mode="lines")
        )
    figure.update_layout(title=title, xaxis_title="date (UTC)", yaxis_title="equity (EUR)", height=420)
    return figure


def drawdown_figure(equity: pd.DataFrame, title: str = "Drawdown"):
    import plotly.graph_objects as go

    figure = go.Figure()
    if equity is not None and not equity.empty and "drawdown" in equity.columns:
        figure.add_trace(
            go.Scatter(x=equity["timestamp"], y=equity["drawdown"], name="drawdown",
                       fill="tozeroy", mode="lines")
        )
    figure.update_layout(title=title, xaxis_title="date (UTC)", yaxis_title="drawdown", height=320)
    return figure


def trade_distribution_figure(trades: pd.DataFrame, title: str = "Net return per trade"):
    import plotly.graph_objects as go

    figure = go.Figure()
    if trades is not None and not trades.empty and "net_return" in trades.columns:
        figure.add_trace(go.Histogram(x=trades["net_return"].dropna(), nbinsx=60, name="net return"))
    figure.update_layout(title=title, xaxis_title="net return per trade", yaxis_title="count", height=380)
    return figure


def parameter_heatmap_figure(pivot: pd.DataFrame, title: str = "Parameter stability"):
    import plotly.graph_objects as go

    figure = go.Figure()
    if pivot is not None and not pivot.empty:
        figure.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=[str(c) for c in pivot.columns],
                y=[str(i) for i in pivot.index],
                colorscale="RdBu",
                zmid=0,
            )
        )
    figure.update_layout(title=title, height=480)
    return figure


def monte_carlo_figure(samples, title: str = "Monte Carlo final equity distribution"):
    import plotly.graph_objects as go

    figure = go.Figure()
    if samples is not None and len(samples):
        figure.add_trace(go.Histogram(x=samples, nbinsx=60, name="final equity"))
    figure.update_layout(title=title, xaxis_title="final equity (EUR)", yaxis_title="count", height=380)
    return figure


def event_study_figure(study: pd.DataFrame, title: str = "Mean forward return by horizon"):
    import plotly.graph_objects as go

    figure = go.Figure()
    if study is not None and not study.empty:
        grouped = study.groupby("horizon_minutes")["mean_return"].mean().reset_index()
        figure.add_trace(
            go.Bar(x=grouped["horizon_minutes"].astype(str), y=grouped["mean_return"], name="mean return")
        )
    figure.update_layout(
        title=title, xaxis_title="horizon (minutes)", yaxis_title="mean forward return (gross)", height=380
    )
    return figure
