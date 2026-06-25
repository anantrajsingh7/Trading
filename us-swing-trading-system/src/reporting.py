"""
Report generation:
  reports/backtest_results.csv
  reports/strategy_ranking.csv
  reports/daily_watchlist.csv
  reports/top_candidates.md
  reports/equity_curve.png
  reports/strategy_comparison.md
"""
import logging
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import BacktestResult
from scanner import Candidate

log = logging.getLogger("swing.reporting")

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def _ensure_reports():
    REPORTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Strategy ranking
# ---------------------------------------------------------------------------

def _rank_strategies(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Rank by composite score:
      profit_factor (40%) + (1 - |max_drawdown|) (25%) +
      win_rate (20%) + trade_count_score (15%)
    """
    rows = []
    for key, res in results.items():
        n_score = min(res.n_trades / 50, 1.0)  # normalise to 0-1
        dd_score = max(0.0, 1.0 + res.max_drawdown)  # max_drawdown is negative
        pf_capped = min(res.profit_factor, 5.0) / 5.0
        composite = (pf_capped * 0.40 + dd_score * 0.25 +
                     res.win_rate * 0.20 + n_score * 0.15)
        d = res.to_dict()
        d["composite_score"] = round(composite * 10, 2)  # out of 10
        rows.append(d)
    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def save_backtest_results(results: dict[str, BacktestResult]) -> Path:
    _ensure_reports()
    rows = [res.to_dict() for res in results.values()]
    df = pd.DataFrame(rows)
    p = REPORTS / "backtest_results.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p)
    return p


def save_strategy_ranking(results: dict[str, BacktestResult]) -> tuple[Path, str]:
    _ensure_reports()
    df = _rank_strategies(results)
    p = REPORTS / "strategy_ranking.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p)
    best = df.iloc[0]["strategy"] if len(df) > 0 else ""
    return p, best


def save_equity_curve(results: dict[str, BacktestResult]) -> Path:
    _ensure_reports()
    fig, ax = plt.subplots(figsize=(14, 7))
    for key, res in results.items():
        if res.equity_curve is not None and len(res.equity_curve) > 1:
            ax.plot(res.equity_curve.index, res.equity_curve.values,
                    label=res.strategy, linewidth=1.5)
    ax.set_title("Strategy Equity Curves", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = REPORTS / "equity_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    log.info("Saved %s", p)
    return p


def save_strategy_comparison(results: dict[str, BacktestResult]) -> Path:
    _ensure_reports()
    ranking = _rank_strategies(results)

    lines = [
        "# Strategy Comparison Report",
        f"\nGenerated: {date.today()}",
        "\n## Rankings\n",
        ranking.to_markdown(index=False) if hasattr(ranking, "to_markdown") else ranking.to_string(),
        "\n\n## Detailed Analysis\n",
    ]
    for _, row in ranking.iterrows():
        lines.append(f"### {row['rank']}. {row['strategy']}")
        lines.append(f"- Composite score: **{row['composite_score']}/10**")
        lines.append(f"- Total return: {row['total_return_%']}%")
        lines.append(f"- CAGR: {row['cagr_%']}%")
        lines.append(f"- Win rate: {row['win_rate_%']}%")
        lines.append(f"- Profit factor: {row['profit_factor']}")
        lines.append(f"- Max drawdown: {row['max_drawdown_%']}%")
        lines.append(f"- Sharpe ratio: {row['sharpe']}")
        lines.append(f"- Avg hold: {row['avg_holding_days']} days")
        lines.append(f"- Trades: {row['n_trades']}")
        lines.append(f"- Best ticker: {row['best_ticker']}")
        lines.append(f"- Worst ticker: {row['worst_ticker']}")
        lines.append("")

    p = REPORTS / "strategy_comparison.md"
    p.write_text("\n".join(lines))
    log.info("Saved %s", p)
    return p


# ---------------------------------------------------------------------------
# Daily watchlist
# ---------------------------------------------------------------------------

def save_daily_watchlist(candidates: list[Candidate]) -> Path:
    _ensure_reports()
    if not candidates:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame([c.to_dict() for c in candidates])
    p = REPORTS / "daily_watchlist.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p)
    return p


# ---------------------------------------------------------------------------
# Top candidates markdown
# ---------------------------------------------------------------------------

def _fmt_candidate(c: Candidate, rank: int) -> list[str]:
    rs_spy_str = (f"{c.rs_spy*100:.1f}%"
                  if not np.isnan(c.rs_spy) else "N/A")
    rs_qqq_str = (f"{c.rs_qqq*100:.1f}%"
                  if not np.isnan(c.rs_qqq) else "N/A")
    return [
        f"### {rank}. {c.ticker} – {c.company}",
        f"- **Strategy:** {c.strategy_name}",
        f"- **Current price:** ${c.current_price:.2f}",
        f"- **Entry trigger:** ${c.entry:.2f}",
        f"- **Stop loss:** ${c.stop_loss:.2f}",
        f"- **Target 1:** ${c.target1:.2f}",
        f"- **Target 2:** ${c.target2:.2f}",
        f"- **Risk per share:** ${c.risk_per_share:.2f}",
        f"- **Risk/reward:** {c.rr_ratio:.1f}:1",
        f"- **Shares:** {c.shares}",
        f"- **Position value:** ${c.position_value:,.0f}",
        f"- **Setup score:** {c.setup_score}/10",
        f"- **RS vs SPY:** {rs_spy_str}",
        f"- **RS vs QQQ:** {rs_qqq_str}",
        f"- **Volume:** {c.volume_ratio:.1f}x average",
        f"- **Trend:** {c.trend_status}",
        f"- **Why it qualifies:** {c.why_qualifies}",
        f"- **Why it may fail:** {c.why_may_fail}",
        f"- **Invalidation:** {c.invalidation}",
        "",
    ]


def save_top_candidates(
    candidates: list[Candidate],
    results: dict[str, BacktestResult],
    regime: dict,
    cfg: dict,
) -> Path:
    _ensure_reports()
    ranking = _rank_strategies(results) if results else pd.DataFrame()
    best_res: BacktestResult | None = None
    best_name = ""
    if len(ranking) > 0:
        best_name = str(ranking.iloc[0]["strategy"])
        for res in results.values():
            if res.strategy == best_name:
                best_res = res
                break

    # split candidates
    top = [c for c in candidates if c.setup_score >= 7.0][:10]
    watchlist = [c for c in candidates if 5.0 <= c.setup_score < 7.0][:10]
    avoid = [c for c in candidates if c.setup_score < 5.0][:5]

    if regime.get("risk_mode") == "no-trade":
        final_decision = "No high-quality swing trade today."
    elif top:
        final_decision = "High-quality candidates available"
    elif watchlist:
        final_decision = "Watchlist only, no entry yet"
    else:
        final_decision = "No high-quality swing trade today."

    lines = [
        "# US Swing Trading Daily Report",
        f"\n**Date:** {date.today()}\n",
        "---\n",
        "## Market Regime",
        f"- **SPY trend:** {regime.get('spy_trend', 'N/A')}",
        f"- **QQQ trend:** {regime.get('qqq_trend', 'N/A')}",
        f"- **Risk mode:** {regime.get('risk_mode', 'N/A')}",
        "",
    ]

    if best_res:
        weakness = "Low trade count" if best_res.n_trades < 20 else \
                   "High drawdown" if best_res.max_drawdown < -0.20 else \
                   "Below-average win rate" if best_res.win_rate < 0.45 else "None identified"
        lines += [
            "## Best Backtested Strategy",
            f"- **Strategy name:** {best_res.strategy}",
            f"- **Win rate:** {best_res.win_rate*100:.1f}%",
            f"- **Profit factor:** {best_res.profit_factor:.2f}",
            f"- **Max drawdown:** {best_res.max_drawdown*100:.1f}%",
            f"- **Average holding period:** {best_res.avg_holding:.1f} trading days",
            f"- **Weakness:** {weakness}",
            "",
        ]

    lines += ["---\n", "## Today's Top Buy Candidates\n"]
    if top:
        for i, c in enumerate(top, 1):
            lines += _fmt_candidate(c, i)
    else:
        lines.append("_No high-quality candidates today._\n")

    lines += ["---\n", "## Watchlist Only\n",
              "_Stocks close to a setup but not ready for entry._\n"]
    if watchlist:
        for i, c in enumerate(watchlist, 1):
            lines += _fmt_candidate(c, i)
    else:
        lines.append("_No watchlist candidates._\n")

    lines += ["---\n", "## Avoid\n",
              "_Stocks that look popular but do not meet strategy rules._\n"]
    if avoid:
        for c in avoid:
            lines.append(f"- **{c.ticker}** ({c.company}): {c.why_may_fail}")
    else:
        lines.append("_None flagged._\n")

    lines += [
        "\n---\n",
        "## Final Decision\n",
        f"**{final_decision}**\n",
        "\n---",
        "\n> *This report is for educational and research purposes only.",
        "> It is NOT financial advice. Do not trade based solely on this output.",
        "> Always do your own due diligence.*",
    ]

    p = REPORTS / "top_candidates.md"
    p.write_text("\n".join(lines))
    log.info("Saved %s", p)
    return p
