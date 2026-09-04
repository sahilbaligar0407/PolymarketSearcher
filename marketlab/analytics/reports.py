"""Terminal (rich) and JSON reporting.

**Sample separation is enforced in the type system, not by convention.** Every report
entry point that spans more than one strategy-lifecycle phase takes a
:class:`PhaseSeries` — four explicitly named, independently optional slots
(``in_sample``, ``validation``, ``forward_paper``, ``live``) — instead of a single
sequence. There is deliberately no constructor that accepts "all periods blended
together": a caller who wants a report has to say, for each phase, what (if anything)
happened in it. Individual table-builder functions below operate on one already-selected
phase's data (a plain ``Sequence[StrategyResult]``); the phase-spanning orchestrators
(:func:`full_league_report`, :func:`experiment_report`, :func:`compare_experiments`) are
the enforcement point and are the only functions that accept a :class:`PhaseSeries`.

**Losers are findings, not embarrassments.** Section headers below are neutral or
explicitly diagnostic ("WORST DRAWDOWN", "MOST LATENCY-SENSITIVE"), and
:func:`findings_section` turns dead/infeasible/miscalibrated strategies into plain-language
statements of what was learned, not apologies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text


class Phase(StrEnum):
    IN_SAMPLE = "IN_SAMPLE"
    VALIDATION = "VALIDATION"
    FORWARD_PAPER = "FORWARD_PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class PhaseSeries[T]:
    """Four independently-optional phase slots. See module docstring: this is the
    mechanism that makes blending phases into one report a type error, not a discipline
    problem."""

    in_sample: T | None = None
    validation: T | None = None
    forward_paper: T | None = None
    live: T | None = None

    def items(self) -> list[tuple[Phase, T]]:
        pairs: list[tuple[Phase, T | None]] = [
            (Phase.IN_SAMPLE, self.in_sample),
            (Phase.VALIDATION, self.validation),
            (Phase.FORWARD_PAPER, self.forward_paper),
            (Phase.LIVE, self.live),
        ]
        return [(p, v) for p, v in pairs if v is not None]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _money(d: Decimal) -> str:
    return f"${d:,.2f}"


def _signed_money(d: Decimal) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:,.2f}"


def _dd_pct(dd: Decimal | None) -> str:
    if dd is None:
        return "-"
    return f"{-float(dd) * 100:.1f}%"


def _pct(v: Decimal | float | None) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.1f}%"


def _fmt(v: Decimal | float | None, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{float(v):.{digits}f}"


# ---------------------------------------------------------------------------
# StrategyResult: the row unit for every league-style table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyResult:
    """One strategy's already-computed summary for one phase. Built by the caller from
    :mod:`marketlab.analytics.metrics` / :mod:`marketlab.analytics.calibration` /
    :mod:`marketlab.analytics.attribution` output — this module only renders."""

    strategy_id: str
    category: str
    equity: Decimal
    net_pnl: Decimal
    max_drawdown: Decimal | None
    n_trades: int
    status: str

    # risk-adjusted
    sharpe_like: float | None = None
    sortino: float | None = None
    return_per_max_exposure: float | None = None

    # calibration
    brier_score: float | None = None
    improvement_vs_market: float | None = None

    # consistency: lower is more consistent (e.g. stdev of per-period returns)
    consistency_std: float | None = None

    # execution
    execution_slippage: Decimal | None = None

    # overfitting
    overfit_score: float | None = None
    n_variants_tried: int = 1

    # latency sensitivity
    breakeven_latency_ms: float | None = None
    laptop_infeasible: bool = False

    #: Plain-language, pre-written finding for this strategy, if the caller has one
    #: (e.g. "edge vanishes above 250ms latency"). Folded into findings_section().
    finding: str = ""


# ---------------------------------------------------------------------------
# STRATEGY LEAGUE
# ---------------------------------------------------------------------------


def strategy_league(results: Sequence[StrategyResult]) -> Table:
    table = Table(title="STRATEGY LEAGUE")
    table.add_column("Rank", justify="right")
    table.add_column("Strategy")
    table.add_column("Equity", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("DD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Status")
    ranked = sorted(results, key=lambda r: r.net_pnl, reverse=True)
    for i, r in enumerate(ranked, start=1):
        table.add_row(
            str(i),
            r.strategy_id,
            _money(r.equity),
            _signed_money(r.net_pnl),
            _dd_pct(r.max_drawdown),
            str(r.n_trades),
            r.status,
        )
    return table


def best_by_category(results: Sequence[StrategyResult]) -> Table:
    table = Table(title="BEST BY CATEGORY")
    table.add_column("Category")
    table.add_column("Strategy")
    table.add_column("Net P&L", justify="right")
    table.add_column("Trades", justify="right")
    by_cat: dict[str, list[StrategyResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    for cat in sorted(by_cat):
        best = max(by_cat[cat], key=lambda r: r.net_pnl)
        table.add_row(cat, best.strategy_id, _signed_money(best.net_pnl), str(best.n_trades))
    return table


def best_risk_adjusted(results: Sequence[StrategyResult], top_n: int = 10) -> Table:
    table = Table(title="BEST RISK-ADJUSTED")
    table.add_column("Strategy")
    table.add_column("Return/MaxExposure", justify="right")
    table.add_column("Sharpe-like", justify="right")
    table.add_column("Sortino", justify="right")
    scored = [r for r in results if r.return_per_max_exposure is not None]
    scored.sort(key=lambda r: r.return_per_max_exposure or 0.0, reverse=True)
    for r in scored[:top_n]:
        table.add_row(r.strategy_id, _fmt(r.return_per_max_exposure), _fmt(r.sharpe_like), _fmt(r.sortino))
    return table


def best_calibrated(results: Sequence[StrategyResult], top_n: int = 10) -> Table:
    table = Table(title="BEST CALIBRATED")
    table.add_column("Strategy")
    table.add_column("Brier", justify="right")
    table.add_column("Vs Market (BSS)", justify="right")
    scored = [r for r in results if r.brier_score is not None]
    scored.sort(key=lambda r: r.brier_score if r.brier_score is not None else 1.0)
    for r in scored[:top_n]:
        table.add_row(r.strategy_id, _fmt(r.brier_score), _fmt(r.improvement_vs_market))
    return table


def most_consistent(results: Sequence[StrategyResult], top_n: int = 10) -> Table:
    table = Table(title="MOST CONSISTENT")
    table.add_column("Strategy")
    table.add_column("Return Stdev", justify="right")
    scored = [r for r in results if r.consistency_std is not None]
    scored.sort(key=lambda r: r.consistency_std if r.consistency_std is not None else float("inf"))
    for r in scored[:top_n]:
        table.add_row(r.strategy_id, _fmt(r.consistency_std, 4))
    return table


def worst_drawdown(results: Sequence[StrategyResult], top_n: int = 10) -> Table:
    table = Table(title="WORST DRAWDOWN", caption="informational -- every entry here is a costed experiment")
    table.add_column("Strategy")
    table.add_column("Max Drawdown", justify="right")
    table.add_column("Status")
    scored = [r for r in results if r.max_drawdown is not None]
    scored.sort(key=lambda r: r.max_drawdown or Decimal(0), reverse=True)
    for r in scored[:top_n]:
        table.add_row(r.strategy_id, _dd_pct(r.max_drawdown), r.status)
    return table


def worst_execution_slippage(results: Sequence[StrategyResult], top_n: int = 10) -> Table:
    table = Table(title="WORST EXECUTION SLIPPAGE")
    table.add_column("Strategy")
    table.add_column("Estimated Slippage", justify="right")
    scored = [r for r in results if r.execution_slippage is not None]
    scored.sort(key=lambda r: r.execution_slippage or Decimal(0), reverse=True)
    for r in scored[:top_n]:
        table.add_row(r.strategy_id, _money(r.execution_slippage) if r.execution_slippage is not None else "-")
    return table


def most_overfit(results: Sequence[StrategyResult], top_n: int = 10) -> Table:
    table = Table(
        title="MOST OVERFIT",
        caption="score scales with the gap over the median variant AND how many variants were tried",
    )
    table.add_column("Strategy")
    table.add_column("Overfit Score", justify="right")
    table.add_column("Variants Tried", justify="right")
    scored = [r for r in results if r.overfit_score is not None]
    scored.sort(key=lambda r: r.overfit_score or 0.0, reverse=True)
    for r in scored[:top_n]:
        table.add_row(r.strategy_id, _fmt(r.overfit_score), str(r.n_variants_tried))
    return table


def most_latency_sensitive(results: Sequence[StrategyResult], top_n: int = 10) -> Table:
    table = Table(
        title="MOST LATENCY-SENSITIVE",
        caption="breakeven latency below which edge is measured to hold; 'infeasible' flags <=500ms",
    )
    table.add_column("Strategy")
    table.add_column("Breakeven Latency", justify="right")
    table.add_column("Laptop-Feasible")
    scored = [r for r in results if r.breakeven_latency_ms is not None]
    scored.sort(key=lambda r: r.breakeven_latency_ms or float("inf"))
    for r in scored[:top_n]:
        table.add_row(
            r.strategy_id,
            f"{r.breakeven_latency_ms:.0f}ms",
            "NO" if r.laptop_infeasible else "yes",
        )
    return table


def findings_section(results: Sequence[StrategyResult]) -> Text:
    """Plain-language statements of what was learned. Dead and infeasible strategies are
    findings, framed as information gained, not as failures to apologize for."""
    lines: list[str] = []
    dead = [r for r in results if r.status.upper() == "DEAD"]
    if dead:
        lines.append(f"{len(dead)} of {len(results)} strategies tested are DEAD (bankroll floor hit).")
    infeasible = [r for r in results if r.laptop_infeasible]
    for r in infeasible:
        lat = f"{r.breakeven_latency_ms:.0f}ms" if r.breakeven_latency_ms is not None else "an untested latency"
        lines.append(f"{r.strategy_id}: edge vanishes above {lat} latency -- not viable on this hardware.")
    worse_than_market = [r for r in results if r.improvement_vs_market is not None and r.improvement_vs_market < 0]
    for r in worse_than_market:
        lines.append(
            f"{r.strategy_id}: model forecasts are worse than reading the market price "
            f"(Brier skill score {r.improvement_vs_market:.3f})."
        )
    overfit = [r for r in results if r.overfit_score is not None and r.overfit_score > 1.0]
    for r in overfit:
        lines.append(
            f"{r.strategy_id}: best-of-{r.n_variants_tried} variants beats the median by a margin "
            f"consistent with search luck, not a discovered edge (overfit score {r.overfit_score:.2f})."
        )
    for r in results:
        if r.finding:
            lines.append(f"{r.strategy_id}: {r.finding}")
    if not lines:
        lines.append("No strategies produced a notable finding this period.")
    return Text("FINDINGS\n" + "\n".join(f"- {line}" for line in lines))


@dataclass(frozen=True)
class LeagueReport:
    """Every table this module can build for one already-selected phase."""

    league: Table
    best_by_category: Table
    best_risk_adjusted: Table
    best_calibrated: Table
    most_consistent: Table
    worst_drawdown: Table
    worst_execution_slippage: Table
    most_overfit: Table
    most_latency_sensitive: Table
    findings: Text


def build_league_report(results: Sequence[StrategyResult]) -> LeagueReport:
    return LeagueReport(
        league=strategy_league(results),
        best_by_category=best_by_category(results),
        best_risk_adjusted=best_risk_adjusted(results),
        best_calibrated=best_calibrated(results),
        most_consistent=most_consistent(results),
        worst_drawdown=worst_drawdown(results),
        worst_execution_slippage=worst_execution_slippage(results),
        most_overfit=most_overfit(results),
        most_latency_sensitive=most_latency_sensitive(results),
        findings=findings_section(results),
    )


def full_league_report(results_by_phase: PhaseSeries[Sequence[StrategyResult]]) -> dict[Phase, LeagueReport]:
    """The phase-spanning entry point: one full :class:`LeagueReport` per phase present.
    There is no code path here that merges two phases' strategies into one table."""
    return {phase: build_league_report(results) for phase, results in results_by_phase.items()}


# ---------------------------------------------------------------------------
# daily_json_report
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"object of type {type(obj)!r} is not JSON serializable")


def build_daily_report(
    *,
    report_date: date,
    strategies_running: int,
    strategies_alive: int,
    strategies_dead: int,
    best_net_pnl: dict[str, Any],
    best_risk_adjusted: dict[str, Any],
    largest_drawdown: dict[str, Any],
    data_health: dict[str, Any],
    venue_health: dict[str, Any],
    phase: Phase = Phase.FORWARD_PAPER,
) -> dict[str, Any]:
    """Pure construction of the PRD's daily-report schema (no I/O). ``phase`` is an
    addition beyond the literal schema list, so a reader never has to guess whether
    "today's best" refers to backtests, paper, or live money -- see module docstring on
    sample separation."""
    return {
        "date": report_date.isoformat(),
        "phase": phase.value,
        "strategies_running": strategies_running,
        "strategies_alive": strategies_alive,
        "strategies_dead": strategies_dead,
        "best_net_pnl": best_net_pnl,
        "best_risk_adjusted": best_risk_adjusted,
        "largest_drawdown": largest_drawdown,
        "data_health": data_health,
        "venue_health": venue_health,
    }


def daily_json_report(
    *,
    report_date: date,
    strategies_running: int,
    strategies_alive: int,
    strategies_dead: int,
    best_net_pnl: dict[str, Any],
    best_risk_adjusted: dict[str, Any],
    largest_drawdown: dict[str, Any],
    data_health: dict[str, Any],
    venue_health: dict[str, Any],
    output_dir: str | Path = "data/reports",
    phase: Phase = Phase.FORWARD_PAPER,
) -> Path:
    """Build the daily report and persist it to ``<output_dir>/daily_<date>.json``."""
    report = build_daily_report(
        report_date=report_date,
        strategies_running=strategies_running,
        strategies_alive=strategies_alive,
        strategies_dead=strategies_dead,
        best_net_pnl=best_net_pnl,
        best_risk_adjusted=best_risk_adjusted,
        largest_drawdown=largest_drawdown,
        data_health=data_health,
        venue_health=venue_health,
        phase=phase,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_{report_date.isoformat()}.json"
    out_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# experiment_report / compare_experiments
# ---------------------------------------------------------------------------

# Any object exposing these attributes can be rendered -- matches
# marketlab.analytics.metrics.TradingMetrics without importing it (keeps this module
# decoupled from the exact metrics dataclass shape).
_METRIC_ROWS: list[tuple[str, str]] = [
    ("net_pnl", "Net P&L"),
    ("realized_pnl", "Realized P&L"),
    ("unrealized_pnl", "Unrealized P&L"),
    ("n_trades", "Trades"),
    ("win_rate", "Win Rate"),
    ("profit_factor", "Profit Factor"),
    ("expectancy_per_trade", "Expectancy/Trade"),
    ("max_drawdown", "Max Drawdown"),
]


def _render_metric(name: str, value: Any) -> str:
    if value is None:
        return "-"
    if name in ("net_pnl", "realized_pnl", "unrealized_pnl"):
        return _signed_money(value)
    if name == "max_drawdown":
        return _dd_pct(value)
    if name == "win_rate":
        return _pct(value)
    if isinstance(value, Decimal):
        return _fmt(value)
    return str(value)


def experiment_report(experiment_id: str, metrics: PhaseSeries[Any]) -> Table:
    """One experiment's headline metrics, one column per phase actually supplied."""
    phases = metrics.items()
    table = Table(title=f"EXPERIMENT REPORT -- {experiment_id}")
    table.add_column("Metric")
    for phase, _ in phases:
        table.add_column(phase.value, justify="right")
    for attr, label in _METRIC_ROWS:
        row = [label]
        for _, m in phases:
            row.append(_render_metric(attr, getattr(m, attr, None)))
        table.add_row(*row)
    return table


def compare_experiments(
    id_a: str,
    metrics_a: PhaseSeries[Any],
    id_b: str,
    metrics_b: PhaseSeries[Any],
) -> Table:
    """Side-by-side comparison restricted to phases BOTH experiments actually have data
    for -- comparing an experiment's backtest against another's live results would be
    exactly the kind of blended, misleading number this module refuses to produce."""
    phases_a = dict(metrics_a.items())
    phases_b = dict(metrics_b.items())
    common = [p for p in Phase if p in phases_a and p in phases_b]
    table = Table(title=f"COMPARE -- {id_a} vs {id_b}")
    table.add_column("Phase")
    table.add_column("Metric")
    table.add_column(id_a, justify="right")
    table.add_column(id_b, justify="right")
    for phase in common:
        m_a, m_b = phases_a[phase], phases_b[phase]
        for attr, label in _METRIC_ROWS:
            table.add_row(
                phase.value,
                label,
                _render_metric(attr, getattr(m_a, attr, None)),
                _render_metric(attr, getattr(m_b, attr, None)),
            )
    if not common:
        table.add_row("-", "no overlapping phase with data for both experiments", "-", "-")
    return table


# ---------------------------------------------------------------------------
# trader_report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraderReportEntry:
    snapshot_time: datetime
    category: str
    rank: int | None
    wallet: str
    username: str
    period_pnl: Decimal | None
    volume: Decimal | None
    #: e.g. DISCOVERED / TRACKING / QUALIFIED / REJECTED (marketlab.storage.state.TraderStatus)
    forward_tracking_status: str


def trader_report(entries: Sequence[TraderReportEntry]) -> Table:
    table = Table(title="TRADER REPORT")
    table.add_column("Snapshot")
    table.add_column("Category")
    table.add_column("Rank", justify="right")
    table.add_column("Wallet")
    table.add_column("Username")
    table.add_column("Period P&L", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Forward Tracking")
    for e in entries:
        table.add_row(
            e.snapshot_time.isoformat(timespec="minutes"),
            e.category,
            str(e.rank) if e.rank is not None else "-",
            e.wallet[:10] + "...",
            e.username or "-",
            _signed_money(e.period_pnl) if e.period_pnl is not None else "-",
            _money(e.volume) if e.volume is not None else "-",
            e.forward_tracking_status,
        )
    return table


# ---------------------------------------------------------------------------
# risk_report / category_report
# ---------------------------------------------------------------------------


def risk_report(results: Sequence[StrategyResult]) -> Table:
    table = Table(title="RISK REPORT")
    table.add_column("Strategy")
    table.add_column("Max Drawdown", justify="right")
    table.add_column("Sharpe-like", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Return/MaxExposure", justify="right")
    for r in sorted(results, key=lambda r: r.max_drawdown or Decimal(0), reverse=True):
        table.add_row(
            r.strategy_id,
            _dd_pct(r.max_drawdown),
            _fmt(r.sharpe_like),
            _fmt(r.sortino),
            _fmt(r.return_per_max_exposure),
        )
    return table


def category_report(results: Sequence[StrategyResult]) -> Table:
    table = Table(title="CATEGORY REPORT")
    table.add_column("Category")
    table.add_column("Strategies", justify="right")
    table.add_column("Dead", justify="right")
    table.add_column("Total Net P&L", justify="right")
    table.add_column("Avg Net P&L", justify="right")
    by_cat: dict[str, list[StrategyResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        total = sum((r.net_pnl for r in rows), Decimal(0))
        dead = sum(1 for r in rows if r.status.upper() == "DEAD")
        table.add_row(
            cat,
            str(len(rows)),
            str(dead),
            _signed_money(total),
            _signed_money(total / Decimal(len(rows))),
        )
    return table
