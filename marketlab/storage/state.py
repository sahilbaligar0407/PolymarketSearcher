"""``StateStore``: the synchronous SQLite façade over everything in ``schema.py``.

Design choices, spelled out once:

* Plain synchronous :mod:`sqlite3`, guarded by a single :class:`threading.RLock`.  The
  daemon is effectively single-writer; a lock is simpler and just as fast as an async
  driver for this workload, and it means the same store can be used from a script, a
  test, or a thread pool without surprises.
* :class:`AsyncStateStore` is a thin façade that runs every blocking call through
  ``asyncio.to_thread`` so async call sites (the daemon's event loop) never block it.
* Every method takes and returns the frozen core pydantic models
  (:mod:`marketlab.core.orders`, :mod:`marketlab.core.portfolio`,
  :mod:`marketlab.core.instruments`, :mod:`marketlab.core.strategy`,
  :mod:`marketlab.core.events`) - never a redefinition of them.  A handful of storage
  domain objects that have no core equivalent (an experiment record, a tracked
  trader's profile, a cross-venue match, ...) are defined below as plain pydantic
  models owned by this package.
* Money: ``Decimal`` in, ``Decimal`` out, ``TEXT`` in SQLite (see ``schema.py``).
* Timestamps: always supplied by the caller. This module never calls ``datetime.now()``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from marketlab.core.events import Alert
from marketlab.core.instruments import ZERO, Category, NormalizedMarket, Side, Venue
from marketlab.core.orders import (
    Action,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    RejectReason,
    TimeInForce,
)
from marketlab.core.portfolio import Portfolio, Position, SleeveStatus
from marketlab.core.strategy import ProbabilityForecast
from marketlab.storage.schema import SQL_NOW, connect, run_migrations

# ---------------------------------------------------------------------------
# Storage-domain models: entities the frozen core doesn't define.
# ---------------------------------------------------------------------------


class ExperimentStatus(StrEnum):
    IDEA = "IDEA"
    BACKTESTING = "BACKTESTING"
    PAPER = "PAPER"
    QUALIFIED = "QUALIFIED"
    CHAMPION = "CHAMPION"
    DEGRADED = "DEGRADED"
    DISABLED = "DISABLED"
    LIVE_SMALL = "LIVE_SMALL"
    LIVE_PROVEN = "LIVE_PROVEN"
    DEAD = "DEAD"


class TraderStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    TRACKING = "TRACKING"
    QUALIFIED = "QUALIFIED"
    REJECTED = "REJECTED"


class VerificationStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    ONCHAIN_OR_API_CONFIRMED = "ONCHAIN_OR_API_CONFIRMED"
    DISPROVEN = "DISPROVEN"
    STALE = "STALE"


class Experiment(BaseModel):
    """Immutable identity for one strategy/parameter/data-vintage run."""

    model_config = ConfigDict(frozen=False)

    experiment_id: str
    strategy_name: str
    strategy_version: str
    git_commit: str = ""
    parameter_hash: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    market_universe: str = ""
    venue: str = ""
    data_version: str = ""
    execution_model_version: str = ""
    feature_version: str = ""
    llm_model_id: str = ""
    prompt_hash: str = ""
    start_timestamp: datetime | None = None
    starting_bankroll: Decimal = Decimal("50.00")
    status: ExperimentStatus = ExperimentStatus.IDEA
    cohort: str = ""
    created_at: datetime
    notes: str = ""


class TraderProfile(BaseModel):
    """One row of ``trader_registry``: everything known about a tracked wallet."""

    model_config = ConfigDict(frozen=False)

    wallet: str
    username: str = ""
    x_username: str = ""
    discovery_date: datetime | None = None
    rank_at_discovery: int | None = None
    category: str = ""
    day_pnl: Decimal | None = None
    week_pnl: Decimal | None = None
    month_pnl: Decimal | None = None
    all_time_pnl: Decimal | None = None
    reported_volume: Decimal | None = None
    number_of_markets: int | None = None
    number_of_observed_trades: int | None = None
    median_trade_size: Decimal | None = None
    position_concentration: Decimal | None = None
    resolved_win_rate: Decimal | None = None
    realized_pnl: Decimal | None = None
    estimated_roi: Decimal | None = None
    largest_loss: Decimal | None = None
    largest_win: Decimal | None = None
    max_observed_drawdown: Decimal | None = None
    category_specialization: str = ""
    median_holding_time: Decimal | None = None
    turnover: Decimal | None = None
    recent_performance_slope: Decimal | None = None
    status: TraderStatus = TraderStatus.DISCOVERED
    last_updated: datetime


class LeaderboardRow(BaseModel):
    """One row of a dated ``trader_leaderboard_snapshots`` entry."""

    model_config = ConfigDict(frozen=True)

    snapshot_time: datetime
    category: str = ""
    period: str = ""
    metric: str = ""
    rank: int | None = None
    wallet: str
    username: str = ""
    pnl: Decimal | None = None
    volume: Decimal | None = None


class TraderActionRecord(BaseModel):
    """An observed public trade by a tracked wallet, as persisted to ``trader_actions``."""

    model_config = ConfigDict(frozen=True)

    wallet: str
    username: str = ""
    canonical_id: str = ""
    poly_market_id: str = ""
    poly_condition_id: str = ""
    title: str = ""
    outcome: str = ""
    side: Side | None = None
    action: str = ""
    price: Decimal | None = None
    size: Decimal | None = None
    usd_size: Decimal | None = None
    category: Category = Category.OTHER
    transaction_hash: str = ""
    event_time: datetime
    first_seen_time: datetime


class MarketMatch(BaseModel):
    """A cross-venue link between two ``NormalizedMarket`` records."""

    model_config = ConfigDict(frozen=True)

    match_id: str
    canonical_id_a: str
    canonical_id_b: str
    match_confidence: Decimal = ZERO
    same_outcome_boolean: bool | None = None
    rule_diff: str = ""
    time_diff: str = ""
    resolution_source_diff: str = ""
    human_review_required: bool = False
    created_at: datetime
    validator_version: str = ""


class Settlement(BaseModel):
    """Authoritative venue resolution, as persisted (mirrors ``SettlementEvent``)."""

    model_config = ConfigDict(frozen=True)

    canonical_id: str
    venue: Venue
    winning_side: Side | None
    settlement_value: Decimal | None = None
    voided: bool = False
    settled_at: datetime
    first_seen_time: datetime


class SocialChallengeCandidate(BaseModel):
    """A claimed "$X to $Y" social-media trading challenge pending verification."""

    model_config = ConfigDict(frozen=False)

    id: int | None = None
    claim: str
    account: str = ""
    platform: str = ""
    claimed_starting_balance: Decimal | None = None
    claimed_current_balance: Decimal | None = None
    claim_date: datetime | None = None
    wallet_if_public: str = ""
    verified_by_market_data: bool = False
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    notes: str = ""


# ---------------------------------------------------------------------------
# Decimal / datetime / JSON <-> TEXT helpers (see schema.py for the convention).
# ---------------------------------------------------------------------------


def _dec(value: str | None) -> Decimal | None:
    return None if value is None else Decimal(value)


def _decs(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _dts(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"object of type {type(obj)!r} is not JSON serializable")


def _jdumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


def _jloads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


_TERMINAL_ORDER_STATUSES = (
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
)


class StateStore:
    """Synchronous SQLite-backed store for all of MarketLab's operational state."""

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.RLock()
        self._conn = connect(db_path)
        run_migrations(self._conn)

    @classmethod
    def open(cls, db_path: str | Path) -> StateStore:
        return cls(db_path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # experiments
    # ------------------------------------------------------------------

    def create_experiment(self, experiment: Experiment) -> Experiment:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, strategy_name, strategy_version, git_commit,
                    parameter_hash, parameters_json, market_universe, venue,
                    data_version, execution_model_version, feature_version,
                    llm_model_id, prompt_hash, start_timestamp, starting_bankroll,
                    status, cohort, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.strategy_name,
                    experiment.strategy_version,
                    experiment.git_commit,
                    experiment.parameter_hash,
                    _jdumps(experiment.parameters),
                    experiment.market_universe,
                    experiment.venue,
                    experiment.data_version,
                    experiment.execution_model_version,
                    experiment.feature_version,
                    experiment.llm_model_id,
                    experiment.prompt_hash,
                    _dts(experiment.start_timestamp),
                    str(experiment.starting_bankroll),
                    str(experiment.status),
                    experiment.cohort,
                    _dts(experiment.created_at),
                    experiment.notes,
                ),
            )
        return experiment

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        return Experiment(
            experiment_id=row["experiment_id"],
            strategy_name=row["strategy_name"],
            strategy_version=row["strategy_version"],
            git_commit=row["git_commit"],
            parameter_hash=row["parameter_hash"],
            parameters=_jloads(row["parameters_json"]) or {},
            market_universe=row["market_universe"],
            venue=row["venue"],
            data_version=row["data_version"],
            execution_model_version=row["execution_model_version"],
            feature_version=row["feature_version"],
            llm_model_id=row["llm_model_id"],
            prompt_hash=row["prompt_hash"],
            start_timestamp=_dt(row["start_timestamp"]),
            starting_bankroll=_dec(row["starting_bankroll"]) or ZERO,
            status=ExperimentStatus(row["status"]),
            cohort=row["cohort"],
            created_at=_dt(row["created_at"]) or datetime.fromtimestamp(0, tz=UTC),
            notes=row["notes"],
        )

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return None if row is None else self._row_to_experiment(row)

    def list_experiments(
        self, status: str | None = None, strategy: str | None = None
    ) -> list[Experiment]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status))
        if strategy is not None:
            clauses.append("strategy_name = ?")
            params.append(strategy)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM experiments {where} ORDER BY created_at", params
            ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def find_live_experiment_by_cohort(self, cohort: str) -> Experiment | None:
        """The still-running experiment for a cohort key, if one exists.

        Used on startup so a process restart RESUMES its sleeves instead of minting a
        fresh $50 bankroll for each one. DEAD and DISABLED sleeves are deliberately
        excluded: a dead sleeve stays dead and a new run is a new cohort, which is the
        rule that stops a failed experiment from being quietly rolled forward.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM experiments WHERE cohort = ? AND status NOT IN (?, ?) "
                "ORDER BY created_at DESC LIMIT 1",
                (cohort, ExperimentStatus.DEAD.value, ExperimentStatus.DISABLED.value),
            ).fetchone()
        return self._row_to_experiment(row) if row else None

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                (str(status), experiment_id),
            )

    # ------------------------------------------------------------------
    # orders / fills
    # ------------------------------------------------------------------

    def save_order(self, order: Order, experiment_id: str) -> None:
        with self._lock, self._conn:
            self._save_order_row(order, experiment_id)
            for fill in order.fills:
                self._save_fill_row(fill, experiment_id)

    def _save_order_row(self, order: Order, experiment_id: str) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO orders (
                order_id, intent_id, strategy_id, experiment_id, canonical_id, venue,
                side, action, order_type, quantity, limit_price, time_in_force,
                status, filled_quantity, average_fill_price, worst_fill_price,
                fees_paid, reject_reason, reject_detail, decision_timestamp,
                simulated_network_send_timestamp, simulated_exchange_arrival_timestamp,
                book_timestamp_used, reference_price, venue_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.intent_id,
                order.strategy_id,
                experiment_id,
                order.canonical_id,
                str(order.venue),
                str(order.side),
                str(order.action),
                str(order.order_type),
                order.quantity,
                _decs(order.limit_price),
                str(order.time_in_force),
                str(order.status),
                order.filled_quantity,
                _decs(order.average_fill_price),
                _decs(order.worst_fill_price),
                str(order.fees_paid),
                str(order.reject_reason) if order.reject_reason else None,
                order.reject_detail,
                _dts(order.decision_timestamp),
                _dts(order.simulated_network_send_timestamp),
                _dts(order.simulated_exchange_arrival_timestamp),
                _dts(order.book_timestamp_used),
                _decs(order.reference_price),
                order.venue_order_id,
            ),
        )

    def update_order(self, order: Order) -> None:
        self.save_order(order, order.experiment_id)

    def _save_fill_row(self, fill: Fill, experiment_id: str) -> None:
        level_breakdown = [[str(p), q] for p, q in fill.level_breakdown]
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fills (
                fill_id, order_id, experiment_id, canonical_id, venue, side, action,
                price, quantity, fee, timestamp, is_maker, book_timestamp_used,
                level_breakdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                experiment_id,
                fill.canonical_id,
                str(fill.venue),
                str(fill.side),
                str(fill.action),
                str(fill.price),
                fill.quantity,
                str(fill.fee),
                _dts(fill.timestamp),
                int(fill.is_maker),
                _dts(fill.book_timestamp_used),
                _jdumps(level_breakdown),
            ),
        )

    def save_fill(self, fill: Fill, experiment_id: str) -> None:
        with self._lock, self._conn:
            self._save_fill_row(fill, experiment_id)

    @staticmethod
    def _row_to_fill(row: sqlite3.Row) -> Fill:
        raw_levels = _jloads(row["level_breakdown"]) or []
        level_breakdown = tuple((Decimal(p), int(q)) for p, q in raw_levels)
        return Fill(
            fill_id=row["fill_id"],
            order_id=row["order_id"],
            canonical_id=row["canonical_id"],
            venue=Venue(row["venue"]),
            side=Side(row["side"]),
            action=Action(row["action"]),
            price=_dec(row["price"]) or ZERO,
            quantity=row["quantity"],
            fee=_dec(row["fee"]) or ZERO,
            timestamp=_dt(row["timestamp"]) or datetime.fromtimestamp(0, tz=UTC),
            is_maker=bool(row["is_maker"]),
            book_timestamp_used=_dt(row["book_timestamp_used"]),
            level_breakdown=level_breakdown,
        )

    def _fills_for_order(self, order_id: str) -> tuple[Fill, ...]:
        rows = self._conn.execute(
            "SELECT * FROM fills WHERE order_id = ? ORDER BY timestamp", (order_id,)
        ).fetchall()
        return tuple(self._row_to_fill(r) for r in rows)

    def _row_to_order(self, row: sqlite3.Row, fills: tuple[Fill, ...]) -> Order:
        return Order(
            order_id=row["order_id"],
            intent_id=row["intent_id"],
            strategy_id=row["strategy_id"],
            experiment_id=row["experiment_id"],
            canonical_id=row["canonical_id"],
            venue=Venue(row["venue"]),
            side=Side(row["side"]),
            action=Action(row["action"]),
            order_type=OrderType(row["order_type"]),
            quantity=row["quantity"],
            limit_price=_dec(row["limit_price"]),
            time_in_force=TimeInForce(row["time_in_force"]),
            status=OrderStatus(row["status"]),
            filled_quantity=row["filled_quantity"],
            average_fill_price=_dec(row["average_fill_price"]),
            worst_fill_price=_dec(row["worst_fill_price"]),
            fees_paid=_dec(row["fees_paid"]) or ZERO,
            reject_reason=RejectReason(row["reject_reason"]) if row["reject_reason"] else None,
            reject_detail=row["reject_detail"],
            decision_timestamp=_dt(row["decision_timestamp"]) or datetime.fromtimestamp(0, tz=UTC),
            simulated_network_send_timestamp=_dt(row["simulated_network_send_timestamp"]),
            simulated_exchange_arrival_timestamp=_dt(
                row["simulated_exchange_arrival_timestamp"]
            ),
            book_timestamp_used=_dt(row["book_timestamp_used"]),
            reference_price=_dec(row["reference_price"]),
            venue_order_id=row["venue_order_id"],
            fills=fills,
        )

    def get_order(self, order_id: str) -> Order | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if row is None:
                return None
            fills = self._fills_for_order(order_id)
        return self._row_to_order(row, fills)

    def open_orders(self, experiment_id: str | None = None) -> list[Order]:
        terminal = ", ".join("?" for _ in _TERMINAL_ORDER_STATUSES)
        params: list[Any] = [str(s) for s in _TERMINAL_ORDER_STATUSES]
        clause = f"status NOT IN ({terminal})"
        if experiment_id is not None:
            clause += " AND experiment_id = ?"
            params.append(experiment_id)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM orders WHERE {clause} ORDER BY decision_timestamp", params
            ).fetchall()
            return [self._row_to_order(r, self._fills_for_order(r["order_id"])) for r in rows]

    def orders_for_experiment(self, experiment_id: str) -> list[Order]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM orders WHERE experiment_id = ? ORDER BY decision_timestamp",
                (experiment_id,),
            ).fetchall()
            return [self._row_to_order(r, self._fills_for_order(r["order_id"])) for r in rows]

    def fills_for_experiment(self, experiment_id: str) -> list[Fill]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fills WHERE experiment_id = ? ORDER BY timestamp",
                (experiment_id,),
            ).fetchall()
        return [self._row_to_fill(r) for r in rows]

    # ------------------------------------------------------------------
    # portfolio (crash-recovery guarantee)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_portfolio_timestamp(portfolio: Portfolio) -> datetime:
        updates = [p.last_update for p in portfolio.positions.values() if p.last_update]
        if updates:
            return max(updates)
        if portfolio.created_at is not None:
            return portfolio.created_at
        if portfolio.died_at is not None:
            return portfolio.died_at
        raise ValueError(
            "cannot infer a timestamp for this portfolio snapshot; pass as_of explicitly"
        )

    def save_portfolio(self, portfolio: Portfolio, as_of: datetime | None = None) -> None:
        ts = as_of or self._infer_portfolio_timestamp(portfolio)
        with self._lock, self._conn:
            for pos in portfolio.positions.values():
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (
                        experiment_id, canonical_id, side, quantity, average_price,
                        realized_pnl, fees_paid, opened_at, last_update, gross_bought,
                        gross_sold
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        portfolio.experiment_id,
                        pos.canonical_id,
                        str(pos.side),
                        pos.quantity,
                        str(pos.average_price),
                        str(pos.realized_pnl),
                        str(pos.fees_paid),
                        _dts(pos.opened_at),
                        _dts(pos.last_update),
                        pos.gross_bought,
                        pos.gross_sold,
                    ),
                )
            self._conn.execute(
                """
                INSERT INTO balances (
                    experiment_id, strategy_id, timestamp, cash, equity, realized_pnl,
                    unrealized_pnl, exposure, high_water_mark, max_drawdown,
                    trade_count, resolved_trade_count, fees_paid, initial_capital,
                    status, created_at, died_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio.experiment_id,
                    portfolio.strategy_id,
                    _dts(ts),
                    str(portfolio.cash),
                    str(portfolio.equity()),
                    str(portfolio.realized_pnl),
                    str(portfolio.unrealized_pnl()),
                    str(portfolio.exposure()),
                    str(portfolio.high_water_mark),
                    str(portfolio.max_drawdown),
                    portfolio.trade_count,
                    portfolio.resolved_trade_count,
                    str(portfolio.fees_paid),
                    str(portfolio.initial_capital),
                    str(portfolio.status),
                    _dts(portfolio.created_at),
                    _dts(portfolio.died_at),
                ),
            )

    def load_portfolio(self, experiment_id: str) -> Portfolio | None:
        with self._lock:
            balance_row = self._conn.execute(
                "SELECT * FROM balances WHERE experiment_id = ? ORDER BY id DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if balance_row is None:
                return None
            position_rows = self._conn.execute(
                "SELECT * FROM positions WHERE experiment_id = ?", (experiment_id,)
            ).fetchall()

        positions: dict[str, Position] = {}
        for prow in position_rows:
            side = Side(prow["side"])
            pos = Position(
                canonical_id=prow["canonical_id"],
                side=side,
                quantity=prow["quantity"],
                average_price=_dec(prow["average_price"]) or ZERO,
                realized_pnl=_dec(prow["realized_pnl"]) or ZERO,
                fees_paid=_dec(prow["fees_paid"]) or ZERO,
                opened_at=_dt(prow["opened_at"]),
                last_update=_dt(prow["last_update"]),
                gross_bought=prow["gross_bought"],
                gross_sold=prow["gross_sold"],
            )
            positions[Portfolio.key(pos.canonical_id, side)] = pos

        return Portfolio(
            experiment_id=experiment_id,
            strategy_id=balance_row["strategy_id"],
            initial_capital=_dec(balance_row["initial_capital"]) or ZERO,
            cash=_dec(balance_row["cash"]) or ZERO,
            positions=positions,
            realized_pnl=_dec(balance_row["realized_pnl"]) or ZERO,
            fees_paid=_dec(balance_row["fees_paid"]) or ZERO,
            status=SleeveStatus(balance_row["status"]),
            created_at=_dt(balance_row["created_at"]),
            died_at=_dt(balance_row["died_at"]),
            high_water_mark=_dec(balance_row["high_water_mark"]) or ZERO,
            max_drawdown=_dec(balance_row["max_drawdown"]) or ZERO,
            trade_count=balance_row["trade_count"],
            resolved_trade_count=balance_row["resolved_trade_count"],
        )

    # ------------------------------------------------------------------
    # strategy internal state (crash recovery for strategy-private data)
    # ------------------------------------------------------------------

    def open_position_market_ids(self) -> list[str]:
        """Every canonical_id any sleeve still holds a non-zero position in.

        The settlement poller works from this rather than from the market registry: the
        registry only ever holds *open* markets, so a contract silently disappears from
        it the moment it closes - which is precisely when we need to go and find out how
        it resolved. Positions are what we actually need settled.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT canonical_id FROM positions WHERE CAST(quantity AS INTEGER) != 0"
            ).fetchall()
        return [r[0] for r in rows]

    def settled_market_ids(self) -> set[str]:
        """Markets already recorded as settled, so the poller does not re-fetch them."""
        with self._lock:
            rows = self._conn.execute("SELECT canonical_id FROM settlements").fetchall()
        return {r[0] for r in rows}

    def save_strategy_state(self, experiment_id: str, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO strategy_state (
                    experiment_id, key, value_json, updated_at
                ) VALUES (?, ?, ?, {SQL_NOW})
                """,
                (experiment_id, key, _jdumps(value)),
            )

    def load_strategy_state(self, experiment_id: str) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value_json FROM strategy_state WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
        return {r["key"]: _jloads(r["value_json"]) for r in rows}

    # ------------------------------------------------------------------
    # copy-trading intelligence
    # ------------------------------------------------------------------

    def upsert_trader(self, trader: TraderProfile) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO trader_registry (
                    wallet, username, x_username, discovery_date, rank_at_discovery,
                    category, day_pnl, week_pnl, month_pnl, all_time_pnl,
                    reported_volume, number_of_markets, number_of_observed_trades,
                    median_trade_size, position_concentration, resolved_win_rate,
                    realized_pnl, estimated_roi, largest_loss, largest_win,
                    max_observed_drawdown, category_specialization,
                    median_holding_time, turnover, recent_performance_slope, status,
                    last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trader.wallet,
                    trader.username,
                    trader.x_username,
                    _dts(trader.discovery_date),
                    trader.rank_at_discovery,
                    trader.category,
                    _decs(trader.day_pnl),
                    _decs(trader.week_pnl),
                    _decs(trader.month_pnl),
                    _decs(trader.all_time_pnl),
                    _decs(trader.reported_volume),
                    trader.number_of_markets,
                    trader.number_of_observed_trades,
                    _decs(trader.median_trade_size),
                    _decs(trader.position_concentration),
                    _decs(trader.resolved_win_rate),
                    _decs(trader.realized_pnl),
                    _decs(trader.estimated_roi),
                    _decs(trader.largest_loss),
                    _decs(trader.largest_win),
                    _decs(trader.max_observed_drawdown),
                    trader.category_specialization,
                    _decs(trader.median_holding_time),
                    _decs(trader.turnover),
                    _decs(trader.recent_performance_slope),
                    str(trader.status),
                    _dts(trader.last_updated),
                ),
            )

    @staticmethod
    def _row_to_trader(row: sqlite3.Row) -> TraderProfile:
        return TraderProfile(
            wallet=row["wallet"],
            username=row["username"],
            x_username=row["x_username"],
            discovery_date=_dt(row["discovery_date"]),
            rank_at_discovery=row["rank_at_discovery"],
            category=row["category"],
            day_pnl=_dec(row["day_pnl"]),
            week_pnl=_dec(row["week_pnl"]),
            month_pnl=_dec(row["month_pnl"]),
            all_time_pnl=_dec(row["all_time_pnl"]),
            reported_volume=_dec(row["reported_volume"]),
            number_of_markets=row["number_of_markets"],
            number_of_observed_trades=row["number_of_observed_trades"],
            median_trade_size=_dec(row["median_trade_size"]),
            position_concentration=_dec(row["position_concentration"]),
            resolved_win_rate=_dec(row["resolved_win_rate"]),
            realized_pnl=_dec(row["realized_pnl"]),
            estimated_roi=_dec(row["estimated_roi"]),
            largest_loss=_dec(row["largest_loss"]),
            largest_win=_dec(row["largest_win"]),
            max_observed_drawdown=_dec(row["max_observed_drawdown"]),
            category_specialization=row["category_specialization"],
            median_holding_time=_dec(row["median_holding_time"]),
            turnover=_dec(row["turnover"]),
            recent_performance_slope=_dec(row["recent_performance_slope"]),
            status=TraderStatus(row["status"]),
            last_updated=_dt(row["last_updated"]) or datetime.fromtimestamp(0, tz=UTC),
        )

    def get_trader(self, wallet: str) -> TraderProfile | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trader_registry WHERE wallet = ?", (wallet,)
            ).fetchone()
        return None if row is None else self._row_to_trader(row)

    def list_traders(self, status: str | None = None) -> list[TraderProfile]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM trader_registry ORDER BY wallet"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM trader_registry WHERE status = ? ORDER BY wallet",
                    (str(status),),
                ).fetchall()
        return [self._row_to_trader(r) for r in rows]

    def save_leaderboard_snapshot(self, rows: Iterable[LeaderboardRow]) -> None:
        payload = [
            (
                _dts(r.snapshot_time),
                r.category,
                r.period,
                r.metric,
                r.rank,
                r.wallet,
                r.username,
                _decs(r.pnl),
                _decs(r.volume),
            )
            for r in rows
        ]
        if not payload:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO trader_leaderboard_snapshots (
                    snapshot_time, category, period, metric, rank, wallet, username,
                    pnl, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def save_trader_action(self, action: TraderActionRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO trader_actions (
                    wallet, username, canonical_id, poly_market_id, poly_condition_id,
                    title, outcome, side, action, price, size, usd_size, category,
                    transaction_hash, event_time, first_seen_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.wallet,
                    action.username,
                    action.canonical_id,
                    action.poly_market_id,
                    action.poly_condition_id,
                    action.title,
                    action.outcome,
                    str(action.side) if action.side else None,
                    action.action,
                    _decs(action.price),
                    _decs(action.size),
                    _decs(action.usd_size),
                    str(action.category),
                    action.transaction_hash,
                    _dts(action.event_time),
                    _dts(action.first_seen_time),
                ),
            )

    @staticmethod
    def _row_to_trader_action(row: sqlite3.Row) -> TraderActionRecord:
        return TraderActionRecord(
            wallet=row["wallet"],
            username=row["username"],
            canonical_id=row["canonical_id"],
            poly_market_id=row["poly_market_id"],
            poly_condition_id=row["poly_condition_id"],
            title=row["title"],
            outcome=row["outcome"],
            side=Side(row["side"]) if row["side"] else None,
            action=row["action"],
            price=_dec(row["price"]),
            size=_dec(row["size"]),
            usd_size=_dec(row["usd_size"]),
            category=Category(row["category"]) if row["category"] else Category.OTHER,
            transaction_hash=row["transaction_hash"],
            event_time=_dt(row["event_time"]) or datetime.fromtimestamp(0, tz=UTC),
            first_seen_time=_dt(row["first_seen_time"]) or datetime.fromtimestamp(0, tz=UTC),
        )

    def trader_actions_since(self, ts: datetime) -> list[TraderActionRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trader_actions WHERE first_seen_time >= ? ORDER BY first_seen_time",
                (_dts(ts),),
            ).fetchall()
        return [self._row_to_trader_action(r) for r in rows]

    # ------------------------------------------------------------------
    # market registry
    # ------------------------------------------------------------------

    def _market_params(self, market: NormalizedMarket) -> tuple[Any, ...]:
        return (
            market.canonical_id,
            str(market.venue),
            market.venue_market_id,
            market.event_id,
            market.title,
            str(market.category),
            str(market.status),
            _dts(market.open_time),
            _dts(market.close_time),
            str(market.tick_size),
            market.min_order,
            _jdumps(market.fees.model_dump()),
            market.model_dump_json(),
        )

    def upsert_market(self, market: NormalizedMarket) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO market_registry (
                    canonical_id, venue, venue_market_id, event_id, title, category,
                    status, open_time, close_time, tick_size, min_order, fees_json,
                    last_seen, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {SQL_NOW}, ?)
                """,
                self._market_params(market),
            )

    def bulk_upsert_markets(self, markets: Iterable[NormalizedMarket]) -> None:
        payload = [self._market_params(m) for m in markets]
        if not payload:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                f"""
                INSERT OR REPLACE INTO market_registry (
                    canonical_id, venue, venue_market_id, event_id, title, category,
                    status, open_time, close_time, tick_size, min_order, fees_json,
                    last_seen, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {SQL_NOW}, ?)
                """,
                payload,
            )

    def get_market(self, canonical_id: str) -> NormalizedMarket | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT raw_json FROM market_registry WHERE canonical_id = ?", (canonical_id,)
            ).fetchone()
        return None if row is None else NormalizedMarket.model_validate_json(row["raw_json"])

    def list_markets(
        self,
        venue: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        category: str | None = None,
    ) -> list[NormalizedMarket]:
        clauses: list[str] = []
        params: list[Any] = []
        if venue is not None:
            clauses.append("venue = ?")
            params.append(str(venue))
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status))
        if category is not None:
            clauses.append("category = ?")
            params.append(str(category))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT raw_json FROM market_registry {where} ORDER BY canonical_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [NormalizedMarket.model_validate_json(r["raw_json"]) for r in rows]

    # ------------------------------------------------------------------
    # cross-venue matches
    # ------------------------------------------------------------------

    def save_match(self, match: MarketMatch) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO market_matches (
                    match_id, canonical_id_a, canonical_id_b, match_confidence,
                    same_outcome_boolean, rule_diff, time_diff,
                    resolution_source_diff, human_review_required, created_at,
                    validator_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match.match_id,
                    match.canonical_id_a,
                    match.canonical_id_b,
                    str(match.match_confidence),
                    None
                    if match.same_outcome_boolean is None
                    else int(match.same_outcome_boolean),
                    match.rule_diff,
                    match.time_diff,
                    match.resolution_source_diff,
                    int(match.human_review_required),
                    _dts(match.created_at),
                    match.validator_version,
                ),
            )

    @staticmethod
    def _row_to_match(row: sqlite3.Row) -> MarketMatch:
        return MarketMatch(
            match_id=row["match_id"],
            canonical_id_a=row["canonical_id_a"],
            canonical_id_b=row["canonical_id_b"],
            match_confidence=_dec(row["match_confidence"]) or ZERO,
            same_outcome_boolean=(
                None
                if row["same_outcome_boolean"] is None
                else bool(row["same_outcome_boolean"])
            ),
            rule_diff=row["rule_diff"],
            time_diff=row["time_diff"],
            resolution_source_diff=row["resolution_source_diff"],
            human_review_required=bool(row["human_review_required"]),
            created_at=_dt(row["created_at"]) or datetime.fromtimestamp(0, tz=UTC),
            validator_version=row["validator_version"],
        )

    def get_matches(self, canonical_id: str) -> list[MarketMatch]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM market_matches WHERE canonical_id_a = ? OR canonical_id_b = ?",
                (canonical_id, canonical_id),
            ).fetchall()
        return [self._row_to_match(r) for r in rows]

    def evaluated_pairs(self, validator_version: str) -> set[tuple[str, str]]:
        """(canonical_id_a, canonical_id_b) for every pair one validator already judged."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT canonical_id_a, canonical_id_b FROM market_matches WHERE validator_version = ?",
                (validator_version,),
            ).fetchall()
        return {(r[0], r[1]) for r in rows}

    def approved_matches(self) -> list[MarketMatch]:
        """Matches that did not require human review, i.e. auto-approved."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM market_matches WHERE human_review_required = 0"
            ).fetchall()
        return [self._row_to_match(r) for r in rows]

    # ------------------------------------------------------------------
    # forecasts
    # ------------------------------------------------------------------

    def save_forecast(self, forecast: ProbabilityForecast) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO forecasts (
                    experiment_id, strategy_id, canonical_id, as_of, p_yes,
                    confidence, market_probability, abstain, evidence_ids_json,
                    rationale, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    forecast.experiment_id,
                    forecast.strategy_id,
                    forecast.canonical_id,
                    _dts(forecast.as_of),
                    str(forecast.p_yes),
                    str(forecast.confidence),
                    _decs(forecast.market_probability),
                    int(forecast.abstain),
                    _jdumps(list(forecast.evidence_ids)),
                    forecast.rationale,
                    _jdumps(forecast.features),
                ),
            )

    _TRADER_SCORE_COLUMNS = (
        "wallet", "username", "computed_at", "resolved_positions", "realized_pnl", "total_staked",
        "roi", "win_rate", "profit_factor", "max_drawdown", "sharpe_like", "trades_per_day",
        "avg_entry_price", "favorite_share", "largest_win_share", "recent_roi", "recent_positions",
        "top_category", "category_share", "score", "status", "reasons",
    )

    def save_trader_score(self, row: Mapping[str, Any]) -> None:
        """Replace one wallet's TraderScore (see migrations/003_trader_scores.sql)."""
        cols = self._TRADER_SCORE_COLUMNS
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO trader_scores ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                tuple(row.get(c) for c in cols),
            )

    def leaderboard_wallets(self, limit: int = 300, days: int = 2) -> list[tuple[str, str, int]]:
        """(wallet, username, best rank) from recent leaderboard snapshots, best rank first."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT wallet, MAX(username), MIN(rank) AS best FROM trader_leaderboard_snapshots
                   WHERE snapshot_time >= datetime((SELECT MAX(snapshot_time) FROM trader_leaderboard_snapshots), ?)
                   GROUP BY wallet ORDER BY best LIMIT ?""",
                (f"-{int(days)} days", limit),
            ).fetchall()
        return [(r[0], r[1] or "", int(r[2] or 0)) for r in rows]

    def trader_scores(self, status: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        sql = "SELECT * FROM trader_scores"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY score DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [dict(r) for r in rows]

    def save_decision(self, intent: Any, order: Order, experiment_id: str) -> None:
        """Persist the why behind a filled order (see migrations/002_decisions.sql)."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO decisions (
                    intent_id, order_id, experiment_id, strategy_id, canonical_id, side,
                    action, quantity, filled_quantity, average_fill_price, fees_paid,
                    model_probability, expected_edge, rationale, features_json, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent.intent_id,
                    order.order_id,
                    experiment_id,
                    intent.strategy_id,
                    intent.canonical_id,
                    str(intent.side),
                    str(intent.action),
                    intent.quantity,
                    order.filled_quantity,
                    _decs(order.average_fill_price),
                    str(order.fees_paid),
                    _decs(intent.model_probability),
                    _decs(intent.expected_edge),
                    intent.rationale,
                    json.dumps(intent.features or {}, default=str),
                    _dts(intent.decision_time),
                ),
            )

    def save_forecasts_bulk(self, forecasts: Sequence[ProbabilityForecast]) -> int:
        """Insert many forecasts in ONE transaction.

        ``save_forecast`` commits per row. The tournament emits a forecast per subscribed
        market per sleeve per tick - tens of thousands per minute - and at roughly a
        millisecond per commit that blocked the event loop for minutes at a time, which
        looked like a hung daemon. Batching turns it into a single commit.
        """
        rows = [
            (
                f.experiment_id,
                f.strategy_id,
                f.canonical_id,
                _dts(f.as_of),
                str(f.p_yes),
                str(f.confidence),
                _decs(f.market_probability),
                int(f.abstain),
                _jdumps(list(f.evidence_ids)),
                f.rationale,
                _jdumps(f.features),
            )
            for f in forecasts
        ]
        if not rows:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO forecasts (
                    experiment_id, strategy_id, canonical_id, as_of, p_yes,
                    confidence, market_probability, abstain, evidence_ids_json,
                    rationale, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    @staticmethod
    def _row_to_forecast(row: sqlite3.Row) -> ProbabilityForecast:
        return ProbabilityForecast(
            strategy_id=row["strategy_id"],
            experiment_id=row["experiment_id"],
            canonical_id=row["canonical_id"],
            as_of=_dt(row["as_of"]) or datetime.fromtimestamp(0, tz=UTC),
            p_yes=_dec(row["p_yes"]) or ZERO,
            confidence=_dec(row["confidence"]) or Decimal("0.5"),
            market_probability=_dec(row["market_probability"]),
            abstain=bool(row["abstain"]),
            evidence_ids=tuple(_jloads(row["evidence_ids_json"]) or ()),
            features=_jloads(row["features_json"]) or {},
            rationale=row["rationale"],
        )

    def forecasts_for_experiment(self, experiment_id: str) -> list[ProbabilityForecast]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM forecasts WHERE experiment_id = ? ORDER BY as_of",
                (experiment_id,),
            ).fetchall()
        return [self._row_to_forecast(r) for r in rows]

    # ------------------------------------------------------------------
    # settlements
    # ------------------------------------------------------------------

    def save_settlement(self, settlement: Settlement) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO settlements (
                    canonical_id, venue, winning_side, settlement_value, voided,
                    settled_at, first_seen_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    settlement.canonical_id,
                    str(settlement.venue),
                    str(settlement.winning_side) if settlement.winning_side else None,
                    _decs(settlement.settlement_value),
                    int(settlement.voided),
                    _dts(settlement.settled_at),
                    _dts(settlement.first_seen_time),
                ),
            )

    @staticmethod
    def _row_to_settlement(row: sqlite3.Row) -> Settlement:
        return Settlement(
            canonical_id=row["canonical_id"],
            venue=Venue(row["venue"]),
            winning_side=Side(row["winning_side"]) if row["winning_side"] else None,
            settlement_value=_dec(row["settlement_value"]),
            voided=bool(row["voided"]),
            settled_at=_dt(row["settled_at"]) or datetime.fromtimestamp(0, tz=UTC),
            first_seen_time=_dt(row["first_seen_time"]) or datetime.fromtimestamp(0, tz=UTC),
        )

    def get_settlement(self, canonical_id: str) -> Settlement | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM settlements WHERE canonical_id = ?", (canonical_id,)
            ).fetchone()
        return None if row is None else self._row_to_settlement(row)

    def unsettled_markets(self) -> list[str]:
        """Canonical ids that closed but have no settlement recorded yet."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.canonical_id FROM market_registry m
                LEFT JOIN settlements s ON s.canonical_id = m.canonical_id
                WHERE s.canonical_id IS NULL AND m.status = 'closed'
                """
            ).fetchall()
        return [r["canonical_id"] for r in rows]

    # ------------------------------------------------------------------
    # alerts
    # ------------------------------------------------------------------

    def save_alert(self, alert: Alert) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO alerts (timestamp, severity, component, message, detail_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    _dts(alert.timestamp),
                    alert.severity,
                    alert.component,
                    alert.message,
                    _jdumps(alert.detail),
                ),
            )

    def recent_alerts(self, n: int = 50) -> list[Alert]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [
            Alert(
                timestamp=_dt(r["timestamp"]) or datetime.fromtimestamp(0, tz=UTC),
                severity=r["severity"],
                component=r["component"],
                message=r["message"],
                detail=_jloads(r["detail_json"]) or {},
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def set_checkpoint(self, component: str, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO service_checkpoints (
                    component, checkpoint_key, checkpoint_value, updated_at
                ) VALUES (?, ?, ?, {SQL_NOW})
                """,
                (component, key, value),
            )

    def get_checkpoint(self, component: str, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT checkpoint_value FROM service_checkpoints "
                "WHERE component = ? AND checkpoint_key = ?",
                (component, key),
            ).fetchone()
        return None if row is None else row["checkpoint_value"]

    # ------------------------------------------------------------------
    # social challenge candidates (bonus helpers; not on the required list)
    # ------------------------------------------------------------------

    def save_social_candidate(self, candidate: SocialChallengeCandidate) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO social_challenge_candidates (
                    claim, account, platform, claimed_starting_balance,
                    claimed_current_balance, claim_date, wallet_if_public,
                    verified_by_market_data, verification_status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.claim,
                    candidate.account,
                    candidate.platform,
                    _decs(candidate.claimed_starting_balance),
                    _decs(candidate.claimed_current_balance),
                    _dts(candidate.claim_date),
                    candidate.wallet_if_public,
                    int(candidate.verified_by_market_data),
                    str(candidate.verification_status),
                    candidate.notes,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_social_candidates(self, status: str | None = None) -> list[SocialChallengeCandidate]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM social_challenge_candidates ORDER BY id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM social_challenge_candidates "
                    "WHERE verification_status = ? ORDER BY id",
                    (str(status),),
                ).fetchall()
        return [
            SocialChallengeCandidate(
                id=r["id"],
                claim=r["claim"],
                account=r["account"],
                platform=r["platform"],
                claimed_starting_balance=_dec(r["claimed_starting_balance"]),
                claimed_current_balance=_dec(r["claimed_current_balance"]),
                claim_date=_dt(r["claim_date"]),
                wallet_if_public=r["wallet_if_public"],
                verified_by_market_data=bool(r["verified_by_market_data"]),
                verification_status=VerificationStatus(r["verification_status"]),
                notes=r["notes"],
            )
            for r in rows
        ]


class AsyncStateStore:
    """Async facade over :class:`StateStore` for the daemon's event loop.

    Every method call is proxied to the wrapped synchronous store and executed in a
    worker thread via ``asyncio.to_thread``; ``StateStore``'s internal lock keeps that
    safe even when several coroutines call in concurrently.
    """

    def __init__(self, store: StateStore) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr

        async def _call(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _call

    async def aclose(self) -> None:
        await asyncio.to_thread(self._store.close)
