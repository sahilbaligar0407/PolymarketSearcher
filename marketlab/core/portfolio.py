"""Position and bankroll accounting for a single experiment sleeve.

Money is carried in ``Decimal`` throughout.  A binary contract pays exactly $1.00 if its
side resolves true and $0 otherwise, so a long YES position of ``q`` contracts bought at
average price ``p`` risks ``q*p`` and can win ``q*(1-p)`` before fees.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from marketlab.core.instruments import ONE, ZERO, Side
from marketlab.core.orders import Action, Fill

CENT = Decimal("0.01")


class SleeveStatus(StrEnum):
    ACTIVE = "active"
    DEAD = "dead"
    RETIRED = "retired"


class Position(BaseModel):
    """Net position in one side of one market.

    YES and NO are tracked separately because holding both is a real, meaningful state
    (a parity/arbitrage pair), not something to net to zero.
    """

    model_config = ConfigDict(frozen=False)

    canonical_id: str
    side: Side
    quantity: int = 0
    #: Volume-weighted average entry price.
    average_price: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    fees_paid: Decimal = ZERO
    opened_at: datetime | None = None
    last_update: datetime | None = None
    #: Total contracts ever bought, for turnover metrics.
    gross_bought: int = 0
    gross_sold: int = 0

    @property
    def cost_basis(self) -> Decimal:
        return self.average_price * Decimal(self.quantity)

    def unrealized_pnl(self, mark: Decimal) -> Decimal:
        return (mark - self.average_price) * Decimal(self.quantity)

    def market_value(self, mark: Decimal) -> Decimal:
        return mark * Decimal(self.quantity)

    def apply(self, fill: Fill) -> Decimal:
        """Fold a fill into the position. Returns realized P&L from this fill."""
        realized = ZERO
        q = fill.quantity
        if fill.action is Action.BUY:
            new_qty = self.quantity + q
            if new_qty > 0:
                total_cost = self.cost_basis + fill.price * Decimal(q)
                self.average_price = total_cost / Decimal(new_qty)
            self.quantity = new_qty
            self.gross_bought += q
            if self.opened_at is None:
                self.opened_at = fill.timestamp
        else:
            closing = min(q, self.quantity)
            if closing > 0:
                realized = (fill.price - self.average_price) * Decimal(closing)
            self.quantity -= q
            self.gross_sold += q
            if self.quantity <= 0:
                self.quantity = max(self.quantity, 0)
                self.average_price = ZERO if self.quantity == 0 else self.average_price
        self.realized_pnl += realized - fill.fee
        self.fees_paid += fill.fee
        self.last_update = fill.timestamp
        return realized

    def settle(self, won: bool) -> Decimal:
        """Resolve the position at $1 or $0. Returns realized P&L from settlement."""
        if self.quantity == 0:
            return ZERO
        payout = ONE if won else ZERO
        realized = (payout - self.average_price) * Decimal(self.quantity)
        self.realized_pnl += realized
        self.quantity = 0
        self.average_price = ZERO
        return realized


class Portfolio(BaseModel):
    """One $50 virtual sleeve (or the single live bankroll).

    Cash is reduced on buys and increased on sells and settlements.  There is no credit:
    an order that would drive cash negative is rejected upstream by the risk gateway.
    """

    model_config = ConfigDict(frozen=False)

    experiment_id: str
    strategy_id: str
    initial_capital: Decimal = Decimal("50.00")
    cash: Decimal = Decimal("50.00")
    positions: dict[str, Position] = Field(default_factory=dict)
    realized_pnl: Decimal = ZERO
    fees_paid: Decimal = ZERO
    status: SleeveStatus = SleeveStatus.ACTIVE
    created_at: datetime | None = None
    died_at: datetime | None = None
    #: Peak equity, for drawdown.
    high_water_mark: Decimal = Decimal("50.00")
    max_drawdown: Decimal = ZERO
    trade_count: int = 0
    resolved_trade_count: int = 0

    @staticmethod
    def key(canonical_id: str, side: Side) -> str:
        return f"{canonical_id}|{side.value}"

    def position(self, canonical_id: str, side: Side) -> Position:
        k = self.key(canonical_id, side)
        if k not in self.positions:
            self.positions[k] = Position(canonical_id=canonical_id, side=side)
        return self.positions[k]

    def apply_fill(self, fill: Fill) -> None:
        pos = self.position(fill.canonical_id, fill.side)
        realized = pos.apply(fill)
        notional = fill.price * Decimal(fill.quantity)
        if fill.action is Action.BUY:
            self.cash -= notional
        else:
            self.cash += notional
        self.cash -= fill.fee
        self.realized_pnl += realized - fill.fee
        self.fees_paid += fill.fee
        self.trade_count += 1

    def settle(self, canonical_id: str, winning_side: Side | None) -> Decimal:
        """Settle both sides of a market from the venue's authoritative outcome."""
        total = ZERO
        for side in (Side.YES, Side.NO):
            k = self.key(canonical_id, side)
            pos = self.positions.get(k)
            if pos is None or pos.quantity == 0:
                continue
            if winning_side is None:
                # Voided market: refund cost basis.
                self.cash += pos.cost_basis
                pos.quantity = 0
                pos.average_price = ZERO
                continue
            won = side is winning_side
            qty = pos.quantity
            realized = pos.settle(won)
            self.cash += (ONE if won else ZERO) * Decimal(qty)
            self.realized_pnl += realized
            total += realized
            self.resolved_trade_count += 1
        return total

    def equity(self, marks: dict[str, Decimal] | None = None) -> Decimal:
        """Cash plus mark-to-market value of open positions."""
        marks = marks or {}
        value = self.cash
        for k, pos in self.positions.items():
            if pos.quantity == 0:
                continue
            mark = marks.get(k)
            if mark is None:
                mark = pos.average_price
            value += pos.market_value(mark)
        return value

    def unrealized_pnl(self, marks: dict[str, Decimal] | None = None) -> Decimal:
        marks = marks or {}
        total = ZERO
        for k, pos in self.positions.items():
            if pos.quantity == 0:
                continue
            total += pos.unrealized_pnl(marks.get(k, pos.average_price))
        return total

    def exposure(self, marks: dict[str, Decimal] | None = None) -> Decimal:
        """Capital currently at risk in open positions."""
        return sum((p.cost_basis for p in self.positions.values() if p.quantity > 0), ZERO)

    def mark(self, marks: dict[str, Decimal] | None = None) -> None:
        """Update high-water mark and drawdown. Call on every equity refresh."""
        eq = self.equity(marks)
        if eq > self.high_water_mark:
            self.high_water_mark = eq
        if self.high_water_mark > ZERO:
            dd = (self.high_water_mark - eq) / self.high_water_mark
            if dd > self.max_drawdown:
                self.max_drawdown = dd

    def is_dead(self, floor: Decimal = Decimal("1.00")) -> bool:
        """A sleeve dies when it can no longer meaningfully trade. It is never deleted."""
        return self.equity() < floor
