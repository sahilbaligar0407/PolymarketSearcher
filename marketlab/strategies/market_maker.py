"""Avellaneda-Stoikov, adapted to a bounded [0, 1] binary claim (evidence class B).

The classic Avellaneda-Stoikov model assumes an unbounded, diffusive mid-price. A Kalshi
contract's price is a probability: it is bounded in ``[0, 1]``, its variance collapses as
the market approaches resolution (a coin that's 99% certain doesn't move much), and the
maker fee schedule (``ceil(0.07 * C * P * (1-P))``) is *itself* shaped like an inverted
parabola that **peaks exactly at P = 0.50** - precisely where a spread-maximizing quoter
most wants to sit. Every departure from the textbook model below is one of those three
facts asserting itself.

Params (see ``configs/strategies.yaml``):
    base_spread        in {0.02, 0.03, 0.05}
    inventory_penalty  in {0.5, 1.0, 2.0}
    mode               in {two_sided, one_sided, inventory_neutral}
    news_pause         in {true}

Plus filters not swept by the default grid: ``high_liquidity_only``, ``low_volatility_only``.

    fair_probability    = market midpoint (a model estimate could override this; not wired
                           up here since no probability model is owned by this team)
    reservation_price   = fair_probability - inventory_skew(inventory, time_to_close, vol)
    desired_half_spread = base_spread/2 + volatility_adjustment + adverse_selection_adjustment

**Cancel mechanism**: this strategy exposes a ``cancel_requests() -> list[str]`` method,
drained exactly like ``generate_intents()``/``drain_forecasts()``. It tracks its own
resting order ids via ``on_order_update`` (the one channel through which a strategy legally
learns an order id - see ``core/strategy.py``) and appends to the cancel list whenever
those orders should be pulled. This was chosen over re-emitting intents with
``replaces_order_id`` because a pure "pull my quotes" event (stale data, a status change, a
news pause) has no replacement order to describe - forcing one would mean inventing a fake
resize-to-zero intent just to carry a cancellation, which is a worse fit for a field
(``OrderIntent.quantity``) that is validated to be strictly positive.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, MarketStatusEvent, NewsEvent
from marketlab.core.instruments import ONE, MarketStatus, NormalizedMarket, Side
from marketlab.core.orders import Action, Order, OrderStatus, OrderType
from marketlab.core.strategy import StrategyContext
from marketlab.signals.rolling import RollingWindow
from marketlab.strategies.base import BaseStrategy

#: How far (in probability) the reservation price shifts per unit of
#: ``inventory * inventory_penalty * volatility * time_factor``.
INVENTORY_SKEW_SCALE = Decimal("1.0")
#: Horizon over which both the inventory skew and the late-close spread widening ramp -
#: expressed in seconds, roughly "how long before resolution does time_to_close start
#: mattering at all".
DEFAULT_SKEW_HORIZON_SECONDS = 3600.0
#: Stop quoting altogether once fewer than this many seconds remain - no amount of
#: widening is worth it right at the resolution boundary.
DEFAULT_STOP_QUOTING_SECONDS = 30.0
#: Start widening for informational/event risk once inside this many seconds of close.
DEFAULT_CLOSE_BUFFER_SECONDS = 300.0
#: Maximum extra half-spread added purely for being close to expiry.
DEFAULT_LATE_WIDEN_MAX = Decimal("0.05")
#: A fair-value move at least this large since the last quote update forces a cancel - the
#: resting quotes were priced against stale information.
DEFAULT_ADVERSE_MOVE_THRESHOLD = Decimal("0.03")
#: How long a high-impact news event pauses ALL quoting for, when news_pause is enabled.
DEFAULT_NEWS_PAUSE_SECONDS = 60.0
DEFAULT_MAX_INVENTORY = 10


class AvellanedaStoikovBinaryStrategy(BaseStrategy):
    """Two-sided (or filtered one-sided) quoting around a bounded-claim reservation price."""

    name = "market_maker"
    version = "1.0.0"
    evidence_class = "B"

    def __init__(
        self,
        strategy_id: str,
        experiment_id: str,
        ctx: StrategyContext,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self._inventory: dict[str, int] = {}
        self._resting_orders: dict[str, set[str]] = {}
        self._cancel_queue: list[str] = []
        self._news_pause_started_at: Any = None  # datetime | None

    # ------------------------------------------------------------------ broker feedback

    def on_fill(self, fill: Any) -> None:
        qty = fill.quantity
        # Inventory is carried in YES-equivalents: long YES or short NO both count +qty;
        # long NO or short YES both count -qty.
        sign = 1 if (fill.side is Side.YES) == (fill.action is Action.BUY) else -1
        self._inventory[fill.canonical_id] = self._inventory.get(fill.canonical_id, 0) + sign * qty

    def on_order_update(self, order: Order) -> None:
        ids = self._resting_orders.setdefault(order.canonical_id, set())
        if order.is_terminal:
            ids.discard(order.order_id)
        elif order.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
            ids.add(order.order_id)

    def cancel_requests(self) -> list[str]:
        """Drain pending cancels - the runner should call this alongside ``generate_intents``."""
        out, self._cancel_queue = self._cancel_queue, []
        return out

    def _cancel_all(self, canonical_id: str) -> None:
        for order_id in self._resting_orders.get(canonical_id, ()):
            self._cancel_queue.append(order_id)
        self._resting_orders[canonical_id] = set()

    # ------------------------------------------------------------------ event handlers

    def on_market_status(self, event: MarketStatusEvent) -> None:
        if event.status is not MarketStatus.OPEN:
            self._cancel_all(event.canonical_id)

    def on_news(self, event: NewsEvent) -> None:
        if not bool(self.param("news_pause", True)):
            return
        tone_threshold = float(self.param("news_impact_tone_threshold", 0.7))
        if self.is_high_impact_news(event, tone_threshold):
            self._news_pause_started_at = event.first_seen_time
            for canonical_id in list(self._resting_orders):
                self._cancel_all(canonical_id)

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._evaluate(event.canonical_id)

    # ------------------------------------------------------------------

    def _in_news_pause(self) -> bool:
        started = self._news_pause_started_at
        if started is None:
            return False
        pause_seconds = float(self.param("news_pause_seconds", DEFAULT_NEWS_PAUSE_SECONDS))
        return (self.now() - started).total_seconds() < pause_seconds

    def _evaluate(self, canonical_id: str) -> None:
        skip_reason = self.should_skip(canonical_id)
        if skip_reason is not None:
            self._cancel_all(canonical_id)
            return
        if self._in_news_pause():
            self._cancel_all(canonical_id)
            return

        book = self.ctx.book(canonical_id)
        market = self.ctx.market(canonical_id)
        assert book is not None and market is not None
        fair = book.mid
        if fair is None:
            self._cancel_all(canonical_id)
            return

        if bool(self.param("high_liquidity_only", False)):
            min_liquidity = Decimal(str(self.param("min_liquidity", "0")))
            if market.liquidity < min_liquidity:
                self._cancel_all(canonical_id)
                return

        st = self.state(canonical_id)
        returns: RollingWindow = st.setdefault("returns", RollingWindow(30))
        prev_fair = st.get("prev_fair")
        if prev_fair is not None:
            returns.push(float(fair - prev_fair))
        st["prev_fair"] = fair
        vol = returns.std() or 0.0

        if bool(self.param("low_volatility_only", False)):
            max_vol = float(self.param("max_volatility", 0.05))
            if vol > max_vol:
                self._cancel_all(canonical_id)
                return

        last_quoted_fair: Decimal | None = st.get("last_quoted_fair")
        adverse_threshold = Decimal(str(self.param("adverse_move_threshold", DEFAULT_ADVERSE_MOVE_THRESHOLD)))
        if last_quoted_fair is not None and abs(fair - last_quoted_fair) >= adverse_threshold:
            self._cancel_all(canonical_id)

        # ---- profitability gate, BEFORE any vol/inventory adjustment ----
        # This is the "would this even be worth it" check: can a round-trip that has to be
        # unwound as a taker (the worst case for a resting quote that gets run over) clear
        # the spread we're proposing to charge? Fee peaks exactly at p=0.50 - the price a
        # maker most wants to sit at - so this gate bites hardest exactly there.
        base_spread = Decimal(str(self.param("base_spread", "0.03")))
        unwind_fee = self.estimate_fee(market, fair, quantity=1, is_maker=False)
        adverse_selection_buffer = Decimal(str(self.param("adverse_selection_buffer", "0")))
        required_cost_cover = 2 * unwind_fee + adverse_selection_buffer
        if base_spread <= required_cost_cover:
            self._cancel_all(canonical_id)
            return

        time_to_close = _seconds_to_close(market, self.now())
        stop_seconds = float(self.param("stop_quoting_seconds", DEFAULT_STOP_QUOTING_SECONDS))
        if time_to_close is not None and time_to_close <= stop_seconds:
            self._cancel_all(canonical_id)
            return

        inventory = self._inventory.get(canonical_id, 0)
        inventory_penalty = Decimal(str(self.param("inventory_penalty", "1.0")))
        horizon = float(self.param("skew_horizon_seconds", DEFAULT_SKEW_HORIZON_SECONDS))
        # Variance of a binary claim collapses toward resolution, so inventory risk over
        # the *remaining* life of the contract shrinks too - the skew scales down with it.
        time_factor = 1.0 if time_to_close is None else max(0.0, min(1.0, time_to_close / horizon))
        vol_decimal = Decimal(str(vol))
        # A perfectly quiet book (no realized variance yet - e.g. the very first quote of a
        # session) would otherwise zero out the inventory term entirely; a small floor keeps
        # inventory management active even before enough ticks have accumulated to measure
        # volatility, since a discrete tick's worth of price uncertainty is always present.
        min_vol_floor = Decimal(str(self.param("min_volatility_floor", "0.001")))
        vol_for_skew = max(vol_decimal, min_vol_floor)
        inventory_skew = (
            INVENTORY_SKEW_SCALE
            * inventory_penalty
            * Decimal(inventory)
            * vol_for_skew
            * Decimal(str(time_factor))
        )
        reservation_price = fair - inventory_skew

        volatility_adjustment = vol_decimal * Decimal(str(self.param("vol_spread_gain", "1.0")))
        adverse_selection_adjustment = unwind_fee * Decimal(str(self.param("adverse_selection_gain", "1.0")))
        desired_half_spread = base_spread / 2 + volatility_adjustment + adverse_selection_adjustment

        close_buffer = float(self.param("close_buffer_seconds", DEFAULT_CLOSE_BUFFER_SECONDS))
        if time_to_close is not None and time_to_close <= close_buffer:
            # Informational/event risk right before a close is *not* the same thing as
            # diffusive volatility - a closing print or last-second news can move the
            # outcome sharply even on a contract whose price has been dead flat. Widen for
            # this explicitly rather than relying on the (shrinking) volatility term above.
            late_widen_max = Decimal(str(self.param("late_widen_max", DEFAULT_LATE_WIDEN_MAX)))
            late_fraction = Decimal(str((close_buffer - time_to_close) / close_buffer))
            desired_half_spread += late_widen_max * late_fraction

        tick = market.tick_size
        bid_price = _clamp_price(reservation_price - desired_half_spread, tick)
        ask_price = _clamp_price(reservation_price + desired_half_spread, tick)

        mode = str(self.param("mode", "two_sided"))
        max_inventory = int(self.param("max_inventory", DEFAULT_MAX_INVENTORY))
        quote_bid, quote_ask = _sides_to_quote(mode, inventory, max_inventory)

        quantity = self.sensible_quantity(fair, risk_fraction=Decimal(str(self.param("risk_fraction", "0.03"))))
        features_base: dict[str, Any] = {
            "fair_probability": float(fair),
            "reservation_price": float(reservation_price),
            "inventory": inventory,
            "inventory_penalty": float(inventory_penalty),
            "inventory_skew": float(inventory_skew),
            "volatility": vol,
            "volatility_adjustment": float(volatility_adjustment),
            "adverse_selection_adjustment": float(adverse_selection_adjustment),
            "unwind_fee_per_contract": float(unwind_fee),
            "desired_half_spread": float(desired_half_spread),
            "base_spread": float(base_spread),
            "time_to_close_seconds": time_to_close,
            "mode": mode,
            "bid_price": float(bid_price),
            "ask_price": float(ask_price),
        }

        self._cancel_all(canonical_id)
        quoted_any = False

        if quote_bid:
            bid_edge = self._maker_edge_buy(fair, bid_price, market)
            intent = self.make_intent(
                canonical_id=canonical_id,
                side=Side.YES,
                action=Action.BUY,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=bid_price,
                rationale=(
                    f"AS-binary bid: fair={fair}, reservation={reservation_price}, "
                    f"half_spread={desired_half_spread}, inventory={inventory}, "
                    f"mode={mode} -> bid={bid_price}, edge={bid_edge}."
                ),
                features={**features_base, "quote_side": "bid", "expected_edge": float(bid_edge)},
                model_probability=fair,
                expected_edge=bid_edge,
            )
            if self.emit_if_profitable(intent):
                quoted_any = True

        if quote_ask:
            ask_edge = self._maker_edge_sell(fair, ask_price, market)
            intent = self.make_intent(
                canonical_id=canonical_id,
                side=Side.YES,
                action=Action.SELL,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=ask_price,
                rationale=(
                    f"AS-binary ask: fair={fair}, reservation={reservation_price}, "
                    f"half_spread={desired_half_spread}, inventory={inventory}, "
                    f"mode={mode} -> ask={ask_price}, edge={ask_edge}."
                ),
                features={**features_base, "quote_side": "ask", "expected_edge": float(ask_edge)},
                model_probability=fair,
                expected_edge=ask_edge,
            )
            if self.emit_if_profitable(intent):
                quoted_any = True

        if quoted_any:
            st["last_quoted_fair"] = fair

    # ------------------------------------------------------------------ maker-side edge

    def _maker_edge_buy(self, fair: Decimal, price: Decimal, market: NormalizedMarket) -> Decimal:
        """Edge of resting a BUY YES quote at ``price``: what we'd earn if it gets filled."""
        fee = self.estimate_fee(market, price, quantity=1, is_maker=True)
        slippage_buffer = Decimal(str(self.param("slippage_buffer", "0.005")))
        uncertainty_buffer = Decimal(str(self.param("uncertainty_buffer", "0.01")))
        return fair - price - fee - slippage_buffer - uncertainty_buffer

    def _maker_edge_sell(self, fair: Decimal, price: Decimal, market: NormalizedMarket) -> Decimal:
        """Edge of resting a SELL YES quote at ``price``: mirror image of the buy-side edge."""
        fee = self.estimate_fee(market, price, quantity=1, is_maker=True)
        slippage_buffer = Decimal(str(self.param("slippage_buffer", "0.005")))
        uncertainty_buffer = Decimal(str(self.param("uncertainty_buffer", "0.01")))
        return price - fair - fee - slippage_buffer - uncertainty_buffer


def _seconds_to_close(market: NormalizedMarket, now: Any) -> float | None:
    if market.close_time is None:
        return None
    return (market.close_time - now).total_seconds()


def _clamp_price(price: Decimal, tick: Decimal) -> Decimal:
    lo = tick
    hi = ONE - tick
    return max(lo, min(hi, price))


def _sides_to_quote(mode: str, inventory: int, max_inventory: int) -> tuple[bool, bool]:
    """Which of (bid, ask) to post, given the quoting mode and current inventory."""
    if mode == "two_sided":
        return True, True
    if mode == "one_sided":
        if inventory > 0:
            return False, True  # long YES -> only offer to sell it down
        if inventory < 0:
            return True, False  # short YES (long NO) -> only offer to buy it back
        return True, False  # flat: default to the bid side
    if mode == "inventory_neutral":
        if inventory >= max_inventory:
            return False, True
        if inventory <= -max_inventory:
            return True, False
        return True, True
    raise ValueError(f"unknown mode: {mode!r}")
