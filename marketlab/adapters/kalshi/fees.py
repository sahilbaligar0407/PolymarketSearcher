"""Kalshi's trading fee schedule.

Standard Kalshi taker fee for a binary contract:

    fee_cents = ceil(rate * C * P * (1 - P) * 100)

where ``C`` is the number of contracts and ``P`` is the price in dollars (probability).
The fee is symmetric and peaks at ``P = 0.50``. ``rate`` is normally ``0.07`` but some
series (Kalshi has historically discounted certain high-volume/partner series) run a
reduced rate, commonly ``0.035``. Maker fills are not charged a taker fee.

All public functions here return **dollars** as ``Decimal``, rounded up to the cent -
that is what "ceil" means for a fee charged in whole cents.
"""

from __future__ import annotations

import contextlib
from decimal import ROUND_CEILING, Decimal

from marketlab.core.instruments import Fees

#: The general-purpose taker rate quoted in Kalshi's fee schedule.
DEFAULT_TAKER_RATE = Decimal("0.07")
#: The reduced rate applied to certain series (e.g. some sports/partner series).
REDUCED_TAKER_RATE = Decimal("0.035")
#: Kalshi does not charge maker fees on the standard schedule.
MAKER_RATE = Decimal("0.00")

_CENTS = Decimal(100)
_ONE = Decimal(1)


def trading_fee(
    price: Decimal,
    contracts: int,
    rate: Decimal = DEFAULT_TAKER_RATE,
    is_maker: bool = False,
) -> Decimal:
    """``ceil(rate * C * P * (1-P))`` cents, converted to dollars.

    Maker fills are free on Kalshi's standard schedule; ``is_maker=True`` short-circuits
    to zero regardless of price/size so callers don't need a separate branch.
    """
    if is_maker or contracts <= 0:
        return Decimal("0.00")
    price = Decimal(price)
    raw_cents = rate * Decimal(contracts) * price * (_ONE - price) * _CENTS
    fee_cents = raw_cents.to_integral_value(rounding=ROUND_CEILING)
    return (fee_cents / _CENTS).quantize(Decimal("0.01"))


def settlement_fee(
    price: Decimal | None = None,
    contracts: int | None = None,
    rate: Decimal = Decimal("0.00"),
) -> Decimal:
    """Kalshi charges no settlement fee on standard markets today.

    Kept as an explicit hook (rather than inlining ``Decimal("0.00")`` at call sites) so
    that if Kalshi introduces one for a market class, only this function changes.
    """
    del price, contracts  # unused while the fee is universally zero
    return (rate or Decimal("0.00")).quantize(Decimal("0.01"))


def fees_for_market(raw: dict) -> Fees:
    """Build a :class:`Fees` for one market, preferring values the API exposes.

    The ``/markets`` payload itself carries no fee fields; a series lookup
    (``/series/<ticker>``) exposes ``fee_type`` ("quadratic") and ``fee_multiplier``
    (an integer/decimal multiplier on the base rate). Callers that have already fetched
    the series may merge ``fee_multiplier`` / ``taker_fee_rate`` into the market dict
    before calling this; absent that, we fall back to the standard 0.07 quadratic
    formula, which is correct for the overwhelming majority of markets.
    """
    taker_rate = DEFAULT_TAKER_RATE

    fee_multiplier = raw.get("fee_multiplier")
    if fee_multiplier is not None:
        with contextlib.suppress(ValueError, ArithmeticError):
            taker_rate = (DEFAULT_TAKER_RATE * Decimal(str(fee_multiplier))).quantize(
                Decimal("0.0001")
            )

    explicit_rate = raw.get("taker_fee_rate", raw.get("fee_rate"))
    if explicit_rate is not None:
        with contextlib.suppress(ValueError, ArithmeticError):
            taker_rate = Decimal(str(explicit_rate))

    formula = "kalshi_quadratic" if str(raw.get("fee_type", "quadratic")) == "quadratic" else str(
        raw.get("fee_type")
    )

    return Fees(
        formula=formula,
        taker_rate=taker_rate,
        maker_rate=MAKER_RATE,
        settlement_rate=Decimal("0.00"),
        min_fee_cents=1,
    )
