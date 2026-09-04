"""Unit tests for Kalshi's quadratic taker-fee formula: ceil(rate*C*P*(1-P)) cents."""

from __future__ import annotations

from decimal import Decimal

from marketlab.adapters.kalshi.fees import (
    DEFAULT_TAKER_RATE,
    REDUCED_TAKER_RATE,
    fees_for_market,
    settlement_fee,
    trading_fee,
)


def test_trading_fee_hand_computed_p50_100_contracts() -> None:
    # 0.07 * 100 * 0.5 * 0.5 * 100 = 175.0 cents exactly -> $1.75, no ceil needed.
    fee = trading_fee(Decimal("0.50"), 100)
    assert fee == Decimal("1.75")


def test_trading_fee_hand_computed_p30_10_contracts() -> None:
    # 0.07 * 10 * 0.30 * 0.70 * 100 = 14.7 cents -> ceil -> 15 cents -> $0.15.
    fee = trading_fee(Decimal("0.30"), 10)
    assert fee == Decimal("0.15")


def test_trading_fee_ceil_boundary_rounds_up_a_fraction_of_a_cent() -> None:
    # 0.07 * 1 * 0.01 * 0.99 * 100 = 0.0693 cents -> ceil -> 1 cent -> $0.01.
    fee = trading_fee(Decimal("0.01"), 1)
    assert fee == Decimal("0.01")


def test_trading_fee_ceil_does_not_round_up_an_exact_integer() -> None:
    # Exactly 175.0 cents must stay 175, not bump to 176 - ceil(x) == x when x is integral.
    fee = trading_fee(Decimal("0.50"), 100)
    assert fee == Decimal("1.75")
    # A case landing exactly on a whole cent with a different rate/price/size combo too:
    # 0.07 * 400 * 0.25 * 0.75 * 100 = 525.0 cents exactly -> $5.25
    fee2 = trading_fee(Decimal("0.25"), 400)
    assert fee2 == Decimal("5.25")


def test_trading_fee_peaks_at_p_050() -> None:
    contracts = 100
    fee_050 = trading_fee(Decimal("0.50"), contracts)
    for p in (Decimal("0.10"), Decimal("0.30"), Decimal("0.70"), Decimal("0.90"), Decimal("0.99")):
        assert trading_fee(p, contracts) <= fee_050
    # And it's symmetric around 0.50.
    assert trading_fee(Decimal("0.30"), contracts) == trading_fee(Decimal("0.70"), contracts)
    assert trading_fee(Decimal("0.10"), contracts) == trading_fee(Decimal("0.90"), contracts)


def test_trading_fee_maker_is_always_zero() -> None:
    assert trading_fee(Decimal("0.50"), 1000, is_maker=True) == Decimal("0.00")
    assert trading_fee(Decimal("0.01"), 1, is_maker=True) == Decimal("0.00")


def test_trading_fee_zero_contracts_is_zero() -> None:
    assert trading_fee(Decimal("0.50"), 0) == Decimal("0.00")


def test_trading_fee_reduced_rate() -> None:
    # 0.035 * 100 * 0.5 * 0.5 * 100 = 87.5 cents -> ceil -> 88 cents -> $0.88.
    # (Not exactly half of the 0.07 rate's $1.75: 87.5 ceils up, 175.0 doesn't.)
    full = trading_fee(Decimal("0.50"), 100, rate=DEFAULT_TAKER_RATE)
    reduced = trading_fee(Decimal("0.50"), 100, rate=REDUCED_TAKER_RATE)
    assert full == Decimal("1.75")
    assert reduced == Decimal("0.88")


def test_settlement_fee_is_zero_by_default() -> None:
    assert settlement_fee() == Decimal("0.00")
    assert settlement_fee(Decimal("0.5"), 10) == Decimal("0.00")


def test_fees_for_market_defaults_to_standard_formula() -> None:
    fees = fees_for_market({"ticker": "KXBTCD-1"})
    assert fees.taker_rate == DEFAULT_TAKER_RATE
    assert fees.maker_rate == Decimal("0.00")
    assert fees.formula == "kalshi_quadratic"


def test_fees_for_market_prefers_fetched_multiplier() -> None:
    # A series exposing fee_multiplier=0.5 should halve the effective rate.
    fees = fees_for_market({"ticker": "KXNFLGAME-1", "fee_multiplier": "0.5"})
    assert fees.taker_rate == Decimal("0.0350")


def test_fees_for_market_prefers_explicit_rate_over_multiplier() -> None:
    fees = fees_for_market(
        {"ticker": "KXNFLGAME-1", "fee_multiplier": "0.5", "taker_fee_rate": "0.02"}
    )
    assert fees.taker_rate == Decimal("0.02")
