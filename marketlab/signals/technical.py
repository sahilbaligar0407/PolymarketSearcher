"""Price/return features for both reference-asset feeds (BTC, SPY) and contract prices.

Market data is irregular - ticks do not arrive on a fixed grid - so every lookback here is
**time-based**, taking explicit timestamps, rather than counting ticks. A "60-tick
lookback" means something completely different on a quiet market than on a busy one; a
60-second lookback means the same thing on both.

Prediction-contract prices live in ``[0, 1]``, where a raw percentage return is badly
behaved near the boundaries (0.01 -> 0.02 is +100% but economically tiny; 0.50 -> 0.51 is
+2% and roughly the same tiny move). :func:`bounded_momentum` is the logit-space fix, and
is the default momentum definition strategies should use on contract prices. Ordinary
``simple_return``/``log_return`` remain correct and are the right tool for
reference-asset feeds like BTC or SPY, which are not bounded to ``[0, 1]``.
"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal

from marketlab.signals.rolling import RollingWindow

Series = list[Decimal] | RollingWindow

#: Clamp floor for :func:`bounded_momentum`. This is deliberately much larger than the
#: 1e-9-scale epsilon that would merely keep ``log`` from seeing 0 or 1 - see that
#: function's docstring for why a small numerical-safety epsilon actually backfires here.
_LOGIT_CLAMP = 0.05


def _values(series: Series) -> list[float]:
    if isinstance(series, RollingWindow):
        return series.values()
    return [float(x) for x in series]


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def simple_return(p0: Decimal, p1: Decimal) -> float | None:
    """``(p1 - p0) / p0``. ``None`` if ``p0`` is zero (undefined, not infinite)."""
    if p0 == 0:
        return None
    return float((p1 - p0) / p0)


def log_return(p0: Decimal, p1: Decimal) -> float | None:
    """``ln(p1 / p0)``. ``None`` if either price is non-positive."""
    if p0 <= 0 or p1 <= 0:
        return None
    return math.log(float(p1) / float(p0))


def momentum(
    series: list[tuple[datetime, Decimal]], lookback_seconds: float, now: datetime
) -> float | None:
    """Simple return over a time-based lookback window ending at ``now``.

    ``series`` is ``[(timestamp, price), ...]`` in chronological order. Finds the most
    recent observation at or before ``now`` (the current price) and the most recent
    observation at or before ``now - lookback_seconds`` (the anchor), then returns the
    simple return between them. ``None`` if there isn't an observation old enough to
    anchor the lookback, or fewer than two points are available.
    """
    if not series:
        return None
    eligible = [(ts, p) for ts, p in series if ts <= now]
    if len(eligible) < 2:
        return None
    current_ts, current_price = eligible[-1]
    cutoff = current_ts.timestamp() - lookback_seconds
    anchor: tuple[datetime, Decimal] | None = None
    for ts, p in eligible:
        if ts.timestamp() <= cutoff:
            anchor = (ts, p)
        else:
            break
    if anchor is None:
        return None
    _, anchor_price = anchor
    return simple_return(anchor_price, current_price)


def volatility(
    returns: list[float], annualize: bool = False, periods_per_year: float = 252.0
) -> float | None:
    """Sample standard deviation of a list of returns. ``None`` for fewer than 2 or zero variance."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return None
    sd = math.sqrt(var)
    if annualize:
        sd *= math.sqrt(periods_per_year)
    return sd


def realized_volatility(
    prices: list[Decimal],
    timestamps: list[datetime],
    window_seconds: float,
    annualize: bool = True,
) -> float | None:
    """Realized vol from log returns over the trailing ``window_seconds``, properly time-scaled.

    Uses only the observations within the window, computes log returns between
    *consecutive* observations (each already irregularly spaced), then scales the sample
    stdev of those returns by the actual average time-step to annualize correctly instead
    of assuming a fixed bar size.
    """
    if len(prices) != len(timestamps) or len(prices) < 3:
        return None
    now = timestamps[-1]
    cutoff = now.timestamp() - window_seconds
    idx = [i for i, ts in enumerate(timestamps) if ts.timestamp() >= cutoff]
    if len(idx) < 3:
        return None
    window_prices = [prices[i] for i in idx]
    window_ts = [timestamps[i] for i in idx]

    log_rets: list[float] = []
    dts: list[float] = []
    for i in range(1, len(window_prices)):
        r = log_return(window_prices[i - 1], window_prices[i])
        dt = (window_ts[i] - window_ts[i - 1]).total_seconds()
        if r is None or dt <= 0:
            continue
        log_rets.append(r)
        dts.append(dt)
    if len(log_rets) < 2:
        return None

    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
    if var <= 0:
        return None
    sd = math.sqrt(var)
    avg_dt = sum(dts) / len(dts)
    if not annualize:
        return sd
    seconds_per_year = 365.0 * 24.0 * 3600.0
    periods_per_year = seconds_per_year / avg_dt
    return sd * math.sqrt(periods_per_year)


def volatility_adjusted_momentum(mom: float | None, vol: float | None) -> float | None:
    """``momentum / volatility`` - the risk-normalized form time-series momentum literature uses.

    ``None`` if either input is ``None`` or vol is zero (undefined, not infinite).
    """
    if mom is None or vol is None or vol == 0:
        return None
    return mom / vol


# ---------------------------------------------------------------------------
# Moving averages / z-scores
# ---------------------------------------------------------------------------


def moving_average(series: Series, n: int) -> float | None:
    values = _values(series)
    if len(values) < n or n <= 0:
        return None
    return sum(values[-n:]) / n


def ema(series: Series, halflife: float) -> float | None:
    """Tick-indexed EMA (uniform decay per observation). Use :class:`signals.rolling.EWMA`
    instead when the series is irregularly time-spaced."""
    values = _values(series)
    if not values or halflife <= 0:
        return None
    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
    out = values[0]
    for v in values[1:]:
        out = out + alpha * (v - out)
    return out


def ma_slope(series: Series, n: int) -> float | None:
    """Change in the ``n``-period moving average between the last two points it can be computed at."""
    values = _values(series)
    if len(values) < n + 1 or n <= 0:
        return None
    prev = sum(values[-n - 1 : -1]) / n
    curr = sum(values[-n:]) / n
    return curr - prev


def zscore(value: float, mean: float, std: float | None) -> float | None:
    """``None`` (never a division error) when ``std`` is ``None`` or zero."""
    if std is None or std == 0:
        return None
    return (value - mean) / std


def rolling_zscore(series: Series, window: int) -> float | None:
    """Z-score of the most recent value against the trailing ``window`` (including itself)."""
    values = _values(series)
    if len(values) < window or window < 2:
        return None
    tail = values[-window:]
    mean = sum(tail) / len(tail)
    var = sum((v - mean) ** 2 for v in tail) / (len(tail) - 1)
    if var <= 0:
        return None
    return (tail[-1] - mean) / math.sqrt(var)


# ---------------------------------------------------------------------------
# VWAP / range
# ---------------------------------------------------------------------------


def vwap(prices: list[Decimal], volumes: list[int]) -> Decimal | None:
    if not prices or len(prices) != len(volumes):
        return None
    total_volume = sum(volumes)
    if total_volume == 0:
        return None
    total_cost = sum((p * v for p, v in zip(prices, volumes, strict=True)), Decimal(0))
    return total_cost / total_volume


def deviation_from_vwap(price: Decimal, vwap_value: Decimal | None) -> float | None:
    if vwap_value is None or vwap_value == 0:
        return None
    return float((price - vwap_value) / vwap_value)


def breakout_distance(price: Decimal, window_high: Decimal, window_low: Decimal) -> float | None:
    """Where ``price`` sits in ``[window_low, window_high]``, normalized to ``[-1, 1]``.

    -1 at the low, +1 at the high, 0 at the midpoint. ``None`` if the range is degenerate
    (high <= low).
    """
    span = window_high - window_low
    if span <= 0:
        return None
    midpoint = (window_high + window_low) / 2
    return float((price - midpoint) / (span / 2))


def atr(highs: list[Decimal], lows: list[Decimal], closes: list[Decimal], n: int) -> float | None:
    """Average True Range over the last ``n`` bars, for reference-asset (BTC/SPY) feeds."""
    if not (len(highs) == len(lows) == len(closes)) or len(closes) < n + 1 or n <= 0:
        return None
    true_ranges: list[float] = []
    for i in range(1, len(closes)):
        h, low, prev_close = highs[i], lows[i], closes[i - 1]
        tr = max(
            float(h - low),
            abs(float(h - prev_close)),
            abs(float(low - prev_close)),
        )
        true_ranges.append(tr)
    tail = true_ranges[-n:]
    if len(tail) < n:
        return None
    return sum(tail) / n


# ---------------------------------------------------------------------------
# Probability-space (contract price) momentum
# ---------------------------------------------------------------------------


def _logit(p: float, clamp: float = _LOGIT_CLAMP) -> float:
    p = min(max(p, clamp), 1.0 - clamp)
    return math.log(p / (1.0 - p))


def bounded_momentum(
    p_now: Decimal | float, p_then: Decimal | float, clamp: float = _LOGIT_CLAMP
) -> float:
    """Logit-space momentum for a ``[0, 1]``-bounded contract price: ``logit(p_now) - logit(p_then)``.

    A raw percentage return is badly behaved near the boundaries of a probability: a move
    from 0.01 to 0.02 is +100% even though, in dollar/tick terms, it is the smallest
    possible move a contract can make. The obvious "fix" is a logit transform, since it
    maps ``[0, 1]`` to ``(-inf, inf)``... but the logit's derivative is ``1/(p(1-p))``,
    which is *largest* right at the boundaries. Taken naively (clamping only enough to
    keep ``log`` from seeing exactly 0 or 1), logit-momentum makes the 0.01->0.02 move
    register as an even *bigger* signal than the naive percentage return does, not a
    smaller one - the fix would make the original problem worse.

    So the clamp here does real work, not just numerical hygiene: probabilities are
    floored/capped at ``clamp`` (default 5%) before the logit is taken. Below that band,
    a contract's touch price is dominated by one-lot noise and bid/ask bounce rather than
    genuine information, so 0.01 and 0.02 both collapse to the same clamped value and
    contribute *zero* logit-momentum - deliberately sacrificing sensitivity in the
    noise-dominated deep tail in exchange for not generating wild momentum spikes off
    single-tick prints there. A move entirely inside the ``[clamp, 1-clamp]`` band (e.g.
    0.50 -> 0.60) is unaffected and scores in proportion to its real informational size.
    This is the default momentum definition for contract-price series; use
    :func:`simple_return`/:func:`log_return` for unbounded reference-asset prices
    (BTC, SPY) instead, where there is no such boundary to guard against.
    """
    return _logit(float(p_now), clamp) - _logit(float(p_then), clamp)


def probability_drift(
    p_series: list[Decimal], timestamps: list[datetime], window_seconds: float, now: datetime
) -> float | None:
    """Drift in probability terms per second over the trailing ``window_seconds``.

    Plain (non-logit) probability difference divided by elapsed time - this is a rate of
    change in the contract's own units (probability/second), distinct from
    :func:`bounded_momentum`'s logit-space magnitude, and is the right primitive for "how
    fast is this contract's price moving right now."
    """
    if len(p_series) != len(timestamps) or not p_series:
        return None
    eligible = [(ts, p) for ts, p in zip(timestamps, p_series, strict=True) if ts <= now]
    if len(eligible) < 2:
        return None
    current_ts, current_p = eligible[-1]
    cutoff = current_ts.timestamp() - window_seconds
    anchor: tuple[datetime, Decimal] | None = None
    for ts, p in eligible:
        if ts.timestamp() <= cutoff:
            anchor = (ts, p)
        else:
            break
    if anchor is None:
        return None
    anchor_ts, anchor_p = anchor
    elapsed = (current_ts - anchor_ts).total_seconds()
    if elapsed <= 0:
        return None
    return float(current_p - anchor_p) / elapsed
