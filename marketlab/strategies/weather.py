"""NWS forecast vs. Kalshi weather-threshold contracts.

Each Kalshi weather series is mapped to its NWS station via the ``stations:`` table in
``configs/universes.yaml`` (owned by another team). :class:`~marketlab.core.strategy.
StrategyContext` has no knowledge of that YAML file, so the table is supplied through
``params["stations"]`` (series-prefix -> station id, e.g. ``{"KXHIGHNY": "KNYC"}``) -
the same "narrow context, config injected via params" pattern used by
:mod:`marketlab.strategies.cross_venue` for match records.

**Converting a forecast into a probability is an explicit, honestly-labelled assumption,
not a known truth.** A forecast high temperature is modelled as
``Normal(mu=forecast_high, sigma=forecast_sigma_f)``, and ``forecast_sigma_f`` (default
3.0 degF) is a *parameter*, not a fitted constant - NWS day-ahead high-temperature error is
commonly cited in that neighbourhood, but this module makes no claim it is correct for any
particular station or lead time. It is recorded in every intent's ``features`` precisely so
it can be measured against realized forecast error later, rather than trusted.

Three arms:

* **official forecast vs market** (``use_forecast_change=false``) - compare the latest NWS
  forecast to the current contract price whenever the book (or a periodic timer) ticks.
* **forecast change** (``use_forecast_change=true``) - react specifically to a *new* NWS
  forecast issuance (``on_weather``), which is when the market is most likely to be lagging
  a just-published update; a minimum forecast-delta gate avoids re-trading on a no-op
  reissue of the same number.
* **ensemble of forecast updates** - not a separate grid dimension (not swept in
  ``configs/strategies.yaml``), but always computed: the mean/stdev of the last few
  forecast issuances for the same target date is folded into an *effective* sigma
  (``sigma_effective = forecast_sigma_f + ensemble_std``), so a station whose forecast has
  been jumping around gets an automatically wider, more honest uncertainty band.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, TimerEvent, WeatherEvent
from marketlab.core.instruments import ONE, Category, Side
from marketlab.core.orders import Action, OrderType
from marketlab.core.probability import normal_cdf
from marketlab.core.strategy import ProbabilityForecast
from marketlab.signals.rolling import RollingWindow
from marketlab.strategies.base import BaseStrategy, clamp_probability

DEFAULT_FORECAST_SIGMA_F = 3.0
DEFAULT_MIN_FORECAST_DELTA_F = 1.0

_TICKER_THRESHOLD_RE = re.compile(r"-T(?P<num>-?\d+(?:\.\d+)?)\b")
_TITLE_TEMP_RE = re.compile(r"(?P<num>-?\d+(?:\.\d+)?)\s*(?:°\s*F|deg(?:rees)?\s*F?|F\b)", re.IGNORECASE)
_BELOW_KEYWORDS = ("below", "under", "or lower", "at or below")


def forecast_to_threshold_probability(
    forecast_high_f: float, threshold_f: float, sigma_f: float, comparator: str = ">="
) -> float:
    """``P(actual satisfies comparator threshold)`` under ``Normal(forecast_high_f, sigma_f)``."""
    if sigma_f <= 0:
        raise ValueError("sigma_f must be positive")
    z = (threshold_f - forecast_high_f) / sigma_f
    p_below = normal_cdf(z)
    if comparator in (">=", ">"):
        return 1.0 - p_below
    if comparator in ("<=", "<"):
        return p_below
    raise ValueError(f"unsupported comparator: {comparator!r}")


_TICKER_BUCKET_RE = re.compile(r"-B(?P<mid>-?\d+(?:\.\d+)?)$")


def parse_temp_bucket(ticker: str) -> Decimal | None:
    """Midpoint of a Kalshi range bucket: ``KXHIGHNY-26OCT03-B68.5`` is "68-69 F"."""
    m = _TICKER_BUCKET_RE.search((ticker or "").upper())
    return Decimal(m.group("mid")) if m else None


def forecast_to_bucket_probability(forecast_high_f: float, mid_f: float, sigma_f: float) -> float:
    """``P(reported high is one of the bucket's two whole degrees)``.

    A ``-B68.5`` bucket covers reported highs of 68 and 69. Highs are reported in whole
    degrees, so with a continuity correction the bucket is [mid - 1, mid + 1).
    """
    if sigma_f <= 0:
        raise ValueError("sigma_f must be positive")
    upper = normal_cdf((mid_f + 1.0 - forecast_high_f) / sigma_f)
    lower = normal_cdf((mid_f - 1.0 - forecast_high_f) / sigma_f)
    return max(0.0, upper - lower)


def parse_temp_threshold(ticker: str, title: str = "") -> Decimal | None:
    m = _TICKER_THRESHOLD_RE.search(ticker or "")
    if m:
        try:
            return Decimal(m.group("num"))
        except Exception:  # noqa: BLE001
            pass
    m2 = _TITLE_TEMP_RE.search(title or "")
    if m2:
        try:
            return Decimal(m2.group("num"))
        except Exception:  # noqa: BLE001
            pass
    return None


def is_below_threshold_contract(title: str) -> bool:
    return any(k in title.lower() for k in _BELOW_KEYWORDS)


_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
_MARKET_DATE_RE = re.compile(r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?:-|$)")


def parse_market_date(ticker: str) -> date | None:
    """The local date a daily-high contract is about: KXHIGHLAX-26OCT04-T98 -> 2026-10-04."""
    m = _MARKET_DATE_RE.search((ticker or "").upper())
    if m is None or m["mon"] not in _MONTHS:
        return None
    try:
        return date(2000 + int(m["yy"]), _MONTHS[m["mon"]], int(m["dd"]))
    except ValueError:
        return None


def _forecast_key(station: str, day: date | None) -> str:
    return f"{station}|{day.isoformat() if day else '-'}"


def series_prefix(ticker: str) -> str:
    return ticker.split("-")[0]


class WeatherForecastStrategy(BaseStrategy):
    """Prices Kalshi weather-threshold contracts against the latest NWS forecast."""

    name = "weather_forecast"
    version = "1.2.0"
    evidence_class = "B"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        #: station -> {"latest": value, "issued_at": ts, "forecast_for": ts, "history": RollingWindow}
        self._station_forecasts: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ inputs

    def on_weather(self, event: WeatherEvent) -> None:
        if event.value is None:
            return
        # Only daytime-high forecasts price a daily-high contract. The NWS feed also
        # carries overnight lows, hourly temperatures and alerts; storing whichever
        # arrived last compared a 66 F night low with a ">98 F high" contract and, by
        # mixing highs and lows in the revision history, inflated sigma to 12 F.
        if event.variable and event.variable != "temperature_forecast_day":
            return
        key = _forecast_key(event.station, event.forecast_for.date() if event.forecast_for else None)
        st = self._station_forecasts.setdefault(
            key, {"latest": None, "issued_at": None, "forecast_for": None, "history": RollingWindow(10)}
        )
        previous_value = st["latest"]
        st["history"].push(float(event.value))
        st["latest"] = event.value
        st["issued_at"] = event.issued_at or event.first_seen_time
        st["forecast_for"] = event.forecast_for

        if bool(self.param("use_forecast_change", False)):
            delta = None if previous_value is None else abs(float(event.value - previous_value))
            min_delta = float(self.param("min_forecast_delta_f", DEFAULT_MIN_FORECAST_DELTA_F))
            if delta is not None and delta >= min_delta:
                self._evaluate_station(event.station, trigger="forecast_update", forecast_delta=delta)

    def on_book_update(self, event: BookUpdateEvent) -> None:
        market = self.ctx.market(event.canonical_id)
        if market is None:
            return
        station = self._station_for(market)
        if station is None:
            return
        self._evaluate_market(event.canonical_id, market, station, trigger="book_update", forecast_delta=None)

    def on_timer(self, event: TimerEvent) -> None:
        del event
        for market in self.ctx.markets():
            if market.category is not Category.WEATHER:
                continue
            station = self._station_for(market)
            if station is None:
                continue
            self._evaluate_market(market.canonical_id, market, station, trigger="timer", forecast_delta=None)

    # ------------------------------------------------------------------ helpers

    def _station_for(self, market: Any) -> str | None:
        stations: dict[str, str] = self.param("stations", {}) or {}
        return stations.get(series_prefix(market.venue_market_id))

    def _evaluate_station(self, station: str, trigger: str, forecast_delta: float | None) -> None:
        for market in self.ctx.markets():
            if market.category is not Category.WEATHER:
                continue
            if self._station_for(market) != station:
                continue
            self._evaluate_market(market.canonical_id, market, station, trigger, forecast_delta)

    def _evaluate_market(
        self, canonical_id: str, market: Any, station: str, trigger: str, forecast_delta: float | None
    ) -> None:
        if self.should_skip(canonical_id) is not None:
            return
        book = self.ctx.book(canonical_id)
        assert book is not None
        # The forecast for the contract's own date; an undated forecast only serves
        # undated callers (unit tests and legacy events).
        market_date = parse_market_date(market.venue_market_id)
        forecast_state = self._station_forecasts.get(_forecast_key(station, market_date))
        if forecast_state is None and market_date is not None:
            forecast_state = self._station_forecasts.get(_forecast_key(station, None))
        if forecast_state is None or forecast_state["latest"] is None:
            return

        base_sigma = float(self.param("forecast_sigma_f", DEFAULT_FORECAST_SIGMA_F))
        history: RollingWindow = forecast_state["history"]
        ensemble_std = history.std() or 0.0
        sigma_effective = base_sigma + ensemble_std

        # Most Kalshi daily-high markets are two-degree range buckets ("-B68.5" = 68-69),
        # not thresholds. Pricing a bucket as ">= 69" answers a different question, so a
        # bucket gets its own probability and anything else must be an explicit "-T".
        bucket_mid = parse_temp_bucket(market.venue_market_id)
        if bucket_mid is not None:
            threshold = bucket_mid
            comparator = "in_bucket"
            p_yes = forecast_to_bucket_probability(float(forecast_state["latest"]), float(bucket_mid), sigma_effective)
        else:
            if _TICKER_THRESHOLD_RE.search(market.venue_market_id or "") is None:
                return  # neither a bucket nor an explicit threshold: do not guess
            threshold = parse_temp_threshold(market.venue_market_id, market.title)
            if threshold is None:
                return
            # Kalshi writes tails as symbols (">98°", "<70°"); words are the fallback.
            if "<" in market.title:
                below = True
            elif ">" in market.title:
                below = False
            else:
                below = is_below_threshold_contract(f"{market.title} {market.description}")
            comparator = "<=" if below else ">="
            p_yes = forecast_to_threshold_probability(
                float(forecast_state["latest"]), float(threshold), sigma_effective, comparator
            )
        model_probability = clamp_probability(Decimal(str(round(p_yes, 6))))

        features: dict[str, Any] = {
            "station": station,
            "forecast_high_f": float(forecast_state["latest"]),
            "threshold_f": float(threshold),
            "comparator": comparator,
            "forecast_sigma_f": base_sigma,
            "ensemble_std_f": ensemble_std,
            "sigma_effective_f": sigma_effective,
            "trigger": trigger,
            "forecast_delta_f": forecast_delta,
            "model_probability": float(model_probability),
            "market_probability": float(book.mid) if book.mid is not None else None,
        }
        self.forecast(
            ProbabilityForecast(
                strategy_id=self.strategy_id,
                experiment_id=self.experiment_id,
                canonical_id=canonical_id,
                as_of=self.now(),
                p_yes=model_probability,
                market_probability=book.mid,
                features=features,
                rationale=(
                    f"NWS {station} forecast={forecast_state['latest']}F, sigma_effective="
                    f"{sigma_effective:.2f}F -> p_yes={model_probability} for threshold "
                    f"{comparator} {threshold}F."
                ),
            )
        )

        price_yes = self.executable_price(book, Side.YES, Action.BUY)
        price_no = self.executable_price(book, Side.NO, Action.BUY)
        candidates: list[tuple[Side, Decimal, Decimal]] = []
        if price_yes is not None:
            candidates.append((Side.YES, price_yes, self.edge_after_costs(model_probability, price_yes, market, Side.YES)))
        if price_no is not None:
            no_p = clamp_probability(ONE - model_probability)
            candidates.append((Side.NO, price_no, self.edge_after_costs(no_p, price_no, market, Side.NO)))
        if not candidates:
            return
        side, price, edge = max(candidates, key=lambda c: c[2])

        min_edge = Decimal(str(self.param("min_edge", "0.04")))
        if edge < min_edge:
            return
        cooldown = float(self.param("cooldown_seconds", 300.0))
        if self.on_cooldown(canonical_id, cooldown):
            return

        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=self.sensible_quantity(price),
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Weather forecast ({trigger}): model_p={model_probability} vs "
                f"executable {side.value} price {price}; edge {edge} clears "
                f"min_edge {min_edge}."
            ),
            features={**features, "executable_price": float(price), "side": side.value},
            model_probability=model_probability,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


__all__ = ["WeatherForecastStrategy", "forecast_to_threshold_probability", "parse_temp_threshold"]
