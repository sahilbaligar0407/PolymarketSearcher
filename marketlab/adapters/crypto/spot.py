"""BTC/ETH reference spot prices and realized volatility.

Endpoints probed live on 2026-09-04:

* ``GET api.exchange.coinbase.com/products/BTC-USD/ticker``   200, ``{"price": "79537.77", ...}``
* ``GET api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60``
  200, ``[[time, low, high, open, close, volume], ...]`` newest-first
* ``wss://ws-feed.exchange.coinbase.com`` subscribe ``{"channels": ["ticker"]}``  works,
  pushes a ``ticker`` message per trade with sub-second latency
* ``api.binance.com`` - DNS resolution failed outright from this network (``Could not
  resolve host``), which is even more restrictive than the documented HTTP 451
  geo-block. Both failure modes are handled identically: disable Binance for the rest
  of the process lifetime rather than retrying a source that cannot work.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
import websockets

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import ExternalPriceEvent
from marketlab.logging import get_logger

log = get_logger(__name__)

_MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0
_MAX_BACKOFF_SECONDS = 60.0


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class CryptoSpotAdapter(Adapter):
    """Coinbase Exchange primary, Binance fallback (disabled cleanly if unreachable)."""

    name = "crypto_spot"

    def __init__(
        self,
        *,
        coinbase_base: str,
        binance_base: str,
        ws_url: str = "wss://ws-feed.exchange.coinbase.com",
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or LiveClock()
        headers = {"User-Agent": "MarketLab research"}
        self._coinbase = HttpAdapter(coinbase_base, name="coinbase", default_headers=headers, clock=self._clock)
        self._binance = HttpAdapter(binance_base, name="binance", default_headers=headers, clock=self._clock)
        self._ws_url = ws_url
        #: Once Binance is confirmed unreachable (DNS failure or HTTP 451 geo-block),
        #: stop calling it - retrying a source that structurally cannot work just adds
        #: latency to every fallback path for no benefit.
        self._binance_status = SourceStatus.HEALTHY

    async def probe(self) -> SourceHealth:
        try:
            data = await self._coinbase.get_json("/products/BTC-USD/ticker")
            price = Decimal(str(data.get("price", "0")))
            status = SourceStatus.HEALTHY if price > 0 else SourceStatus.DEGRADED
            return SourceHealth(name=self.name, status=status, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("crypto_spot_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await asyncio.gather(self._coinbase.close(), self._binance.close())

    @staticmethod
    def _coinbase_product(symbol: str) -> str:
        s = symbol.upper()
        return s if "-" in s else f"{s}-USD"

    @staticmethod
    def _binance_symbol(symbol: str) -> str:
        s = symbol.upper()
        if "-" in s:
            base, quote = s.split("-", 1)
            quote = "USDT" if quote == "USD" else quote
            return f"{base}{quote}"
        return s if s.endswith("USDT") else f"{s}USDT"

    async def get_spot(self, symbol: str) -> ExternalPriceEvent:
        """Coinbase first; fall back to Binance only while it hasn't been disabled."""
        now = self._clock.now()
        product = self._coinbase_product(symbol)
        try:
            data = await self._coinbase.get_json(f"/products/{product}/ticker")
            price = Decimal(str(data["price"]))
            return ExternalPriceEvent(
                event_time=now,
                published_time=_parse_iso(data.get("time")) or now,
                first_seen_time=now,
                ingested_time=now,
                source="coinbase",
                venue="coinbase",
                symbol=product,
                price=price,
            )
        except Exception as exc:  # noqa: BLE001 - fall through to Binance
            log.warning("coinbase_spot_failed", symbol=symbol, error=str(exc))

        if self._binance_status is SourceStatus.DISABLED:
            raise RuntimeError(
                f"no working spot source for {symbol}: coinbase request failed and "
                "binance is disabled (unreachable from this network)"
            )

        bsym = self._binance_symbol(symbol)
        try:
            data = await self._binance.get_json("/api/v3/ticker/price", params={"symbol": bsym})
            price = Decimal(str(data["price"]))
            return ExternalPriceEvent(
                event_time=now, published_time=now, first_seen_time=now, ingested_time=now,
                source="binance", venue="binance", symbol=bsym, price=price,
            )
        except httpx.ConnectError as exc:
            # DNS resolution failure - observed in this environment. Treat identically
            # to a geo-block: Binance cannot serve us, stop trying it.
            self._binance_status = SourceStatus.DISABLED
            log.warning("binance_disabled_unreachable", error=str(exc))
            raise RuntimeError(f"binance unreachable, disabling: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 451:
                self._binance_status = SourceStatus.DISABLED
                log.warning("binance_disabled_geoblock_451")
            raise

    async def get_candles(
        self,
        symbol: str,
        granularity: int = 60,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[list[float]]:
        """Raw Coinbase candles: ``[time, low, high, open, close, volume]``, newest first."""
        product = self._coinbase_product(symbol)
        params: dict[str, Any] = {"granularity": granularity}
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()
        data = await self._coinbase.get_json(f"/products/{product}/candles", params=params)
        return data if isinstance(data, list) else []

    async def realized_volatility(self, symbol: str, window_minutes: int = 60) -> float:
        """Annualized realized volatility from 1-minute candle closes.

        Uses the sample stdev of log returns over the trailing ``window_minutes``
        one-minute bars, annualized by ``sqrt(minutes per year)`` - the standard
        estimator strategies use to feed :func:`marketlab.core.probability.gbm_touch_probability`.
        Returns ``0.0`` if there isn't enough data (fewer than 3 candles) rather than
        raising, since a strategy calling this on a quiet market shouldn't crash.
        """
        candles = await self.get_candles(symbol, granularity=60)
        if not candles:
            return 0.0
        recent = candles[: max(window_minutes, 2)]
        # Coinbase returns newest-first; closes must be oldest->newest for returns.
        closes = [float(c[4]) for c in reversed(recent) if len(c) >= 5 and float(c[4]) > 0]
        if len(closes) < 3:
            return 0.0
        log_returns = [
            math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0
        ]
        if len(log_returns) < 2:
            return 0.0
        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        stdev = math.sqrt(max(variance, 0.0))
        return stdev * math.sqrt(_MINUTES_PER_YEAR)

    async def stream(
        self,
        symbols: list[str],
        queue: asyncio.Queue[ExternalPriceEvent],
        clock: Clock | None = None,
        min_interval_seconds: float = 0.0,
    ) -> None:
        """Subscribe to Coinbase's ``ticker`` channel and push events onto ``queue`` forever.

        Reconnects with exponential backoff (capped at 60s) on any failure. Runs until
        the calling task cancels it - there is no natural end to a live price stream.
        ``min_interval_seconds`` passes at most one tick per symbol per interval (BTC
        prints several times a second; every event fans out to every BTC sleeve).
        """
        last_sent: dict[str, float] = {}
        clock = clock or self._clock
        products = [self._coinbase_product(s) for s in symbols]
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self._ws_url, open_timeout=10, ping_interval=20) as ws:
                    await ws.send(json.dumps({"type": "subscribe", "product_ids": products, "channels": ["ticker"]}))
                    backoff = 1.0
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if msg.get("type") != "ticker":
                            continue
                        price_str = msg.get("price")
                        if not price_str:
                            continue
                        symbol = str(msg.get("product_id", ""))
                        mono = time.monotonic()
                        if mono - last_sent.get(symbol, float("-inf")) < min_interval_seconds:
                            continue
                        last_sent[symbol] = mono
                        now = clock.now()
                        event = ExternalPriceEvent(
                            event_time=_parse_iso(msg.get("time")) or now,
                            published_time=_parse_iso(msg.get("time")) or now,
                            first_seen_time=now,
                            ingested_time=now,
                            source="coinbase_ws",
                            venue="coinbase",
                            symbol=symbol,
                            price=Decimal(str(price_str)),
                        )
                        await queue.put(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect loop must never die
                log.warning("coinbase_ws_reconnect", error=str(exc), backoff_seconds=backoff)
                await clock.sleep(backoff)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_SECONDS)
