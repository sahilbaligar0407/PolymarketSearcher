"""The Odds API (``the-odds-api.com``) adapter.

Probed live on 2026-09-04 without a key:

* ``GET api.the-odds-api.com/v4/sports?apiKey=`` -> HTTP 401,
  ``{"message":"API key is missing","error_code":"MISSING_KEY",...}``
  Confirms the auth gate; this adapter checks for a configured key *before* calling,
  rather than relying on that 401 - ``probe()`` reports ``NO_CREDENTIALS`` directly.

With a key, ``GET /v4/sports/{sport}/odds?regions=...&markets=...&oddsFormat=american``
is the documented endpoint. Every response carries an ``x-requests-remaining`` header;
since the free tier is small and this project layers its own ``monthly_request_budget``
on top, this adapter reads that header directly. The shared ``HttpAdapter`` only returns
parsed JSON (no header access), so this client manages its own small ``httpx.AsyncClient``
and rate limiter rather than going through it - the same pattern used for SEC's raw
Form 4 XML fetch, for the same reason (need something ``HttpAdapter``'s JSON-only
contract doesn't expose).

**Every American odds quote is converted through ``marketlab.core.probability`` before
being emitted.** Raw sportsbook odds embed the book's margin (vig); summing the raw
implied probabilities of a market's outcomes exceeds 1.0. Comparing a raw American-odds
probability directly to a Kalshi/Polymarket price would systematically overstate the
book's confidence, so ``implied_probability`` on every emitted ``ExternalPriceEvent`` is
always the vig-removed value from ``remove_vig``, never the raw one.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from marketlab.adapters.base import Adapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import ExternalPriceEvent
from marketlab.core.probability import american_to_probability, remove_vig
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Stop calling once fewer than this many requests remain - the free tier is small
#: enough that one greedy poll loop could exhaust a month's quota in an afternoon.
_LOW_QUOTA_THRESHOLD = 10


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class TheOddsApiAdapter(Adapter):
    """Sportsbook odds -> vig-free probabilities. Optional: degrades cleanly with no key."""

    name = "sports_odds"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        monthly_request_budget: int = 450,
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or LiveClock()
        self._base_url = base_url.rstrip("/")
        self._api_key = (api_key or "").strip()
        self._monthly_request_budget = monthly_request_budget
        self._requests_used_local = 0
        #: Set from the live ``x-requests-remaining`` header once any request succeeds;
        #: until then, falls back to the locally-tracked budget.
        self._requests_remaining_reported: int | None = None
        self._client = httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "MarketLab research"})
        self._limiter = RateLimiter(rate=2.0, burst=4.0)
        if not self.has_credentials:
            log.warning(
                "odds_no_credentials",
                detail="THE_ODDS_API_KEY not set; sports odds sources degrade to empty results.",
            )

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key)

    @property
    def requests_remaining(self) -> int | None:
        if self._requests_remaining_reported is not None:
            return self._requests_remaining_reported
        return max(self._monthly_request_budget - self._requests_used_local, 0)

    def _quota_exhausted(self) -> bool:
        remaining = self.requests_remaining
        return remaining is not None and remaining < _LOW_QUOTA_THRESHOLD

    async def probe(self) -> SourceHealth:
        if not self.has_credentials:
            return SourceHealth(name=self.name, status=SourceStatus.NO_CREDENTIALS, last_message_at=self._clock.now())
        if self._quota_exhausted():
            return SourceHealth(
                name=self.name,
                status=SourceStatus.DEGRADED,
                detail=f"low quota: {self.requests_remaining} requests remaining",
            )
        try:
            await self._get("/v4/sports", {})
            return SourceHealth(
                name=self.name,
                status=SourceStatus.HEALTHY,
                last_message_at=self._clock.now(),
                detail=f"requests_remaining={self.requests_remaining}",
            )
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("odds_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=0.5, max=10.0),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    )
    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        await self._limiter.acquire(cost=1)
        full_params = dict(params)
        full_params["apiKey"] = self._api_key
        resp = await self._client.get(f"{self._base_url}{path}", params=full_params)
        self._requests_used_local += 1
        remaining_header = resp.headers.get("x-requests-remaining")
        if remaining_header is not None:
            with contextlib.suppress(ValueError):
                self._requests_remaining_reported = int(remaining_header)
        resp.raise_for_status()
        return resp.json()

    async def get_odds(
        self,
        sport: str,
        regions: list[str] | None = None,
        markets: list[str] | None = None,
    ) -> list[ExternalPriceEvent]:
        """Per-outcome vig-free implied probabilities for one sport's current odds.

        Empty list under NO_CREDENTIALS or once tracked quota drops below the safety
        margin - never a request that would run the account's quota to zero.
        """
        if not self.has_credentials:
            log.warning("odds_get_odds_no_credentials", sport=sport)
            return []
        if self._quota_exhausted():
            log.warning("odds_quota_exhausted", sport=sport, remaining=self.requests_remaining)
            return []

        params = {
            "regions": ",".join(regions or ["us"]),
            "markets": ",".join(markets or ["h2h"]),
            "oddsFormat": "american",
        }
        try:
            payload = await self._get(f"/v4/sports/{sport}/odds", params)
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            log.warning("odds_get_odds_failed", sport=sport, error=str(exc))
            return []

        now = self._clock.now()
        events: list[ExternalPriceEvent] = []
        for game in payload or []:
            commence = _parse_iso(game.get("commence_time"))
            for bookmaker in game.get("bookmakers", []) or []:
                book_key = str(bookmaker.get("key", ""))
                updated = _parse_iso(bookmaker.get("last_update")) or commence
                for market in bookmaker.get("markets", []) or []:
                    market_key = str(market.get("key", ""))
                    outcomes = market.get("outcomes", []) or []

                    raw_probabilities = []
                    valid_outcomes = []
                    for outcome in outcomes:
                        price = outcome.get("price")
                        if price is None:
                            continue
                        try:
                            raw_probabilities.append(american_to_probability(int(price)))
                            valid_outcomes.append(outcome)
                        except (ValueError, TypeError):
                            continue
                    if len(raw_probabilities) < 2:
                        continue
                    try:
                        vig_free_probabilities = remove_vig(raw_probabilities)
                    except ValueError:
                        continue

                    for outcome, vig_free_prob in zip(valid_outcomes, vig_free_probabilities, strict=True):
                        symbol = f"{game.get('id', '')}:{market_key}:{outcome.get('name', '')}"
                        events.append(
                            ExternalPriceEvent(
                                event_time=updated or now,
                                published_time=updated,
                                first_seen_time=now,
                                ingested_time=now,
                                source=f"the_odds_api:{book_key}",
                                symbol=symbol,
                                price=Decimal(str(outcome.get("price", 0))),
                                implied_probability=vig_free_prob,
                                venue=book_key,
                            )
                        )
        return events
