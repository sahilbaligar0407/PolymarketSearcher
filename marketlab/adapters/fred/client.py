"""FRED / ALFRED client.

Probed live on 2026-09-04 (no key available in this environment):

* ``GET api.stlouisfed.org/fred/series/observations`` without ``api_key`` -> HTTP 400,
  ``{"error_code":400,"error_message":"Bad Request.  Variable api_key is not set. ..."}``
  Confirms the shape of the (only) failure mode without credentials; this adapter never
  makes that request when no key is configured - it short-circuits to an empty result
  and a ``NO_CREDENTIALS`` health status instead of paying for a guaranteed-400 round trip.

With a key, ``/fred/series/observations`` (used by both :meth:`get_series` and
:meth:`get_series_vintages`) and ``/fred/releases/dates`` are FRED's documented,
stable endpoints - not independently reachable here without credentials, but their
request/response shape is part of FRED's public API contract.

**Point-in-time discipline**: FRED's default query (no ``realtime_start``/``realtime_end``)
still returns each observation's *own* ``realtime_start`` - the date that particular
value's revision became the officially published one - not "today". That field is used
directly as ``EconomicEvent.vintage_date`` on every event this adapter produces. A
backtest that used a later vintage than the ALFRED history says was known at decision
time would be look-ahead; the analytics layer checks this against ``vintage_date``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import EconomicEvent
from marketlab.logging import get_logger

log = get_logger(__name__)


def _parse_fred_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_fred_value(value: str | None) -> Decimal | None:
    if value in (None, "", "."):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


class FredAdapter(Adapter):
    """St. Louis Fed FRED/ALFRED client. Degrades cleanly with no ``FRED_API_KEY``."""

    name = "fred"

    def __init__(self, *, base_url: str, api_key: str, clock: Clock | None = None) -> None:
        self._clock = clock or LiveClock()
        self._api_key = (api_key or "").strip()
        self._http = HttpAdapter(
            base_url, name="fred", default_headers={"User-Agent": "MarketLab research"}, clock=self._clock
        )
        if not self._api_key:
            log.warning("fred_no_credentials", detail="FRED_API_KEY not set; FRED sources degrade to empty results.")

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key)

    async def probe(self) -> SourceHealth:
        if not self.has_credentials:
            return SourceHealth(name=self.name, status=SourceStatus.NO_CREDENTIALS, last_message_at=self._clock.now())
        try:
            await self._http.get_json(
                "/series/observations",
                params={"series_id": "UNRATE", "api_key": self._api_key, "file_type": "json", "limit": 1},
            )
            return SourceHealth(name=self.name, status=SourceStatus.HEALTHY, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("fred_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await self._http.close()

    def _params(self, **kwargs: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"api_key": self._api_key, "file_type": "json"}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params

    def _observations_to_events(
        self, series_id: str, payload: dict[str, Any], release_name: str = ""
    ) -> list[EconomicEvent]:
        now = self._clock.now()
        events: list[EconomicEvent] = []
        for obs in payload.get("observations", []) or []:
            obs_date = _parse_fred_date(obs.get("date"))
            vintage_date = _parse_fred_date(obs.get("realtime_start"))
            events.append(
                EconomicEvent(
                    event_time=obs_date or now,
                    published_time=vintage_date,
                    first_seen_time=now,
                    ingested_time=now,
                    source="fred",
                    series_id=series_id,
                    value=_parse_fred_value(obs.get("value")),
                    observation_date=obs_date,
                    vintage_date=vintage_date,
                    release_name=release_name,
                )
            )
        return events

    async def get_series(
        self,
        series_id: str,
        *,
        realtime_start: date | None = None,
        realtime_end: date | None = None,
        observation_start: date | None = None,
        observation_end: date | None = None,
        limit: int | None = None,
    ) -> list[EconomicEvent]:
        """Observations for one series, each stamped with its own ALFRED ``vintage_date``."""
        if not self.has_credentials:
            log.warning("fred_get_series_no_credentials", series_id=series_id)
            return []
        params = self._params(
            series_id=series_id,
            realtime_start=realtime_start.isoformat() if realtime_start else None,
            realtime_end=realtime_end.isoformat() if realtime_end else None,
            observation_start=observation_start.isoformat() if observation_start else None,
            observation_end=observation_end.isoformat() if observation_end else None,
            limit=limit,
        )
        payload = await self._http.get_json("/series/observations", params=params)
        return self._observations_to_events(series_id, payload)

    async def get_series_vintages(
        self,
        series_id: str,
        *,
        realtime_start: date,
        realtime_end: date,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> list[EconomicEvent]:
        """ALFRED point-in-time query: the value(s) in effect during ``[realtime_start,
        realtime_end]``, as they were actually known at that time - not today's revised value.
        """
        return await self.get_series(
            series_id,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    async def get_releases_calendar(
        self,
        *,
        realtime_start: date | None = None,
        realtime_end: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Scheduled economic release dates (``/fred/releases/dates``)."""
        if not self.has_credentials:
            log.warning("fred_get_releases_no_credentials")
            return []
        params = self._params(
            realtime_start=realtime_start.isoformat() if realtime_start else None,
            realtime_end=realtime_end.isoformat() if realtime_end else None,
            limit=limit,
        )
        payload = await self._http.get_json("/releases/dates", params=params)
        return payload.get("release_dates", []) or []
