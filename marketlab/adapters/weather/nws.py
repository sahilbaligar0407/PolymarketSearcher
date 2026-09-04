"""NWS (``api.weather.gov``) adapter.

Endpoints probed live on 2026-09-04 (all require a descriptive ``User-Agent`` or NWS
returns a degraded/blocked response - there's no API key, but the UA requirement is
functionally identical to SEC's, so this adapter reuses ``settings.secrets.sec_user_agent``
by default rather than inventing a second secret for the same "identify yourself" ask):

* ``GET /points/{lat},{lon}``                         200 - resolves gridId/gridX/gridY
  plus ready-to-use absolute URLs for ``forecast``/``forecastHourly``/``forecastGridData``
* ``GET /gridpoints/{office}/{x},{y}/forecast``        (reached via the ``points`` response)
* ``GET /stations/{id}/observations/latest``           200, current conditions
* ``GET /alerts/active?area={state}``                  200, ``FeatureCollection``

The ``points`` response's URLs are absolute (``https://api.weather.gov/...``); this
adapter fetches them as-is rather than re-deriving paths, since httpx honors an absolute
URL passed to a client with a different ``base_url`` and this is one fewer thing to get
wrong about NWS's gridpoint scheme changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import WeatherEvent
from marketlab.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CityStation:
    key: str
    name: str
    station: str
    lat: float
    lon: float
    #: Best-effort mapping to Kalshi's daily-high-temperature series ticker prefix.
    #: Verify against the live Kalshi markets list before relying on this for routing -
    #: Kalshi's exact ticker scheme is owned by another team and may not match this guess.
    kalshi_series_hint: str = ""


#: The cities Kalshi lists ``KXHIGH*``-style daily high-temperature markets on.
CITY_STATIONS: dict[str, CityStation] = {
    "nyc": CityStation("nyc", "New York City", "KNYC", 40.7128, -74.0060, "KXHIGHNY"),
    "chicago": CityStation("chicago", "Chicago", "KORD", 41.9742, -87.9073, "KXHIGHCHI"),
    "miami": CityStation("miami", "Miami", "KMIA", 25.7959, -80.2870, "KXHIGHMIA"),
    "austin": CityStation("austin", "Austin", "KAUS", 30.1975, -97.6664, "KXHIGHAUS"),
    "denver": CityStation("denver", "Denver", "KDEN", 39.8561, -104.6737, "KXHIGHDEN"),
    "los_angeles": CityStation("los_angeles", "Los Angeles", "KLAX", 33.9425, -118.4081, "KXHIGHLAX"),
    "philadelphia": CityStation("philadelphia", "Philadelphia", "KPHL", 39.8719, -75.2411, "KXHIGHPHIL"),
}


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class NwsAdapter(Adapter):
    """``api.weather.gov`` client. No API key; a descriptive User-Agent is required."""

    name = "nws"

    def __init__(self, *, base_url: str, user_agent: str = "", clock: Clock | None = None) -> None:
        self._clock = clock or LiveClock()
        ua = user_agent.strip() or "MarketLab research (contact: set SEC_USER_AGENT)"
        headers = {"User-Agent": ua, "Accept": "application/geo+json"}
        self._http = HttpAdapter(base_url, name="nws", default_headers=headers, clock=self._clock)
        self._gridpoint_cache: dict[tuple[float, float], dict[str, Any]] = {}

    async def probe(self) -> SourceHealth:
        try:
            await self._http.get_json("/alerts/active", params={"area": "NY"})
            return SourceHealth(name=self.name, status=SourceStatus.HEALTHY, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("nws_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await self._http.close()

    async def get_gridpoint(self, lat: float, lon: float) -> dict[str, Any]:
        """Resolve ``(lat, lon)`` -> gridpoint properties, cached (NWS grids are static)."""
        key = (round(lat, 4), round(lon, 4))
        if key in self._gridpoint_cache:
            return self._gridpoint_cache[key]
        data = await self._http.get_json(f"/points/{lat},{lon}")
        props = data.get("properties", {}) if isinstance(data, dict) else {}
        self._gridpoint_cache[key] = props
        return props

    async def get_forecast(self, lat: float, lon: float) -> dict[str, Any]:
        props = await self.get_gridpoint(lat, lon)
        url = props.get("forecast")
        if not url:
            return {}
        return await self._http.get_json(url)

    async def get_hourly_forecast(self, lat: float, lon: float) -> dict[str, Any]:
        props = await self.get_gridpoint(lat, lon)
        url = props.get("forecastHourly")
        if not url:
            return {}
        return await self._http.get_json(url)

    async def get_latest_observation(self, station_id: str) -> dict[str, Any]:
        return await self._http.get_json(f"/stations/{station_id}/observations/latest")

    async def get_active_alerts(self, area: str | None = None) -> dict[str, Any]:
        params = {"area": area} if area else None
        return await self._http.get_json("/alerts/active", params=params)

    # -- WeatherEvent construction --------------------------------------------------

    async def forecast_events(self, city_key: str, *, hourly: bool = False) -> list[WeatherEvent]:
        """Forecast periods for a configured city, as ``WeatherEvent``s.

        ``first_seen_time`` is the injected Clock's ``now()``; ``forecast_for``/``issued_at``
        carry the forecast's own timestamps so a backtest can distinguish "when this
        forecast said the high would occur" from "when we actually saw the forecast."
        """
        city = CITY_STATIONS.get(city_key)
        if city is None:
            raise KeyError(f"unknown city key: {city_key!r}; known: {sorted(CITY_STATIONS)}")
        now = self._clock.now()
        payload = await (self.get_hourly_forecast(city.lat, city.lon) if hourly else self.get_forecast(city.lat, city.lon))
        props = payload.get("properties", {}) if isinstance(payload, dict) else {}
        office = props.get("gridId", "")
        issued_at = _parse_iso(props.get("updated")) or now
        events: list[WeatherEvent] = []
        for period in props.get("periods", []) or []:
            start_time = _parse_iso(period.get("startTime"))
            temp = period.get("temperature")
            variable = "temperature_forecast_hourly" if hourly else (
                "temperature_forecast_day" if period.get("isDaytime") else "temperature_forecast_night"
            )
            events.append(
                WeatherEvent(
                    event_time=start_time or now,
                    published_time=issued_at,
                    first_seen_time=now,
                    ingested_time=now,
                    source="nws_forecast",
                    station=city.station,
                    office=office,
                    variable=variable,
                    value=Decimal(str(temp)) if temp is not None else None,
                    forecast_for=start_time,
                    issued_at=issued_at,
                )
            )
        return events

    async def observation_event(self, city_key: str) -> WeatherEvent | None:
        """Latest observed temperature for a configured city, or ``None`` if unavailable."""
        city = CITY_STATIONS.get(city_key)
        if city is None:
            raise KeyError(f"unknown city key: {city_key!r}; known: {sorted(CITY_STATIONS)}")
        now = self._clock.now()
        data = await self.get_latest_observation(city.station)
        props = data.get("properties", {}) if isinstance(data, dict) else {}
        temp = (props.get("temperature") or {}).get("value")
        if temp is None:
            return None
        timestamp = _parse_iso(props.get("timestamp"))
        return WeatherEvent(
            event_time=timestamp or now,
            published_time=timestamp,
            first_seen_time=now,
            ingested_time=now,
            source="nws_observation",
            station=city.station,
            variable="temperature_c",
            value=Decimal(str(temp)),
            issued_at=timestamp,
        )

    async def alert_events(self, area: str | None = None) -> list[WeatherEvent]:
        """Active alerts as coarse ``WeatherEvent``s (``variable="alert"``, ``value=None``)."""
        now = self._clock.now()
        data = await self.get_active_alerts(area)
        features = data.get("features", []) if isinstance(data, dict) else []
        events: list[WeatherEvent] = []
        for feature in features:
            props = feature.get("properties", {})
            onset = _parse_iso(props.get("onset"))
            events.append(
                WeatherEvent(
                    event_time=onset or now,
                    published_time=_parse_iso(props.get("sent")),
                    first_seen_time=now,
                    ingested_time=now,
                    source="nws_alert",
                    station=props.get("id", ""),
                    office=props.get("senderName", ""),
                    variable=f"alert:{props.get('event', 'unknown')}",
                    forecast_for=onset,
                    issued_at=_parse_iso(props.get("sent")),
                )
            )
        return events
