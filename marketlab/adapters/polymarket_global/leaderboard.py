"""Leaderboard adapter: dynamic Polymarket trader discovery (read-only).

Queries both Polymarket leaderboard surfaces and dedupes by wallet:

* the official leaderboard (``data-api.polymarket.com/v1/leaderboard``), which accepts
  ``category`` + ``period`` + ``metric`` and is the richer of the two (usernames, X
  handles, verified badges), and
* the legacy leaderboard (``lb-api.polymarket.com``), which only accepts ``window`` (no
  category) but is a second, independent source of top wallets - useful because its
  ranking sometimes surfaces wallets the official board does not.

Live-probe findings (2026-09-04), recorded here because they change how this adapter
sweeps combinations:

* Official endpoint: ``category`` is validated server-side - an unknown value (e.g.
  ``bogus``) returns HTTP 400 ``{"error": "invalid category parameter"}``. The eight
  categories this module sweeps (``overall, crypto, sports, politics, economics,
  finance, weather, tech``) were all confirmed to return HTTP 200 with genuinely
  different top wallets per category (i.e. filtering really happens).
* Official endpoint: ``period`` and ``metric`` do **not** appear to change the
  response - ``period=day``, ``period=week``, ``period=month``, ``period=all`` and
  ``metric=pnl``, ``metric=vol``, ``metric=volume`` were all observed to return the
  identical top-N list (same ranks, same pnl figures) for a fixed category. An invalid
  value for either does not error either - it silently falls back to the same result.
  We still pass the requested ``period``/``metric`` through (in case Polymarket starts
  honoring them, and so a snapshot row records what was *asked for*), and we record
  every ``LeaderboardRow`` with the metric/period metadata the caller requested, but
  callers should not expect distinct numbers for `pnl` by period from this endpoint
  today - the underlying `pnl` figure is all-time regardless of `period`.
* Legacy endpoint: ``window`` is validated and *does* change the ranking - only
  ``1d``, ``7d``, ``30d`` and ``all`` were confirmed to return HTTP 200 with different
  top wallets per window; single-letter windows (``d``, ``w``, ``m``) return HTTP 400.
  It has no category dimension and no verified/X-handle fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock
from marketlab.core.instruments import Category
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Categories confirmed (2026-09-04) to be accepted by the official leaderboard.
OFFICIAL_CATEGORIES: tuple[str, ...] = (
    "overall", "crypto", "sports", "politics", "economics", "finance", "weather", "tech",
)
#: Periods requested; see module docstring - the API currently ignores this dimension.
OFFICIAL_PERIODS: tuple[str, ...] = ("day", "week", "month", "all")

#: The official host accepts `period`/`metric` and ignores them. These are the single
#: values `snapshot_all` sends so the request is well-formed; they carry no meaning, and
#: the resulting rows are labelled "ignored" rather than with these values.
OFFICIAL_CANONICAL_PERIOD = "all"
OFFICIAL_CANONICAL_METRIC = "pnl"
#: Metrics requested; see module docstring - `vol` is only meaningful for `overall`
#: (and today isn't actually honored either), but we still sweep it for `overall`.
OFFICIAL_METRICS: tuple[str, ...] = ("pnl", "vol")

#: Windows confirmed (2026-09-04) to be accepted by the legacy leaderboard.
LEGACY_WINDOWS: tuple[str, ...] = ("1d", "7d", "30d", "all")

_CATEGORY_TO_ENUM: dict[str, Category] = {
    "overall": Category.OTHER,
    "crypto": Category.CRYPTO,
    "sports": Category.SPORTS,
    "politics": Category.POLITICS,
    "economics": Category.ECONOMICS,
    "finance": Category.FINANCE,
    "weather": Category.WEATHER,
    "tech": Category.TECH,
}


class LeaderboardRow(BaseModel):
    """One dated snapshot row: one wallet's rank in one category/period/metric combo."""

    model_config = ConfigDict(frozen=True)

    snapshot_time: datetime
    source: str  # "official" | "legacy"
    category: Category
    category_raw: str = ""
    period: str = ""
    metric: str = ""
    rank: int
    wallet: str
    username: str = ""
    x_username: str = ""
    verified: bool = False
    pnl: Decimal | None = None
    volume: Decimal | None = None


def _to_decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


class LeaderboardAdapter(Adapter):
    """Read-only client sweeping both Polymarket leaderboard surfaces."""

    name = "poly_leaderboard"

    def __init__(
        self,
        official_base_url: str,
        legacy_base_url: str,
        clock: Clock,
        *,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 10.0,
        official_http: HttpAdapter | None = None,
        legacy_http: HttpAdapter | None = None,
    ) -> None:
        self._clock = clock
        # `settings.sources.poly_leaderboard` is the *full* endpoint URL
        # ("https://data-api.polymarket.com/v1/leaderboard"), but httpx's base_url +
        # empty-path joining round-trips through a 301 (it normalizes to a trailing
        # slash, which the origin then redirects away since `HttpAdapter` does not
        # follow redirects). Split origin from path ourselves so the real request path
        # is always non-empty and no redirect is ever involved.
        split = urlsplit(official_base_url)
        origin = f"{split.scheme}://{split.netloc}"
        self._official_path = split.path or "/v1/leaderboard"
        self._official = official_http or HttpAdapter(
            origin, name="poly_leaderboard_official", timeout=timeout,
            rate_limiter=rate_limiter, clock=clock,
        )
        self._legacy = legacy_http or HttpAdapter(
            legacy_base_url, name="poly_leaderboard_legacy", timeout=timeout,
            rate_limiter=rate_limiter, clock=clock,
        )

    async def probe(self) -> SourceHealth:
        try:
            await self.get_official(category="overall", period="week", metric="pnl", limit=1)
        except Exception as exc:
            log.warning("poly_leaderboard_probe_failed", error=str(exc))
            health = self._official.health()
            return health.model_copy(update={"status": SourceStatus.DOWN, "detail": str(exc)})
        return self._official.health()

    async def close(self) -> None:
        await self._official.close()
        await self._legacy.close()

    def health(self) -> SourceHealth:
        return self._official.health()

    async def get_official(
        self, *, category: str = "overall", period: str = "week", metric: str = "pnl", limit: int = 50
    ) -> list[dict[str, Any]]:
        data = await self._official.get_json(
            self._official_path,
            params={"category": category, "period": period, "metric": metric, "limit": limit},
        )
        return data if isinstance(data, list) else []

    async def get_legacy_profit(self, *, window: str = "30d", limit: int = 50) -> list[dict[str, Any]]:
        data = await self._legacy.get_json("/profit", params={"window": window, "limit": limit})
        return data if isinstance(data, list) else []

    async def get_legacy_volume(self, *, window: str = "30d", limit: int = 50) -> list[dict[str, Any]]:
        data = await self._legacy.get_json("/volume", params={"window": window, "limit": limit})
        return data if isinstance(data, list) else []

    def _rows_from_official(
        self, raw_rows: Sequence[dict[str, Any]], *, category: str, period: str, metric: str, now: datetime
    ) -> list[LeaderboardRow]:
        rows: list[LeaderboardRow] = []
        for r in raw_rows:
            try:
                rank = int(r.get("rank", 0))
            except (TypeError, ValueError):
                continue
            rows.append(
                LeaderboardRow(
                    snapshot_time=now,
                    source="official",
                    category=_CATEGORY_TO_ENUM.get(category, Category.OTHER),
                    category_raw=category,
                    period=period,
                    metric=metric,
                    rank=rank,
                    wallet=str(r.get("proxyWallet", "")),
                    username=str(r.get("userName", "")),
                    x_username=str(r.get("xUsername", "")),
                    verified=bool(r.get("verifiedBadge", False)),
                    pnl=_to_decimal_or_none(r.get("pnl")),
                    volume=_to_decimal_or_none(r.get("vol")),
                )
            )
        return rows

    def _rows_from_legacy(
        self, raw_rows: Sequence[dict[str, Any]], *, metric: str, window: str, now: datetime
    ) -> list[LeaderboardRow]:
        rows: list[LeaderboardRow] = []
        for idx, r in enumerate(raw_rows, start=1):
            amount = _to_decimal_or_none(r.get("amount"))
            rows.append(
                LeaderboardRow(
                    snapshot_time=now,
                    source="legacy",
                    category=Category.OTHER,  # legacy board has no category dimension
                    category_raw="overall",
                    period=window,
                    metric=metric,
                    rank=idx,
                    wallet=str(r.get("proxyWallet", "")),
                    username=str(r.get("name") or r.get("pseudonym") or ""),
                    pnl=amount if metric == "pnl" else None,
                    volume=amount if metric == "vol" else None,
                )
            )
        return rows

    async def snapshot_all(self, *, limit_per_board: int = 50) -> list[LeaderboardRow]:
        """One dated snapshot across every working category/period/metric combo (top
        ``limit_per_board`` each) on both leaderboard surfaces, deduped by wallet
        within each (category, period, metric) bucket - not globally, since the whole
        point is to preserve per-category rank context for trader discovery.

        Degrades gracefully: an individual combo that 400s/404s is logged and skipped
        rather than aborting the whole snapshot.
        """
        now = self._clock.now()
        rows: list[LeaderboardRow] = []

        # The official host validates `category` but SILENTLY IGNORES `period` and
        # `metric` - every value returns byte-identical rows (verified 2026-09-04, see
        # docs/FINDINGS.md #11). So we sweep categories only.
        #
        # Sweeping the ignored dimensions would store the same board 4-8 times per
        # category as though they were independent dated snapshots, which is a fabricated
        # sample size - the exact self-deception this project exists to prevent. The rows
        # are labelled period="ignored"/metric="ignored" so nothing downstream can mistake
        # them for a genuine time series. The real time dimension comes from the legacy
        # host below, whose `window` parameter is honoured.
        for category in OFFICIAL_CATEGORIES:
            try:
                raw = await self.get_official(
                    category=category,
                    period=OFFICIAL_CANONICAL_PERIOD,
                    metric=OFFICIAL_CANONICAL_METRIC,
                    limit=limit_per_board,
                )
            except Exception as exc:
                log.warning(
                    "poly_leaderboard_official_combo_failed",
                    category=category, error=str(exc),
                )
                continue
            rows.extend(
                self._rows_from_official(
                    raw, category=category, period="ignored", metric="ignored", now=now
                )
            )

        for window in LEGACY_WINDOWS:
            try:
                profit_raw = await self.get_legacy_profit(window=window, limit=limit_per_board)
            except Exception as exc:
                log.warning("poly_leaderboard_legacy_profit_failed", window=window, error=str(exc))
                profit_raw = []
            rows.extend(self._rows_from_legacy(profit_raw, metric="pnl", window=window, now=now))

            try:
                volume_raw = await self.get_legacy_volume(window=window, limit=limit_per_board)
            except Exception as exc:
                log.warning("poly_leaderboard_legacy_volume_failed", window=window, error=str(exc))
                volume_raw = []
            rows.extend(self._rows_from_legacy(volume_raw, metric="vol", window=window, now=now))

        return rows

    @staticmethod
    def dedupe_wallets(rows: Sequence[LeaderboardRow]) -> dict[str, LeaderboardRow]:
        """Best (lowest-rank) row seen per wallet across an entire snapshot - used by
        the discovery service to decide which wallets are worth a full history pull."""
        best: dict[str, LeaderboardRow] = {}
        for row in rows:
            if not row.wallet:
                continue
            current = best.get(row.wallet)
            if current is None or row.rank < current.rank:
                best[row.wallet] = row
        return best
