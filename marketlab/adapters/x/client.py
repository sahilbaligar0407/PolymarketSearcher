"""Official X API v2 client.

No ``X_BEARER_TOKEN`` was available in this environment, so the live endpoints below are
not independently probed here; X's v2 REST surface (recent search, user lookup, user
tweet timeline) is documented and stable, and every method here degrades to an empty
list rather than firing a request guaranteed to 401 when no token is configured -
``probe()`` reports ``NO_CREDENTIALS`` with one WARN log, per the contract.

X is usage-priced. ``monthly_request_budget`` (from ``configs/sources.yaml``'s
``sources.x.monthly_request_budget``) caps the number of calls this process will make;
once exhausted, calls short-circuit to empty results and log a WARN instead of
continuing to spend against the account's quota. There is no scraping fallback -
that's explicitly out of scope.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import SocialEvent, SourceClass
from marketlab.logging import get_logger

log = get_logger(__name__)

_X_BASE = "https://api.twitter.com/2"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class XAdapter(Adapter):
    """X API v2 client, gated by a bearer token and a client-side monthly cost budget."""

    name = "x"

    def __init__(
        self,
        *,
        bearer_token: str,
        base_url: str = _X_BASE,
        monthly_request_budget: int = 1000,
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or LiveClock()
        self._bearer_token = (bearer_token or "").strip()
        self._budget = monthly_request_budget
        self._requests_used = 0
        headers = {"Authorization": f"Bearer {self._bearer_token}"} if self._bearer_token else {}
        self._http = HttpAdapter(base_url, name="x", default_headers=headers, clock=self._clock)
        if not self._bearer_token:
            log.warning("x_no_credentials", detail="X_BEARER_TOKEN not set; X sources degrade to empty results.")

    @property
    def has_credentials(self) -> bool:
        return bool(self._bearer_token)

    @property
    def requests_remaining(self) -> int:
        return max(self._budget - self._requests_used, 0)

    async def probe(self) -> SourceHealth:
        if not self.has_credentials:
            return SourceHealth(name=self.name, status=SourceStatus.NO_CREDENTIALS, last_message_at=self._clock.now())
        if self.requests_remaining <= 0:
            return SourceHealth(
                name=self.name, status=SourceStatus.DEGRADED, detail="monthly request budget exhausted"
            )
        try:
            await self._get("/tweets/search/recent", {"query": "the", "max_results": 10})
            return SourceHealth(name=self.name, status=SourceStatus.HEALTHY, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("x_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await self._http.close()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.has_credentials:
            return {}
        if self.requests_remaining <= 0:
            log.warning("x_cost_budget_exhausted", path=path, budget=self._budget)
            return {}
        self._requests_used += 1
        return await self._http.get_json(path, params=params)

    def _tweet_to_event(self, tweet: dict[str, Any], author_username: str = "") -> SocialEvent:
        now = self._clock.now()
        created_at = _parse_iso(tweet.get("created_at"))
        tweet_id = str(tweet.get("id", ""))
        url = f"https://x.com/{author_username}/status/{tweet_id}" if author_username and tweet_id else ""
        return SocialEvent(
            event_time=created_at or now,
            published_time=created_at,
            first_seen_time=now,
            ingested_time=now,
            source="x",
            source_class=SourceClass.SOCIAL_VERIFIED,
            post_id=tweet_id,
            platform="x",
            author=author_username,
            text=str(tweet.get("text", "")),
            url=url,
        )

    async def recent_search(
        self, query: str, since_id: str | None = None, max_results: int = 25
    ) -> list[SocialEvent]:
        """``GET /2/tweets/search/recent``. Empty list under NO_CREDENTIALS or exhausted budget."""
        params: dict[str, Any] = {
            "query": query,
            "max_results": max(10, min(max_results, 100)),
            "tweet.fields": "created_at,author_id",
        }
        if since_id:
            params["since_id"] = since_id
        data = await self._get("/tweets/search/recent", params)
        tweets = data.get("data", []) or []
        return [self._tweet_to_event(t) for t in tweets]

    async def user_timeline(
        self, username: str, max_results: int = 25, since_id: str | None = None
    ) -> list[SocialEvent]:
        """Recent tweets from one user. Costs two API calls (username -> id, then timeline)."""
        user_payload = await self._get(f"/users/by/username/{username}", {})
        user_id = (user_payload.get("data") or {}).get("id")
        if not user_id:
            return []
        params: dict[str, Any] = {"max_results": max(5, min(max_results, 100)), "tweet.fields": "created_at"}
        if since_id:
            params["since_id"] = since_id
        data = await self._get(f"/users/{user_id}/tweets", params)
        tweets = data.get("data", []) or []
        return [self._tweet_to_event(t, author_username=username) for t in tweets]
