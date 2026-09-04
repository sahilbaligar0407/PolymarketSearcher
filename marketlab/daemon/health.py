"""Feed health tracking and the trading kill-switch.

Every adapter/feed the daemon owns reports into one :class:`HealthMonitor`: websocket
message arrivals, REST poll successes/failures, reconnects. ``evaluate()`` turns that
into a per-source :class:`~marketlab.adapters.base.SourceHealth` snapshot, and
``trading_allowed()`` is the single gate the supervisor must consult before dispatching
any intent to a broker -- a *required* source (Kalshi, per ``configs/sources.yaml``)
going stale or down means the system stops trading rather than trading blind, per
``docs/live_safety.md``'s data-staleness policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from marketlab.adapters.base import SourceHealth, SourceStatus
from marketlab.clock import Clock
from marketlab.core.events import Alert
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Default "no message in this long -> STALE" threshold when a source doesn't specify one.
DEFAULT_STALE_AFTER_SECONDS = 120.0
#: A source that has been stale for this multiple of its stale threshold is DOWN, not
#: merely STALE -- distinguishing "hasn't spoken in a bit" from "is not there".
DOWN_MULTIPLIER = 3.0

#: Severity ranking used to decide whether a status transition is a *degradation*
#: (worth an alert) as opposed to a recovery or a lateral move between two "not really
#: healthy" states.
_SEVERITY: dict[SourceStatus, int] = {
    SourceStatus.HEALTHY: 0,
    SourceStatus.DEGRADED: 1,
    SourceStatus.NO_CREDENTIALS: 1,
    SourceStatus.DISABLED: 1,
    SourceStatus.STALE: 2,
    SourceStatus.DOWN: 3,
}


def _severity(status: SourceStatus) -> int:
    return _SEVERITY.get(status, 1)


@dataclass
class _FeedState:
    name: str
    required: bool = False
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS
    registered_at: datetime | None = None
    last_message_at: datetime | None = None
    latency_ms: float | None = None
    error_count: int = 0
    reconnect_count: int = 0
    status: SourceStatus = SourceStatus.DEGRADED
    detail: str = ""


@dataclass
class TradingGate:
    allowed: bool
    detail: str = ""

    def __iter__(self):  # allows `allowed, detail = gate` unpacking
        yield self.allowed
        yield self.detail


class HealthMonitor:
    """Tracks every ingest feed's health and decides whether trading may proceed."""

    def __init__(
        self,
        clock: Clock,
        store: object | None = None,
        *,
        default_stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self._clock = clock
        self._store = store
        self._default_stale = default_stale_after_seconds
        self._feeds: dict[str, _FeedState] = {}

    # ------------------------------------------------------------------
    # registration / recording
    # ------------------------------------------------------------------

    def register(
        self,
        source: str,
        *,
        required: bool = False,
        stale_after_seconds: float | None = None,
    ) -> None:
        """Declare a feed up front so ``required``/threshold are known before any
        message ever arrives (a feed that has never spoken is DEGRADED, not invisible)."""
        existing = self._feeds.get(source)
        if existing is not None:
            existing.required = required
            if stale_after_seconds is not None:
                existing.stale_after_seconds = stale_after_seconds
            return
        self._feeds[source] = _FeedState(
            name=source,
            required=required,
            stale_after_seconds=stale_after_seconds or self._default_stale,
            registered_at=self._clock.now(),
        )

    def _feed(self, source: str) -> _FeedState:
        feed = self._feeds.get(source)
        if feed is None:
            feed = _FeedState(
                name=source, stale_after_seconds=self._default_stale, registered_at=self._clock.now()
            )
            self._feeds[source] = feed
        return feed

    def record_message(self, source: str, latency_ms: float | None = None) -> None:
        feed = self._feed(source)
        feed.last_message_at = self._clock.now()
        if latency_ms is not None:
            feed.latency_ms = latency_ms

    def record_error(self, source: str, exc: BaseException | str) -> None:
        feed = self._feed(source)
        feed.error_count += 1
        feed.detail = str(exc)
        log.warning("feed_error", source=source, error=feed.detail, error_count=feed.error_count)

    def record_reconnect(self, source: str) -> None:
        feed = self._feed(source)
        feed.reconnect_count += 1
        log.warning("feed_reconnect", source=source, reconnect_count=feed.reconnect_count)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def _classify(self, feed: _FeedState, now: datetime) -> SourceStatus:
        if feed.last_message_at is None:
            reg_age = (now - feed.registered_at).total_seconds() if feed.registered_at else 0.0
            down_after = feed.stale_after_seconds * DOWN_MULTIPLIER
            return SourceStatus.DOWN if reg_age > down_after else SourceStatus.DEGRADED
        age = (now - feed.last_message_at).total_seconds()
        if age > feed.stale_after_seconds * DOWN_MULTIPLIER:
            return SourceStatus.DOWN
        if age > feed.stale_after_seconds:
            return SourceStatus.STALE
        return SourceStatus.HEALTHY

    def _write_alert(self, source: str, old: SourceStatus, new: SourceStatus, now: datetime) -> None:
        message = f"{source} degraded: {old.value} -> {new.value}"
        log.warning("feed_degraded", source=source, old_status=old.value, new_status=new.value)
        if self._store is not None and hasattr(self._store, "save_alert"):
            try:
                self._store.save_alert(
                    Alert(
                        timestamp=now,
                        severity="warning" if new is not SourceStatus.DOWN else "critical",
                        component=f"health.{source}",
                        message=message,
                        detail={"old_status": old.value, "new_status": new.value},
                    )
                )
            except Exception as exc:  # noqa: BLE001 - health checks must never crash the daemon
                log.warning("health_alert_write_failed", source=source, error=str(exc))

    def evaluate(self, now: datetime | None = None) -> dict[str, SourceHealth]:
        """Recompute every feed's status, writing an alert on any degradation."""
        now = now or self._clock.now()
        out: dict[str, SourceHealth] = {}
        for name, feed in self._feeds.items():
            new_status = self._classify(feed, now)
            if _severity(new_status) > _severity(feed.status):
                self._write_alert(name, feed.status, new_status, now)
            feed.status = new_status
            out[name] = SourceHealth(
                name=name,
                status=new_status,
                last_message_at=feed.last_message_at,
                latency_ms=feed.latency_ms,
                error_count=feed.error_count,
                reconnect_count=feed.reconnect_count,
                detail=feed.detail,
            )
        return out

    def apply_probe(self, health: SourceHealth) -> None:
        """Fold a one-off adapter ``probe()`` result into this monitor's view.

        Used by ``doctor``/startup wiring so a source that only ever gets probed (not
        polled continuously) still shows up in ``summary()``.
        """
        feed = self._feed(health.name)
        if health.last_message_at is not None:
            feed.last_message_at = health.last_message_at
        if health.latency_ms is not None:
            feed.latency_ms = health.latency_ms
        feed.error_count = max(feed.error_count, health.error_count)
        feed.reconnect_count = max(feed.reconnect_count, health.reconnect_count)
        feed.detail = health.detail
        feed.status = health.status

    # ------------------------------------------------------------------
    # the kill switch
    # ------------------------------------------------------------------

    def trading_allowed(self, now: datetime | None = None) -> tuple[bool, str]:
        """``(False, detail)`` when a *required* source is STALE or DOWN.

        This is the structural guarantee behind "the system stops trading rather than
        trading blind" (``docs/live_safety.md``): the supervisor must call this before
        ever handing an intent to a broker.
        """
        now = now or self._clock.now()
        statuses = self.evaluate(now)
        bad = sorted(
            name
            for name, feed in self._feeds.items()
            if feed.required and statuses[name].status in (SourceStatus.STALE, SourceStatus.DOWN)
        )
        if bad:
            return False, f"required source(s) unhealthy: {', '.join(bad)}"
        return True, ""

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def summary(self, now: datetime | None = None) -> dict[str, dict[str, object]]:
        """Per-source health, JSON-friendly -- feeds `doctor` and the daily report's
        ``data_health`` / ``venue_health`` blocks."""
        statuses = self.evaluate(now)
        out: dict[str, dict[str, object]] = {}
        for name, health in statuses.items():
            feed = self._feeds[name]
            out[name] = {
                "status": health.status.value,
                "required": feed.required,
                "last_message_at": health.last_message_at.isoformat() if health.last_message_at else None,
                "latency_ms": health.latency_ms,
                "error_count": health.error_count,
                "reconnect_count": health.reconnect_count,
                "detail": health.detail,
            }
        return out

    def health_for(self, source: str) -> SourceHealth | None:
        feed = self._feeds.get(source)
        if feed is None:
            return None
        return SourceHealth(
            name=source,
            status=feed.status,
            last_message_at=feed.last_message_at,
            latency_ms=feed.latency_ms,
            error_count=feed.error_count,
            reconnect_count=feed.reconnect_count,
            detail=feed.detail,
        )

    def sources(self) -> list[str]:
        return list(self._feeds.keys())
