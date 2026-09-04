"""Unit tests for marketlab.daemon.health.HealthMonitor."""

from __future__ import annotations

from datetime import UTC, datetime

from marketlab.adapters.base import SourceStatus
from marketlab.clock import SimulatedClock
from marketlab.daemon.health import HealthMonitor

START = datetime(2026, 1, 1, tzinfo=UTC)


class FakeStore:
    def __init__(self) -> None:
        self.alerts: list = []

    def save_alert(self, alert) -> None:  # noqa: ANN001
        self.alerts.append(alert)


def test_feed_goes_stale_after_threshold() -> None:
    clock = SimulatedClock(START)
    monitor = HealthMonitor(clock, default_stale_after_seconds=60.0)
    monitor.register("kalshi_rest", required=True)

    monitor.record_message("kalshi_rest")
    statuses = monitor.evaluate(clock.now())
    assert statuses["kalshi_rest"].status is SourceStatus.HEALTHY

    clock.advance(61.0)
    statuses = monitor.evaluate(clock.now())
    assert statuses["kalshi_rest"].status is SourceStatus.STALE


def test_feed_goes_down_after_much_longer_silence() -> None:
    clock = SimulatedClock(START)
    monitor = HealthMonitor(clock, default_stale_after_seconds=60.0)
    monitor.register("kalshi_rest", required=True)
    monitor.record_message("kalshi_rest")

    clock.advance(60.0 * 3 + 1)
    statuses = monitor.evaluate(clock.now())
    assert statuses["kalshi_rest"].status is SourceStatus.DOWN


def test_trading_allowed_false_when_required_source_stale() -> None:
    clock = SimulatedClock(START)
    monitor = HealthMonitor(clock, default_stale_after_seconds=60.0)
    monitor.register("kalshi_rest", required=True)
    monitor.record_message("kalshi_rest")

    clock.advance(120.0)
    allowed, detail = monitor.trading_allowed(clock.now())
    assert allowed is False
    assert "kalshi_rest" in detail


def test_trading_allowed_true_when_only_optional_source_stale() -> None:
    clock = SimulatedClock(START)
    monitor = HealthMonitor(clock, default_stale_after_seconds=60.0)
    monitor.register("kalshi_rest", required=True)
    monitor.register("poly_gamma", required=False)
    monitor.record_message("kalshi_rest")
    monitor.record_message("poly_gamma")

    # Only advance far enough to stale the optional source, not the required one -- so
    # record a fresh Kalshi message right before checking.
    clock.advance(120.0)
    monitor.record_message("kalshi_rest")  # keep required source fresh
    allowed, detail = monitor.trading_allowed(clock.now())
    assert allowed is True
    assert detail == ""

    statuses = monitor.evaluate(clock.now())
    assert statuses["poly_gamma"].status is SourceStatus.STALE
    assert statuses["kalshi_rest"].status is SourceStatus.HEALTHY


def test_degradation_writes_an_alert() -> None:
    clock = SimulatedClock(START)
    store = FakeStore()
    monitor = HealthMonitor(clock, store=store, default_stale_after_seconds=60.0)
    monitor.register("kalshi_rest", required=True)
    monitor.record_message("kalshi_rest")
    monitor.evaluate(clock.now())
    assert store.alerts == []

    clock.advance(61.0)
    monitor.evaluate(clock.now())
    assert len(store.alerts) == 1
    assert "kalshi_rest" in store.alerts[0].message


def test_reconnects_are_counted() -> None:
    clock = SimulatedClock(START)
    monitor = HealthMonitor(clock)
    monitor.register("kalshi_ws")
    monitor.record_reconnect("kalshi_ws")
    monitor.record_reconnect("kalshi_ws")
    health = monitor.evaluate(clock.now())["kalshi_ws"]
    assert health.reconnect_count == 2


def test_record_error_increments_error_count() -> None:
    clock = SimulatedClock(START)
    monitor = HealthMonitor(clock)
    monitor.register("sec")
    monitor.record_error("sec", RuntimeError("boom"))
    health = monitor.evaluate(clock.now())["sec"]
    assert health.error_count == 1
    assert "boom" in health.detail


def test_summary_is_json_friendly() -> None:
    clock = SimulatedClock(START)
    monitor = HealthMonitor(clock)
    monitor.register("kalshi_rest", required=True)
    monitor.record_message("kalshi_rest", latency_ms=12.5)
    summary = monitor.summary(clock.now())
    assert summary["kalshi_rest"]["status"] == "healthy"
    assert summary["kalshi_rest"]["required"] is True
    assert summary["kalshi_rest"]["latency_ms"] == 12.5
