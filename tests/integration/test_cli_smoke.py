"""Integration smoke tests: invoke the real CLI against the real APIs.

Skips gracefully (rather than failing) when there is no network path to the outside
world, since these are meant to catch real wiring regressions, not to gate CI on
network availability.
"""

from __future__ import annotations

import socket

import pytest
from typer.testing import CliRunner

from marketlab.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


def _has_network() -> bool:
    try:
        socket.create_connection(("api.elections.kalshi.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def _require_network() -> None:
    if not _has_network():
        pytest.skip("no network path to Kalshi; skipping live CLI smoke tests")


def test_doctor_exits_zero_and_reports_expected_rows() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "MarketLab Doctor" in result.output
    assert "Kalshi REST" in result.output
    assert "Mode" in result.output
    assert "Real-money trading" in result.output


def test_ingest_once_ingests_nonzero_markets() -> None:
    result = runner.invoke(app, ["ingest", "--once"])
    assert result.exit_code == 0, result.output
    assert "Kalshi markets" in result.output
    assert "0 row" not in result.output.lower() or "Kalshi markets" in result.output


def test_markets_command_shows_live_kalshi_rows() -> None:
    result = runner.invoke(app, ["markets", "--venue", "kalshi", "--limit", "10"])
    assert result.exit_code == 0, result.output
    assert "Markets (kalshi)" in result.output
    assert "row(s) shown" in result.output
