from __future__ import annotations

import inspect
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import marketlab.signals.social as social
from marketlab.signals.social import (
    DEFAULT_PUBLIC_FIGURES,
    EVENT_STUDY_HORIZONS,
    PublicFigureConfig,
    StatementClassification,
    classify_statement,
    event_study_windows,
    load_public_figures,
    market_hours_state,
)

EASTERN = ZoneInfo("America/New_York")


def _et(y, m, d, h, minute=0) -> datetime:
    return datetime(y, m, d, h, minute, tzinfo=EASTERN)


# ---------------------------------------------------------------------------
# Classification content
# ---------------------------------------------------------------------------


def test_tariff_statement_classified_as_tariff():
    text = "We are placing a 25% tariff on all steel imports from China effective immediately."
    result = classify_statement(text, _et(2026, 3, 10, 11, 0))
    assert result.action_type == "tariff"
    assert result.policy_topic == "trade"


def test_sanctions_statement_classified_as_sanctions():
    text = "The administration announced new sanctions targeting the regime's oil exports."
    result = classify_statement(text, _et(2026, 3, 10, 11, 0))
    assert result.action_type == "sanctions"


def test_praise_statement_classified_as_praise():
    text = "Doing a fantastic job leading the company, tremendous work, thank you!"
    result = classify_statement(text, _et(2026, 3, 10, 11, 0))
    assert result.action_type == "praise"
    assert result.sentiment > 0


def test_criticism_statement_classified_as_criticism():
    text = "This has been a total disaster and a terrible job by management, very disappointed."
    result = classify_statement(text, _et(2026, 3, 10, 11, 0))
    assert result.action_type == "criticism"
    assert result.sentiment < 0


def test_mentioned_tickers_and_sector():
    text = "NVIDIA CORP shares jumped after new semiconductor export rules were announced."
    ticker_map = {"NVDA": "NVIDIA CORP"}
    result = classify_statement(text, _et(2026, 3, 10, 11, 0), ticker_map=ticker_map)
    assert "NVDA" in result.mentioned_tickers
    assert result.sector == "technology"


def test_no_credentials_style_default_figures_available():
    figures = load_public_figures(None)
    assert "trump" in figures
    assert isinstance(figures["trump"], PublicFigureConfig)
    assert figures["trump"].enabled


def test_load_public_figures_from_config_block():
    cfg = {
        "public_figures": {
            "powell": {"enabled": True, "sources": ["gdelt"], "keywords": ["rates", "inflation"]}
        }
    }
    figures = load_public_figures(cfg)
    assert "powell" in figures
    assert figures["powell"].sources == ("gdelt",)
    assert figures["powell"].keywords == ("rates", "inflation")


def test_default_public_figures_include_trump_key():
    assert "trump" in DEFAULT_PUBLIC_FIGURES


# ---------------------------------------------------------------------------
# market_hours_state, including the DST boundary
# ---------------------------------------------------------------------------


def test_market_hours_state_open():
    assert market_hours_state(_et(2026, 6, 15, 11, 0)) == "open"  # Monday, mid-day


def test_market_hours_state_premarket():
    assert market_hours_state(_et(2026, 6, 15, 7, 0)) == "premarket"


def test_market_hours_state_afterhours():
    assert market_hours_state(_et(2026, 6, 15, 17, 0)) == "afterhours"


def test_market_hours_state_closed_overnight():
    assert market_hours_state(_et(2026, 6, 15, 2, 0)) == "closed"


def test_market_hours_state_closed_weekend():
    # 2026-06-14 is a Sunday.
    assert market_hours_state(_et(2026, 6, 14, 11, 0)) == "closed"


def test_market_hours_state_requires_tz_aware():
    with pytest.raises(ValueError):
        market_hours_state(datetime(2026, 6, 15, 11, 0))


def test_market_hours_state_across_dst_boundary():
    # US DST starts 2026-03-08 (spring forward, second Sunday of March). Compare a
    # weekday just before the switch (Friday 2026-03-06, EST) against a weekday just
    # after (Monday 2026-03-09, EDT) at the same UTC instant, to prove the UTC->Eastern
    # conversion via zoneinfo gets the offset right on both sides of the boundary.
    before_dst = datetime(2026, 3, 6, 14, 0, tzinfo=ZoneInfo("UTC")).astimezone(EASTERN)
    after_dst = datetime(2026, 3, 9, 14, 0, tzinfo=ZoneInfo("UTC")).astimezone(EASTERN)
    # UTC 14:00 is 09:00 EST before the switch, 10:00 EDT after.
    assert before_dst.utcoffset().total_seconds() / 3600 == -5
    assert after_dst.utcoffset().total_seconds() / 3600 == -4
    assert market_hours_state(before_dst) == "premarket"
    assert market_hours_state(after_dst) == "open"


# ---------------------------------------------------------------------------
# event_study_windows
# ---------------------------------------------------------------------------


def test_event_study_horizons_constant():
    assert EVENT_STUDY_HORIZONS == ("1m", "5m", "15m", "1h", "close", "1d", "3d", "5d")


def test_event_study_windows_mapping():
    windows = event_study_windows()
    assert set(windows) == set(EVENT_STUDY_HORIZONS)
    assert windows["close"] is None
    assert windows["1h"].total_seconds() == 3600


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def test_novelty_lower_for_repeated_statement():
    text = "We are placing a 25% tariff on all steel imports from China effective immediately."
    result = classify_statement(text, _et(2026, 3, 10, 11, 0), recent_texts=(text,))
    assert result.novelty < 0.3


def test_novelty_high_with_no_recent_context():
    text = "We are placing a 25% tariff on all steel imports from China effective immediately."
    result = classify_statement(text, _et(2026, 3, 10, 11, 0))
    assert result.novelty == 1.0


# ---------------------------------------------------------------------------
# The hard invariant: nothing here produces a trade direction.
# ---------------------------------------------------------------------------

_FORBIDDEN_NAMES = {"buy", "sell", "side", "action", "direction", "order", "trade_direction"}


def test_no_function_returns_a_trade_direction():
    """Static sweep: no public function/dataclass field in this module is named or
    shaped like a trade direction. This module produces event-study features only."""
    field_names = set(StatementClassification.__dataclass_fields__.keys())
    for forbidden in _FORBIDDEN_NAMES:
        assert forbidden not in field_names, f"found forbidden field name: {forbidden}"

    for name, obj in vars(social).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if obj.__module__ != social.__name__:
            continue
        lowered_name = name.lower()
        assert not any(f == lowered_name or lowered_name.startswith(f + "_") for f in _FORBIDDEN_NAMES), (
            f"function name looks like a trade-direction signal: {name}"
        )
        sig = inspect.signature(obj)
        return_annotation = str(sig.return_annotation).lower()
        for forbidden in ("side", "buy", "sell"):
            assert forbidden not in return_annotation, (
                f"{name} return annotation mentions '{forbidden}': {return_annotation}"
            )


def test_classification_dataclass_has_no_trade_fields():
    result = classify_statement("Some neutral statement about the weather.", _et(2026, 3, 10, 11, 0))
    assert not hasattr(result, "side")
    assert not hasattr(result, "action")
    assert not hasattr(result, "buy_sell")
