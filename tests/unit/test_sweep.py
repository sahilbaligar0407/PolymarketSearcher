"""Tests for marketlab.experiments.sweep."""

from __future__ import annotations

import copy

import pytest

from marketlab.experiments.sweep import (
    SweepError,
    SweepLimitExceeded,
    VariantSpec,
    count_variants,
    generate_variants,
    load_strategy_class,
)

# Two real, always-importable dotted paths from the frozen core - used as stand-ins for
# "a strategy class that exists" so this test never depends on marketlab.strategies
# (owned by another team and possibly empty/under construction).
_GOOD_CLASS_A = "marketlab.core.strategy:Strategy"
_GOOD_CLASS_B = "marketlab.core.portfolio:Portfolio"
_BAD_CLASS = "marketlab.nonexistent_module_xyz:NoSuchClass"


def _strategies_config() -> dict:
    return {
        "meta": {"max_total_variants": 100},
        "strategies": {
            "strat_a": {
                "enabled": True,
                "evidence": "A",
                "class": _GOOD_CLASS_A,
                "universes": ["uni_1", "uni_2"],
                "variants": {"x": [1, 2], "y": [10, 20]},
                "variant_limit": 3,
            },
            "strat_b_disabled": {
                "enabled": False,
                "class": _GOOD_CLASS_B,
                "universes": ["uni_1"],
                "variants": {},
            },
            "strat_c_bad_class": {
                "enabled": True,
                "class": _BAD_CLASS,
                "universes": ["uni_1"],
                "variants": {},
            },
            "strat_d_unavailable_universe": {
                "enabled": True,
                "class": _GOOD_CLASS_A,
                "universes": ["uni_unavailable"],
                "variants": {},
            },
            "strat_e_ai": {
                "enabled": True,
                "class": _GOOD_CLASS_B,
                "universes": ["uni_1"],
                "variants": {},
                "requires_ai": True,
            },
        },
    }


def _universes_config() -> dict:
    return {
        "universes": {
            "uni_1": {"available": True, "category": "crypto"},
            "uni_2": {"available": True, "category": "crypto"},
            "uni_unavailable": {"available": False, "unavailable_reason": "test fixture"},
        }
    }


def test_load_strategy_class_resolves_real_path() -> None:
    cls = load_strategy_class(_GOOD_CLASS_A)
    assert cls.__name__ == "Strategy"


def test_load_strategy_class_reports_missing_module() -> None:
    with pytest.raises(SweepError, match="cannot import module"):
        load_strategy_class(_BAD_CLASS)


def test_load_strategy_class_reports_missing_attribute() -> None:
    with pytest.raises(SweepError, match="no attribute"):
        load_strategy_class("marketlab.core.strategy:NoSuchClassHere")


def test_load_strategy_class_reports_malformed_path() -> None:
    with pytest.raises(SweepError):
        load_strategy_class("not_a_valid_path")


def test_generate_variants_expands_to_expected_count_and_set() -> None:
    variants = generate_variants(_strategies_config(), _universes_config(), ai_enabled=True)
    # strat_a: 4 combos truncated to 3, x 2 available universes = 6.
    # strat_b_disabled: 0 (disabled).
    # strat_c_bad_class: 0 (bad import, logged, rest still generate).
    # strat_d_unavailable_universe: 0 (its only universe is unavailable).
    # strat_e_ai: 1 combo (no variants block) x 1 universe, ai_enabled=True -> 1.
    assert len(variants) == 7

    by_strategy: dict[str, list[VariantSpec]] = {}
    for v in variants:
        by_strategy.setdefault(v.strategy_name, []).append(v)

    assert set(by_strategy) == {"strat_a", "strat_e_ai"}
    assert len(by_strategy["strat_a"]) == 6
    assert {v.universe for v in by_strategy["strat_a"]} == {"uni_1", "uni_2"}
    assert len(by_strategy["strat_e_ai"]) == 1
    assert by_strategy["strat_e_ai"][0].requires_ai is True


def test_generate_variants_skips_ai_when_disabled() -> None:
    variants = generate_variants(_strategies_config(), _universes_config(), ai_enabled=False)
    assert all(v.strategy_name != "strat_e_ai" for v in variants)


def test_variant_limit_truncates_deterministically() -> None:
    first = generate_variants(_strategies_config(), _universes_config())
    second = generate_variants(_strategies_config(), _universes_config())
    first_params = sorted(
        (v.universe, tuple(sorted(v.params.items()))) for v in first if v.strategy_name == "strat_a"
    )
    second_params = sorted(
        (v.universe, tuple(sorted(v.params.items()))) for v in second if v.strategy_name == "strat_a"
    )
    assert first_params == second_params
    # Exactly 3 distinct param combos survived the variant_limit=3 truncation (out of 4).
    distinct_param_combos = {tuple(sorted(v.params.items())) for v in first if v.strategy_name == "strat_a"}
    assert len(distinct_param_combos) == 3


def test_unavailable_universes_are_skipped() -> None:
    variants = generate_variants(_strategies_config(), _universes_config())
    assert all(v.universe != "uni_unavailable" for v in variants)
    assert all(v.strategy_name != "strat_d_unavailable_universe" for v in variants)


def test_bad_class_path_is_reported_and_rest_still_generate() -> None:
    # Must not raise, and strat_a's variants must still be produced in full.
    variants = generate_variants(_strategies_config(), _universes_config())
    assert all(v.strategy_name != "strat_c_bad_class" for v in variants)
    assert sum(1 for v in variants if v.strategy_name == "strat_a") == 6


def test_max_total_variants_refuses_with_clear_message() -> None:
    config = copy.deepcopy(_strategies_config())
    config["meta"]["max_total_variants"] = 3
    with pytest.raises(SweepLimitExceeded, match="max_total_variants=3"):
        generate_variants(config, _universes_config())


def test_count_variants_is_import_free_and_ignores_ai_gating() -> None:
    counts = count_variants(_strategies_config(), _universes_config())
    assert counts["strat_a"] == 6
    assert counts["strat_c_bad_class"] == 1  # no import validation in the dry-run counter
    assert counts["strat_d_unavailable_universe"] == 0
    assert counts["strat_e_ai"] == 1  # requires_ai not gated by a pure YAML dry run
    assert "strat_b_disabled" not in counts
