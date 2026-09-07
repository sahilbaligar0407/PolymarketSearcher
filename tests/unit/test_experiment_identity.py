"""Tests for marketlab.experiments.identity."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

from marketlab.experiments.identity import (
    ExperimentIdentity,
    canonical_json,
    git_commit,
    parameter_hash,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _base_kwargs() -> dict:
    return dict(
        strategy_name="momentum",
        strategy_version="1.0.0",
        git_commit="abc123",
        parameter_hash=parameter_hash({"lookback_seconds": 300, "threshold": Decimal("0.02")}),
        market_universe="btc_1h",
        venue="kalshi",
        data_version="data.2026.09.04",
        execution_model_version="exec.v1",
        feature_version="feat.v1",
        llm_model_id=None,
        prompt_hash=None,
        start_timestamp=T0,
        starting_bankroll=Decimal("50.00"),
    )


def test_same_inputs_same_id_across_two_constructions() -> None:
    a = ExperimentIdentity(**_base_kwargs())
    b = ExperimentIdentity(**_base_kwargs())
    assert a.experiment_id == b.experiment_id
    assert a.identity_hash == b.identity_hash


def test_experiment_id_is_readable_and_stable_shape() -> None:
    ident = ExperimentIdentity(**_base_kwargs())
    exp_id = ident.experiment_id
    assert exp_id.startswith("MOMENTUM_BTC_1H__")
    prefix, _, tail = exp_id.partition("__")
    assert len(tail) == 12
    assert all(c in "0123456789abcdef" for c in tail)


def test_every_differing_field_changes_the_id() -> None:
    base = ExperimentIdentity(**_base_kwargs())
    overrides = {
        "strategy_name": "mean_reversion",
        "strategy_version": "1.0.1",
        "git_commit": "def456",
        "parameter_hash": parameter_hash({"lookback_seconds": 900, "threshold": Decimal("0.02")}),
        "market_universe": "btc_15m",
        "venue": "poly-us",
        "data_version": "data.2026.09.05",
        "execution_model_version": "exec.v2",
        "feature_version": "feat.v2",
        "llm_model_id": "gpt-oss:20b",
        "prompt_hash": "deadbeef0000",
        "start_timestamp": datetime(2026, 1, 2, tzinfo=UTC),
        "starting_bankroll": Decimal("25.00"),
    }
    for field_name, new_value in overrides.items():
        changed = dataclasses.replace(base, **{field_name: new_value})
        assert changed.experiment_id != base.experiment_id, f"field {field_name!r} did not change the id"
        assert changed.identity_hash != base.identity_hash, f"field {field_name!r} did not change the hash"


def test_parameter_hash_stable_under_key_reordering() -> None:
    a = parameter_hash({"lookback_seconds": 300, "threshold": Decimal("0.02"), "volatility_filter": True})
    b = parameter_hash({"volatility_filter": True, "threshold": Decimal("0.02"), "lookback_seconds": 300})
    assert a == b


def test_parameter_hash_exact_for_decimal_representation() -> None:
    """0.02 and 0.020 are numerically equal but textually different - the hash must see
    the difference, because they came from different config text."""
    a = parameter_hash({"threshold": Decimal("0.02")})
    b = parameter_hash({"threshold": Decimal("0.020")})
    assert a != b


def test_parameter_hash_deterministic_length_and_charset() -> None:
    h = parameter_hash({"a": 1})
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_canonical_json_sorts_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_git_commit_never_raises_and_returns_string() -> None:
    result = git_commit()
    assert isinstance(result, str)
    assert result != ""


def test_git_commit_returns_nogit_for_nonexistent_repo(tmp_path) -> None:
    result = git_commit(repo_root=tmp_path)
    assert result == "nogit"


def test_a_new_commit_does_not_start_a_new_cohort() -> None:
    """An ordinary commit must not orphan every running sleeve.

    `git_commit` belongs in the experiment identity (provenance) but not in the cohort
    key. When it was in both, every commit - including docs-only ones - changed the
    cohort key, so all ~400 live sleeves were abandoned and recreated at a fresh $50.
    Three commits during one debugging session produced 1,164 experiments where there
    should have been 388, and no sleeve ever accumulated enough forward history to be
    promotable.
    """
    base = ExperimentIdentity(**_base_kwargs())
    other_commit = dataclasses.replace(base, git_commit="deadbeefcafe")

    # Same research setup, different commit -> same cohort, so the run continues.
    assert base.cohort_key == other_commit.cohort_key
    # ...but still a distinguishable experiment, because provenance must be preserved.
    assert base.experiment_id != other_commit.experiment_id


def test_a_semantic_version_bump_does_start_a_new_cohort() -> None:
    """The designated switches still work: bumping them invalidates prior results."""
    base = ExperimentIdentity(**_base_kwargs())
    for field_name in ("data_version", "execution_model_version", "feature_version"):
        changed = dataclasses.replace(base, **{field_name: "bumped.v99"})
        assert base.cohort_key != changed.cohort_key, f"{field_name} must change the cohort"

    # Strategy version and parameters are semantic too.
    assert base.cohort_key != dataclasses.replace(base, strategy_version="9.9.9").cohort_key
    assert base.cohort_key != dataclasses.replace(base, parameter_hash="ffffffffffff").cohort_key
