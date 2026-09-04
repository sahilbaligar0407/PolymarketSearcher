"""Tests for marketlab.analytics.bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from marketlab.analytics.bootstrap import (
    bootstrap_mean,
    ci_excludes_zero,
    cluster_bootstrap,
    effective_sample_size,
    minimum_detectable_effect,
)


def test_seeded_bootstrap_is_deterministic():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5]
    r1 = bootstrap_mean(samples, n_resamples=500, seed=42)
    r2 = bootstrap_mean(samples, n_resamples=500, seed=42)
    assert r1 == r2


def test_seeded_bootstrap_differs_across_seeds_but_stays_deterministic_within_one():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5]
    r_a = bootstrap_mean(samples, n_resamples=500, seed=1)
    r_b = bootstrap_mean(samples, n_resamples=500, seed=2)
    # Different seeds may legitimately produce slightly different SE estimates.
    assert r_a.point_estimate == r_b.point_estimate  # point estimate never depends on resampling
    # But re-running seed 1 must reproduce exactly.
    r_a2 = bootstrap_mean(samples, n_resamples=500, seed=1)
    assert r_a == r_a2


def test_ci_on_known_normal_sample_brackets_true_mean_at_roughly_the_right_rate():
    # A single 95% CI will legitimately miss the true mean ~5% of the time -- test
    # coverage over many independent replicate samples instead of one draw.
    true_mean = 5.0
    n_trials = 60
    hits = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        sample = rng.normal(loc=true_mean, scale=2.0, size=40)
        result = bootstrap_mean(sample.tolist(), n_resamples=500, confidence=0.95, seed=trial)
        if result.ci_low < true_mean < result.ci_high:
            hits += 1
    coverage = hits / n_trials
    # Loose bound: a correctly-behaving 95% interval should cover the truth well over
    # half the time even with n=40 and only 500 resamples; this guards against a
    # grossly broken (e.g. inverted or zero-width) interval, not against exact 95%.
    assert coverage >= 0.80


def test_cluster_bootstrap_one_cluster_is_far_wider_than_many_clusters():
    rng = np.random.default_rng(99)
    values = rng.normal(loc=0.1, scale=0.05, size=500).tolist()

    one_cluster_ids = ["EVENT_A"] * 500
    many_cluster_ids = [f"cluster_{i % 100}" for i in range(500)]

    result_one = cluster_bootstrap(values, one_cluster_ids, n_resamples=1000, seed=5)
    result_many = cluster_bootstrap(values, many_cluster_ids, n_resamples=1000, seed=5)

    width_one = result_one.ci_high - result_one.ci_low
    width_many = result_many.ci_high - result_many.ci_low

    # One cluster: cannot estimate between-cluster variance at all -> unbounded interval.
    assert width_one == float("inf")
    assert width_many < float("inf")
    assert width_one > width_many


def test_effective_sample_size_reflects_clustering():
    one_cluster_ids = ["EVENT_A"] * 500
    many_cluster_ids = [f"cluster_{i % 100}" for i in range(500)]  # 100 clusters of 5

    assert effective_sample_size(one_cluster_ids) == 1.0
    assert effective_sample_size(many_cluster_ids) == 100.0
    assert effective_sample_size([]) == 0.0


def test_three_lucky_trades_do_not_exclude_zero():
    # "3 trades, 3 wins" -- but the wins have some spread. A promotion gate must not be
    # fooled by this: the sample is too small to reject "this was a fluke."
    samples = [0.1, 0.8, 1.5]
    result = bootstrap_mean(samples, n_resamples=2000, confidence=0.95, seed=11)
    assert result.point_estimate == pytest.approx(0.8)
    assert ci_excludes_zero((result.ci_low, result.ci_high)) is False


def test_ci_excludes_zero_straightforward_cases():
    assert ci_excludes_zero((1.0, 2.0)) is True
    assert ci_excludes_zero((-2.0, -1.0)) is True
    assert ci_excludes_zero((-1.0, 1.0)) is False
    assert ci_excludes_zero((float("-inf"), float("inf"))) is False


def test_minimum_detectable_effect_shrinks_with_more_samples():
    mde_small = minimum_detectable_effect(n=10, std=1.0)
    mde_large = minimum_detectable_effect(n=1000, std=1.0)
    assert mde_small is not None and mde_large is not None
    assert mde_large < mde_small
    assert minimum_detectable_effect(n=0, std=1.0) is None
