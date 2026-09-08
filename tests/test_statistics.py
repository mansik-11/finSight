"""Tests for src.statistics (A/B testing, balance checks, power analysis)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    achieved_power,
    check_group_balance,
    required_sample_size_per_group,
    run_ab_test,
    wilson_confidence_interval,
)


def _make_ab_df(n_control=1000, n_treatment=1000, p_control=0.10, p_treatment=0.15, seed=0):
    rng = np.random.default_rng(seed)
    control = rng.binomial(1, p_control, n_control)
    treatment = rng.binomial(1, p_treatment, n_treatment)
    df = pd.DataFrame({
        "experiment_group": ["control"] * n_control + ["treatment"] * n_treatment,
        "conversion": np.concatenate([control, treatment]),
    })
    return df


def test_run_ab_test_detects_positive_lift():
    df = _make_ab_df(p_control=0.10, p_treatment=0.20, n_control=5000, n_treatment=5000, seed=1)
    result = run_ab_test(df, "experiment_group", "conversion")
    assert result.treatment_rate > result.control_rate
    assert result.absolute_lift > 0
    assert result.is_statistically_significant


def test_run_ab_test_lift_calculation_is_correct():
    # Deterministic small example: control 2/10, treatment 5/10.
    df = pd.DataFrame({
        "experiment_group": ["control"] * 10 + ["treatment"] * 10,
        "conversion": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    })
    result = run_ab_test(df, "experiment_group", "conversion")
    assert result.control_rate == pytest.approx(0.2)
    assert result.treatment_rate == pytest.approx(0.5)
    assert result.absolute_lift == pytest.approx(0.3)
    assert result.relative_lift == pytest.approx(1.5)


def test_run_ab_test_confidence_interval_contains_point_estimate():
    df = _make_ab_df(seed=2)
    result = run_ab_test(df, "experiment_group", "conversion")
    assert result.ci_lower <= result.absolute_lift <= result.ci_upper


def test_run_ab_test_raises_on_empty_group():
    df = pd.DataFrame({"experiment_group": ["control"] * 5, "conversion": [0, 1, 0, 1, 0]})
    with pytest.raises(ValueError):
        run_ab_test(df, "experiment_group", "conversion")


def test_run_ab_test_no_lift_is_not_significant():
    df = _make_ab_df(p_control=0.10, p_treatment=0.10, n_control=500, n_treatment=500, seed=3)
    result = run_ab_test(df, "experiment_group", "conversion")
    assert result.p_value > 0.01  # extremely unlikely to be "significant" with no true effect


def test_wilson_confidence_interval_bounds():
    lower, upper = wilson_confidence_interval(50, 100)
    assert 0 <= lower <= 0.5 <= upper <= 1


def test_wilson_confidence_interval_zero_n():
    lower, upper = wilson_confidence_interval(0, 0)
    assert np.isnan(lower) and np.isnan(upper)


def test_check_group_balance_flags_imbalanced_variable():
    df = pd.DataFrame({
        "experiment_group": ["control"] * 200 + ["treatment"] * 200,
        "income": list(np.random.default_rng(4).normal(50000, 5000, 200)) +
                  list(np.random.default_rng(5).normal(70000, 5000, 200)),  # clearly shifted
    })
    balance = check_group_balance(df, "experiment_group", ["income"])
    assert len(balance) == 1
    assert balance.iloc[0]["balanced"] == False


def test_check_group_balance_passes_similar_distributions():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({
        "experiment_group": ["control"] * 500 + ["treatment"] * 500,
        "age": list(rng.normal(40, 10, 500)) + list(rng.normal(40, 10, 500)),
    })
    balance = check_group_balance(df, "experiment_group", ["age"])
    assert balance.iloc[0]["balanced"] == True


def test_required_sample_size_increases_as_effect_shrinks():
    n_large_effect = required_sample_size_per_group(0.10, 0.05)
    n_small_effect = required_sample_size_per_group(0.10, 0.01)
    assert n_small_effect > n_large_effect


def test_achieved_power_increases_with_sample_size():
    power_small = achieved_power(n_per_group=100, baseline_rate=0.10, observed_lift=0.03)
    power_large = achieved_power(n_per_group=5000, baseline_rate=0.10, observed_lift=0.03)
    assert power_large > power_small
    assert 0 <= power_small <= 1
    assert 0 <= power_large <= 1
