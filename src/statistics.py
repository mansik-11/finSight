"""Statistical inference for FinSight's A/B test.

Implements a two-proportion z-test with confidence intervals, effect size,
a practical/business-significance check, and a basic power analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


@dataclass
class ABTestResult:
    """Structured result of a two-proportion A/B test."""

    n_control: int
    n_treatment: int
    conversions_control: int
    conversions_treatment: int
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    is_statistically_significant: bool
    is_practically_significant: bool
    min_relative_lift_threshold: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def run_ab_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str = "control",
    treatment_label: str = "treatment",
    ci_level: float = 0.95,
    min_relative_lift_threshold: float = 0.05,
) -> ABTestResult:
    """Run a two-proportion z-test comparing treatment vs. control conversion rates.

    Parameters
    ----------
    df:
        Analysis population (should already exclude ineligible/missing-outcome rows).
    group_col, outcome_col:
        Column names for experiment arm and binary outcome (0/1).
    ci_level:
        Confidence level for the interval on the absolute lift (default 95%).
    min_relative_lift_threshold:
        Minimum relative lift considered "practically significant" for the business.

    Returns
    -------
    ABTestResult
    """
    control = df.loc[df[group_col] == control_label, outcome_col].dropna()
    treatment = df.loc[df[group_col] == treatment_label, outcome_col].dropna()

    if len(control) == 0 or len(treatment) == 0:
        raise ValueError("Both control and treatment groups must have at least one observation.")

    n_control, n_treatment = len(control), len(treatment)
    x_control, x_treatment = int(control.sum()), int(treatment.sum())

    p_control = x_control / n_control
    p_treatment = x_treatment / n_treatment

    absolute_lift = p_treatment - p_control
    relative_lift = (absolute_lift / p_control) if p_control > 0 else np.nan

    # Two-proportion z-test (pooled variance under H0)
    count = np.array([x_treatment, x_control])
    nobs = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs)

    # 95% CI on the difference in proportions (Wald, unpooled variance)
    se_diff = np.sqrt(p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment)
    z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_lower = absolute_lift - z_crit * se_diff
    ci_upper = absolute_lift + z_crit * se_diff

    is_stat_sig = bool(p_value < (1 - ci_level))
    is_practical_sig = bool((not np.isnan(relative_lift)) and (relative_lift >= min_relative_lift_threshold))

    return ABTestResult(
        n_control=n_control,
        n_treatment=n_treatment,
        conversions_control=x_control,
        conversions_treatment=x_treatment,
        control_rate=p_control,
        treatment_rate=p_treatment,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        z_statistic=float(z_stat),
        p_value=float(p_value),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        is_statistically_significant=is_stat_sig,
        is_practically_significant=is_practical_sig,
        min_relative_lift_threshold=min_relative_lift_threshold,
    )


def wilson_confidence_interval(successes: int, n: int, ci_level: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a single proportion (more stable
    than Wald for small samples or extreme proportions)."""
    if n == 0:
        return (np.nan, np.nan)
    lower, upper = proportion_confint(successes, n, alpha=1 - ci_level, method="wilson")
    return float(lower), float(upper)


def check_group_balance(
    df: pd.DataFrame,
    group_col: str,
    pre_treatment_vars: list[str],
    control_label: str = "control",
    treatment_label: str = "treatment",
) -> pd.DataFrame:
    """Balance check across pre-treatment covariates using Welch's t-test for
    numeric columns. Returns a tidy dataframe with per-variable means and
    a p-value; large imbalances (p < 0.05) should be flagged in the report,
    though with a large N, trivial numeric differences can still be
    "significant" -- standardized mean difference is reported for context.
    """
    rows = []
    control_df = df[df[group_col] == control_label]
    treatment_df = df[df[group_col] == treatment_label]

    for var in pre_treatment_vars:
        c = control_df[var].dropna()
        t = treatment_df[var].dropna()
        if c.empty or t.empty:
            continue
        t_stat, p_val = stats.ttest_ind(t, c, equal_var=False)
        pooled_std = np.sqrt((c.var() + t.var()) / 2)
        smd = (t.mean() - c.mean()) / pooled_std if pooled_std > 0 else 0.0
        rows.append({
            "variable": var,
            "control_mean": c.mean(),
            "treatment_mean": t.mean(),
            "difference": t.mean() - c.mean(),
            "standardized_mean_diff": smd,
            "t_statistic": t_stat,
            "p_value": p_val,
            "balanced": bool(abs(smd) < 0.1),  # common SMD < 0.1 rule of thumb
        })
    return pd.DataFrame(rows)


def required_sample_size_per_group(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Minimum sample size per arm to detect a given absolute lift via a
    two-proportion z-test at the specified alpha/power.
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    effect_size = _cohens_h(p1, p2)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0, alternative="two-sided")
    return int(np.ceil(n))


def achieved_power(
    n_per_group: int,
    baseline_rate: float,
    observed_lift: float,
    alpha: float = 0.05,
) -> float:
    """Statistical power actually achieved given the realized sample sizes and effect."""
    p1 = baseline_rate
    p2 = baseline_rate + observed_lift
    effect_size = _cohens_h(p1, p2)
    analysis = NormalIndPower()
    power = analysis.power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha, ratio=1.0, alternative="two-sided")
    return float(power)


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    p2 = np.clip(p2, 1e-9, 1 - 1e-9)
    return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
