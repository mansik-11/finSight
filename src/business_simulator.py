"""Risk-aware targeting and business impact simulator for FinSight.

The business should not simply target every customer with a high predicted
conversion probability -- financial risk and intervention cost matter too.
This module implements a transparent expected-value scoring framework and
an interactive simulator that recomputes outcomes from live assumptions
(never hardcoded results).

Expected value per targeted customer:

    EV = P(conversion) * financial_value - intervention_cost - P(conversion) * risk_cost(risk_level)

Risk cost is only incurred if the customer actually converts (e.g., takes on
a product that can later default), so it is weighted by predicted conversion
probability, consistent with the documented assumption in config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


@dataclass
class SimulatorInputs:
    min_conversion_probability: float
    max_risk_level: str            # "Low", "Medium", or "High" (inclusive ceiling)
    intervention_cost: float
    financial_value_per_conversion: float
    risk_cost_by_level: dict
    target_population_pct: float = 100.0   # cap on % of eligible customers targeted, ranked by EV


@dataclass
class SimulatorResult:
    customers_eligible: int
    customers_targeted: int
    targeting_rate: float
    expected_conversions: int
    baseline_expected_conversions: float
    incremental_expected_conversions: float
    total_intervention_cost: float
    expected_gross_value: float
    expected_risk_cost: float
    expected_net_impact: float
    avg_expected_value_per_targeted: float
    scored_customers: Optional[pd.DataFrame] = field(default=None, repr=False)


def score_customers(
    df: pd.DataFrame,
    predicted_probability_col: str,
    risk_level_col: str,
    financial_value_per_conversion: float,
    intervention_cost: float,
    risk_cost_by_level: dict,
) -> pd.DataFrame:
    """Attach an expected-value score to every customer row."""
    df = df.copy()
    risk_cost = df[risk_level_col].astype(str).map(risk_cost_by_level).fillna(risk_cost_by_level.get("Medium", 0))
    df["_risk_cost"] = risk_cost.astype(float)
    df["expected_value"] = (
        df[predicted_probability_col] * financial_value_per_conversion
        - intervention_cost
        - df[predicted_probability_col] * risk_cost
    )
    return df


def run_simulation(
    df: pd.DataFrame,
    inputs: SimulatorInputs,
    predicted_probability_col: str = "predicted_probability",
    risk_level_col: str = "risk_level",
    baseline_probability_col: Optional[str] = None,
) -> SimulatorResult:
    """Run the targeting simulation given current assumptions.

    Parameters
    ----------
    df:
        Customer-level dataframe with at least the predicted-probability and
        risk-level columns.
    inputs:
        Current simulator assumptions (thresholds, costs, values).
    baseline_probability_col:
        Optional column giving each customer's baseline (control-arm-like)
        conversion probability, used to compute *incremental* expected
        conversions attributable to targeting/treatment rather than the raw
        expected conversions alone. If omitted, incremental conversions
        equal expected conversions (i.e., no baseline correction).
    """
    max_risk_rank = RISK_ORDER.get(inputs.max_risk_level, 2)

    scored = score_customers(
        df,
        predicted_probability_col=predicted_probability_col,
        risk_level_col=risk_level_col,
        financial_value_per_conversion=inputs.financial_value_per_conversion,
        intervention_cost=inputs.intervention_cost,
        risk_cost_by_level=inputs.risk_cost_by_level,
    )

    eligible_mask = (
        (scored[predicted_probability_col] >= inputs.min_conversion_probability)
        & (scored[risk_level_col].astype(str).map(RISK_ORDER).fillna(2) <= max_risk_rank)
    )
    eligible = scored[eligible_mask]

    # Rank eligible customers by expected value and cap at target_population_pct.
    n_cap = int(np.ceil(len(eligible) * inputs.target_population_pct / 100.0))
    targeted = eligible.sort_values("expected_value", ascending=False).head(n_cap)

    n_eligible = len(eligible)
    n_targeted = len(targeted)

    expected_conversions = float(targeted[predicted_probability_col].sum())
    total_cost = inputs.intervention_cost * n_targeted
    expected_gross_value = expected_conversions * inputs.financial_value_per_conversion
    expected_risk_cost = float((targeted[predicted_probability_col] * targeted["_risk_cost"]).sum())
    expected_net_impact = expected_gross_value - total_cost - expected_risk_cost

    if baseline_probability_col and baseline_probability_col in targeted.columns:
        baseline_expected = float(targeted[baseline_probability_col].sum())
    else:
        baseline_expected = 0.0
    incremental_expected = expected_conversions - baseline_expected

    return SimulatorResult(
        customers_eligible=n_eligible,
        customers_targeted=n_targeted,
        targeting_rate=(n_targeted / len(df)) if len(df) else 0.0,
        expected_conversions=round(expected_conversions),
        baseline_expected_conversions=baseline_expected,
        incremental_expected_conversions=incremental_expected,
        total_intervention_cost=total_cost,
        expected_gross_value=expected_gross_value,
        expected_risk_cost=expected_risk_cost,
        expected_net_impact=expected_net_impact,
        avg_expected_value_per_targeted=(expected_net_impact / n_targeted) if n_targeted else 0.0,
        scored_customers=targeted,
    )


def recommend_rollout(
    ab_test_result,
    simulator_result: SimulatorResult,
    min_relative_lift_pct: float,
    significance_alpha: float,
    min_net_impact: float,
) -> dict:
    """Transparent rollout recommendation combining statistical, business, and
    risk evidence. This is decision *support*, not an automated financial
    decision -- final rollout calls should involve human judgment.
    """
    is_significant = ab_test_result.p_value < significance_alpha
    relative_lift_pct = (ab_test_result.relative_lift or 0) * 100
    meets_lift_bar = relative_lift_pct >= min_relative_lift_pct
    positive_net_impact = simulator_result.expected_net_impact >= min_net_impact

    if is_significant and meets_lift_bar and positive_net_impact:
        decision = "ROLLOUT"
        reason = (
            f"The treatment shows a statistically significant lift (p={ab_test_result.p_value:.2e}) "
            f"of {relative_lift_pct:.1f}% relative conversion improvement, and targeting under current "
            f"risk-aware assumptions yields a positive expected net impact of "
            f"${simulator_result.expected_net_impact:,.0f}."
        )
    elif is_significant and (meets_lift_bar or positive_net_impact):
        decision = "HOLD / REVIEW"
        reason = (
            "The treatment effect is statistically significant, but either the relative lift or the "
            "expected net financial impact does not clearly clear the configured business bar. "
            "Recommend reviewing targeting assumptions or running a follow-up test before a full rollout."
        )
    else:
        decision = "DO NOT ROLLOUT"
        reason = (
            "The treatment effect is not statistically significant and/or the expected net financial "
            "impact under current risk-aware targeting assumptions is not positive."
        )

    return {
        "decision": decision,
        "reason": reason,
        "is_statistically_significant": is_significant,
        "relative_lift_pct": relative_lift_pct,
        "meets_lift_bar": meets_lift_bar,
        "expected_net_impact": simulator_result.expected_net_impact,
        "positive_net_impact": positive_net_impact,
    }
