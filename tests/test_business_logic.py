"""Tests for src.business_simulator (expected value, targeting, risk logic, edge cases)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.business_simulator import SimulatorInputs, run_simulation, score_customers

RISK_COST = {"Low": 5, "Medium": 40, "High": 150}


def _sample_customers(n=100, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "predicted_probability": rng.uniform(0, 1, n),
        "risk_level": rng.choice(["Low", "Medium", "High"], n),
    })


def test_score_customers_expected_value_formula():
    df = pd.DataFrame({"predicted_probability": [0.5], "risk_level": ["Low"]})
    scored = score_customers(
        df, "predicted_probability", "risk_level",
        financial_value_per_conversion=100.0, intervention_cost=10.0, risk_cost_by_level=RISK_COST,
    )
    # EV = 0.5*100 - 10 - 0.5*5 = 50 - 10 - 2.5 = 37.5
    assert scored["expected_value"].iloc[0] == pytest.approx(37.5)


def test_score_customers_higher_risk_lowers_expected_value():
    df = pd.DataFrame({
        "predicted_probability": [0.5, 0.5],
        "risk_level": ["Low", "High"],
    })
    scored = score_customers(
        df, "predicted_probability", "risk_level",
        financial_value_per_conversion=100.0, intervention_cost=10.0, risk_cost_by_level=RISK_COST,
    )
    low_ev = scored.loc[scored["risk_level"] == "Low", "expected_value"].iloc[0]
    high_ev = scored.loc[scored["risk_level"] == "High", "expected_value"].iloc[0]
    assert high_ev < low_ev


def test_run_simulation_respects_probability_threshold():
    df = _sample_customers(200)
    inputs = SimulatorInputs(
        min_conversion_probability=0.8, max_risk_level="High",
        intervention_cost=10, financial_value_per_conversion=100,
        risk_cost_by_level=RISK_COST, target_population_pct=100,
    )
    result = run_simulation(df, inputs, "predicted_probability", "risk_level")
    assert (result.scored_customers["predicted_probability"] >= 0.8).all()


def test_run_simulation_respects_risk_ceiling():
    df = _sample_customers(300)
    inputs = SimulatorInputs(
        min_conversion_probability=0.0, max_risk_level="Low",
        intervention_cost=10, financial_value_per_conversion=100,
        risk_cost_by_level=RISK_COST, target_population_pct=100,
    )
    result = run_simulation(df, inputs, "predicted_probability", "risk_level")
    assert (result.scored_customers["risk_level"] == "Low").all()


def test_run_simulation_target_population_pct_caps_targeting():
    df = _sample_customers(400)
    inputs_full = SimulatorInputs(
        min_conversion_probability=0.0, max_risk_level="High",
        intervention_cost=10, financial_value_per_conversion=100,
        risk_cost_by_level=RISK_COST, target_population_pct=100,
    )
    inputs_half = SimulatorInputs(
        min_conversion_probability=0.0, max_risk_level="High",
        intervention_cost=10, financial_value_per_conversion=100,
        risk_cost_by_level=RISK_COST, target_population_pct=50,
    )
    result_full = run_simulation(df, inputs_full, "predicted_probability", "risk_level")
    result_half = run_simulation(df, inputs_half, "predicted_probability", "risk_level")
    assert result_half.customers_targeted < result_full.customers_targeted
    assert result_half.customers_targeted == pytest.approx(result_full.customers_targeted / 2, abs=1)


def test_run_simulation_targets_highest_expected_value_first():
    df = _sample_customers(500)
    inputs = SimulatorInputs(
        min_conversion_probability=0.0, max_risk_level="High",
        intervention_cost=10, financial_value_per_conversion=100,
        risk_cost_by_level=RISK_COST, target_population_pct=20,
    )
    result = run_simulation(df, inputs, "predicted_probability", "risk_level")
    all_scored = score_customers(
        df, "predicted_probability", "risk_level", 100, 10, RISK_COST
    )
    top_20pct_ev_threshold = all_scored["expected_value"].quantile(0.79)
    # Every targeted customer should be at/above roughly the top-20% EV threshold.
    assert result.scored_customers["expected_value"].min() >= top_20pct_ev_threshold - 1e-6


def test_run_simulation_empty_eligible_population_returns_zero_targeted():
    df = _sample_customers(50)
    inputs = SimulatorInputs(
        min_conversion_probability=1.1,  # impossible threshold -> nobody eligible
        max_risk_level="High", intervention_cost=10, financial_value_per_conversion=100,
        risk_cost_by_level=RISK_COST, target_population_pct=100,
    )
    result = run_simulation(df, inputs, "predicted_probability", "risk_level")
    assert result.customers_targeted == 0
    assert result.expected_conversions == 0
    assert result.expected_net_impact == 0


def test_run_simulation_empty_dataframe_does_not_crash():
    df = pd.DataFrame({"predicted_probability": [], "risk_level": []})
    inputs = SimulatorInputs(
        min_conversion_probability=0.1, max_risk_level="High",
        intervention_cost=10, financial_value_per_conversion=100,
        risk_cost_by_level=RISK_COST, target_population_pct=100,
    )
    result = run_simulation(df, inputs, "predicted_probability", "risk_level")
    assert result.customers_targeted == 0
    assert result.targeting_rate == 0.0
