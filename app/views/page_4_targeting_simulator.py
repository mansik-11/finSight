"""Page 4 — Targeting & Business Simulator."""
from __future__ import annotations

import streamlit as st

from app.components.charts import simulator_impact_funnel
from app.components.metrics import render_metric_row, render_recommendation_banner
from app.utils.app_utils import fmt_currency, fmt_pct, get_config, load_test_split, score_all_customers, selected_final_model_name

from src.business_simulator import SimulatorInputs, recommend_rollout, run_simulation
from src.statistics import run_ab_test


def render() -> None:
    config = get_config()
    business_cfg = config["business"]
    final_model_name = selected_final_model_name()

    st.title("Targeting & Business Simulator")
    st.caption(
        f"Uses live predictions from the **{final_model_name.replace('_', ' ').title()}** model. "
        "Adjust the assumptions below — every output recalculates immediately."
    )

    scored_df = score_all_customers(final_model_name)

    st.markdown("#### Targeting Assumptions")
    c1, c2 = st.columns(2)
    with c1:
        min_prob = st.slider(
            "Minimum predicted conversion probability", 0.0, 1.0,
            float(business_cfg["min_predicted_probability"]), step=0.01,
        )
        target_pct = st.slider("Target population (top % by expected value)", 1, 100, 100, step=1)
    with c2:
        max_risk = st.select_slider("Maximum acceptable risk level", options=["Low", "Medium", "High"], value="Medium")
        intervention_cost = st.number_input("Intervention cost per customer ($)", min_value=0.0, value=float(business_cfg["intervention_cost"]), step=1.0)

    financial_value = st.number_input(
        "Estimated financial value per conversion ($)", min_value=0.0,
        value=float(business_cfg["financial_value_per_conversion"]), step=5.0,
    )

    risk_cost_by_level = {
        "Low": business_cfg["risk_cost_multiplier"] * 6,
        "Medium": business_cfg["risk_cost_multiplier"] * 44,
        "High": business_cfg["risk_cost_multiplier"] * 167,
    }
    with st.expander("Advanced: risk cost assumptions (per risk band, incurred only if the customer converts)"):
        for level in ["Low", "Medium", "High"]:
            risk_cost_by_level[level] = st.number_input(f"{level}-risk expected loss ($)", min_value=0.0, value=float(risk_cost_by_level[level]), step=1.0, key=f"risk_{level}")

    sim_inputs = SimulatorInputs(
        min_conversion_probability=min_prob,
        max_risk_level=max_risk,
        intervention_cost=intervention_cost,
        financial_value_per_conversion=financial_value,
        risk_cost_by_level=risk_cost_by_level,
        target_population_pct=target_pct,
    )
    result = run_simulation(scored_df, sim_inputs, predicted_probability_col="predicted_probability", risk_level_col="risk_band")

    st.markdown("---")
    st.markdown("#### Simulated Outcomes")
    render_metric_row([
        ("Customers targeted", f"{result.customers_targeted:,}", None),
        ("Targeting rate", fmt_pct(result.targeting_rate), None),
        ("Expected conversions", f"{result.expected_conversions:,}", None),
        ("Avg. net value / targeted customer", fmt_currency(result.avg_expected_value_per_targeted), None),
    ])
    render_metric_row([
        ("Total intervention cost", fmt_currency(result.total_intervention_cost), None),
        ("Expected gross value", fmt_currency(result.expected_gross_value), None),
        ("Expected risk cost", fmt_currency(result.expected_risk_cost), None),
        ("Expected net impact", fmt_currency(result.expected_net_impact), None),
    ])

    st.plotly_chart(
        simulator_impact_funnel(result.customers_eligible, result.customers_targeted, result.expected_conversions),
        width='stretch',
    )

    st.markdown("---")
    st.markdown("#### Business Recommendation")
    analysis_df = scored_df[scored_df[config["experiment"]["treatment_column"]].isin(["control", "treatment"])]
    ab_result = run_ab_test(analysis_df, config["experiment"]["treatment_column"], config["experiment"]["outcome_column"])
    rec = recommend_rollout(
        ab_result, result,
        min_relative_lift_pct=config["recommendation"]["min_relative_lift"] * 100,
        significance_alpha=config["recommendation"]["significance_level"],
        min_net_impact=config["recommendation"]["min_net_impact"],
    )
    render_recommendation_banner(rec["decision"], rec["reason"])
    st.caption(
        "This recommendation combines statistical evidence from the A/B test with the risk-aware "
        "expected financial impact under your current targeting assumptions above. It is decision "
        "**support**, not an automated financial decision — final rollout calls should involve human review."
    )
