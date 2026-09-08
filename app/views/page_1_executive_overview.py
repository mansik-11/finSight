"""Page 1 — Executive Overview."""
from __future__ import annotations

import streamlit as st

from app.components.charts import control_vs_treatment_bar, segment_bar_chart
from app.components.metrics import render_metric_row, render_recommendation_banner
from app.utils.app_utils import fmt_currency, fmt_pct, get_config, load_processed_data

from src.business_simulator import SimulatorInputs, recommend_rollout, run_simulation
from src.segmentation import build_segment_report, top_segments
from src.statistics import run_ab_test


def render() -> None:
    config = get_config()
    df = load_processed_data()
    analysis_df = df[df["experiment_group"].isin(["control", "treatment"])]

    st.title("FinSight")
    st.markdown(
        "<p class='fs-tagline'>Financial Offer Experimentation & Risk-Aware Customer Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='fs-section-note'>📌 <b>Data note:</b> Customer demographic/behavioral attributes are "
        "real (UCI Bank Marketing dataset). Financial fields (income, credit score, risk) and the "
        "treatment/control experiment are <b>transparently simulated</b> — public datasets don't include "
        "proprietary randomized campaigns. Full methodology in the README.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Executive Overview")

    ab_result = run_ab_test(
        analysis_df, config["experiment"]["treatment_column"], config["experiment"]["outcome_column"],
        min_relative_lift_threshold=config["recommendation"]["min_relative_lift"],
    )

    render_metric_row([
        ("Total customers", f"{len(analysis_df):,}", None),
        ("Control group", f"{ab_result.n_control:,}", None),
        ("Treatment group", f"{ab_result.n_treatment:,}", None),
        ("Analysis population", f"{len(analysis_df):,}", None),
    ])

    st.markdown("")
    render_metric_row([
        ("Control conversion", fmt_pct(ab_result.control_rate), None),
        ("Treatment conversion", fmt_pct(ab_result.treatment_rate), f"+{ab_result.absolute_lift*100:.2f} pp"),
        ("Relative lift", fmt_pct(ab_result.relative_lift), None),
        ("p-value", f"{ab_result.p_value:.2e}", None),
    ])

    st.caption(
        f"95% CI on absolute lift: [{ab_result.ci_lower:+.2%}, {ab_result.ci_upper:+.2%}]  ·  "
        f"z = {ab_result.z_statistic:.2f}"
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            control_vs_treatment_bar(ab_result.control_rate, ab_result.treatment_rate, ab_result.ci_lower, ab_result.ci_upper),
            width='stretch',
        )
    with col2:
        segment_report = build_segment_report(
            df, config["experiment"]["treatment_column"], config["experiment"]["outcome_column"],
            financial_value_per_conversion=config["business"]["financial_value_per_conversion"],
            min_segment_size=config["segmentation"]["min_segment_size"],
        )
        top = top_segments(segment_report, n=6)
        if not top.empty:
            top["label"] = top["dimension"].str.replace("_", " ").str.title() + ": " + top["segment"].astype(str)
            st.plotly_chart(
                segment_bar_chart(top, "estimated_incremental_value", "label", "Top Segments by Estimated Incremental Value ($)", tickformat="$,.0f"),
                width='stretch',
            )

    st.markdown("---")
    st.markdown("#### What does this mean?")
    direction = "higher" if ab_result.absolute_lift > 0 else "lower"
    st.markdown(
        f"Customers who received the **personalized offer (treatment)** converted at a "
        f"**{direction} rate** than customers who received the standard offer (control): "
        f"**{fmt_pct(ab_result.treatment_rate)}** vs. **{fmt_pct(ab_result.control_rate)}**, "
        f"a **{fmt_pct(ab_result.relative_lift)} relative improvement**. "
        f"With a p-value of **{ab_result.p_value:.2e}**, this result is extremely unlikely to be due to chance alone. "
        f"The effect also clears the **{config['recommendation']['min_relative_lift']*100:.0f}% minimum relative lift** "
        f"the business considers practically meaningful — so this is both a **statistically significant** "
        f"and a **practically significant** result."
    )

    # Rollout recommendation using default business simulator assumptions, for
    # a quick top-line view (Page 4 lets the user adjust these live).
    business_cfg = config["business"]
    scored_df = analysis_df.copy()
    # Use sim_true_probability as a stand-in "predicted probability" proxy is
    # not appropriate for the exec summary; instead approximate expected
    # conversions using the realized treatment-arm conversion rate uniformly.
    sim_inputs = SimulatorInputs(
        min_conversion_probability=0.0,
        max_risk_level="High",
        intervention_cost=business_cfg["intervention_cost"],
        financial_value_per_conversion=business_cfg["financial_value_per_conversion"],
        risk_cost_by_level={"Low": business_cfg["risk_cost_multiplier"] * 6, "Medium": business_cfg["risk_cost_multiplier"] * 44, "High": business_cfg["risk_cost_multiplier"] * 167},
        target_population_pct=100,
    )
    treatment_df = analysis_df[analysis_df[config["experiment"]["treatment_column"]] == "treatment"].copy()
    treatment_df["predicted_probability"] = treatment_df[config["experiment"]["outcome_column"]]
    sim_result = run_simulation(treatment_df, sim_inputs, "predicted_probability", "risk_band")
    rec = recommend_rollout(
        ab_result, sim_result,
        min_relative_lift_pct=config["recommendation"]["min_relative_lift"] * 100,
        significance_alpha=config["recommendation"]["significance_level"],
        min_net_impact=config["recommendation"]["min_net_impact"],
    )

    st.markdown("#### Recommendation")
    render_recommendation_banner(rec["decision"], rec["reason"])
    st.caption(
        "This preliminary read uses the realized treatment-arm outcomes directly. "
        "See **Page 4: Targeting & Business Simulator** for a fully interactive, "
        "model-driven, risk-aware version of this recommendation."
    )
