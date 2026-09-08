"""Page 2 — Experiment & Statistics."""
from __future__ import annotations

import streamlit as st

from app.components.charts import lift_confidence_interval_chart
from app.components.metrics import render_metric_row, render_significance_badges
from app.components.tables import render_balance_table
from app.utils.app_utils import fmt_pct, get_config, load_processed_data

from src.statistics import achieved_power, check_group_balance, required_sample_size_per_group, run_ab_test

PRE_TREATMENT_VARS = ["income", "credit_score", "tenure_years", "credit_utilization", "age"]


def render() -> None:
    config = get_config()
    df = load_processed_data()
    treatment_col = config["experiment"]["treatment_column"]
    outcome_col = config["experiment"]["outcome_column"]
    analysis_df = df[df[treatment_col].isin(["control", "treatment"])]

    st.title("Experiment & Statistics")

    st.markdown("#### Hypothesis")
    st.markdown(
        "- **H0 (null):** The personalized offer has no effect on conversion — "
        "treatment and control conversion rates are equal in the population.\n"
        "- **H1 (alternative):** The personalized offer has an effect on conversion."
    )

    ab_result = run_ab_test(
        analysis_df, treatment_col, outcome_col,
        min_relative_lift_threshold=config["recommendation"]["min_relative_lift"],
    )

    st.markdown("#### Results")
    render_metric_row([
        ("Control (n)", f"{ab_result.n_control:,}", None),
        ("Treatment (n)", f"{ab_result.n_treatment:,}", None),
        ("Control rate", fmt_pct(ab_result.control_rate), None),
        ("Treatment rate", fmt_pct(ab_result.treatment_rate), None),
    ])
    render_metric_row([
        ("Absolute lift", f"{ab_result.absolute_lift:+.2%}", None),
        ("Relative lift", f"{ab_result.relative_lift:+.1%}", None),
        ("z-statistic", f"{ab_result.z_statistic:.2f}", None),
        ("p-value", f"{ab_result.p_value:.2e}", None),
    ])

    st.plotly_chart(
        lift_confidence_interval_chart(ab_result.absolute_lift, ab_result.ci_lower, ab_result.ci_upper),
        width='stretch',
    )

    st.markdown("#### Statistical vs. Practical Significance")
    render_significance_badges(ab_result.is_statistically_significant, ab_result.is_practically_significant)
    st.markdown(
        "> **Statistical significance** tells us the observed difference is very unlikely to be due to random "
        "chance alone. **Practical (business) significance** asks a separate question: is the effect *large "
        f"enough to matter*? Here, the business requires at least a "
        f"**{config['recommendation']['min_relative_lift']*100:.0f}% relative lift** to consider a result "
        "practically meaningful — a statistically significant but tiny lift would not automatically justify a rollout."
    )

    st.markdown("---")
    st.markdown("#### Pre-Treatment Balance Diagnostics")
    st.markdown(
        "Before trusting the treatment effect above, we check that randomization actually produced "
        "comparable groups on variables measured **before** the experiment (so nothing here could have "
        "been influenced by the treatment itself)."
    )
    balance_df = check_group_balance(analysis_df, treatment_col, PRE_TREATMENT_VARS)
    render_balance_table(balance_df)
    n_imbalanced = int((~balance_df["balanced"]).sum()) if not balance_df.empty else 0
    if n_imbalanced == 0:
        st.success("All pre-treatment covariates are balanced (standardized mean difference < 0.10).")
    else:
        st.warning(f"{n_imbalanced} covariate(s) show some imbalance (|SMD| ≥ 0.10). Interpret results with appropriate caution.")

    st.markdown("---")
    st.markdown("#### Experiment Quality")
    missing_outcome = df[treatment_col].eq("ineligible").sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample size (analysis pop.)", f"{len(analysis_df):,}")
    with col2:
        st.metric("Ineligible / excluded", f"{missing_outcome:,}")
    with col3:
        balance_ratio = min(ab_result.n_control, ab_result.n_treatment) / max(ab_result.n_control, ab_result.n_treatment)
        st.metric("Group size balance ratio", f"{balance_ratio:.3f}")

    st.markdown(
        "<div class='fs-section-note'><b>Note on segment-level findings:</b> Segment-level lift results "
        "shown elsewhere in this app (Executive Overview, Customer & ML Insights) are <b>exploratory</b> — "
        "they were not each individually powered as their own experiment, and testing many segments "
        "increases the chance of a false positive by chance alone. Treat segment differences as "
        "hypothesis-generating, not confirmatory, unless followed up with a dedicated test.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Power Analysis")
    st.markdown("**Minimum required sample size per arm**, for a range of minimum detectable effects (MDE), holding power at 80% and alpha at 5%:")
    mde_options = [0.01, 0.02, 0.03, 0.05]
    rows = []
    for mde in mde_options:
        n_required = required_sample_size_per_group(ab_result.control_rate, mde)
        rows.append({"Minimum detectable absolute lift": f"{mde:.0%}", "Required n per arm": f"{n_required:,}"})
    st.dataframe(rows, width='stretch', hide_index=True)

    achieved = achieved_power(min(ab_result.n_control, ab_result.n_treatment), ab_result.control_rate, ab_result.absolute_lift)
    st.metric("Achieved power (at observed effect & sample size)", f"{achieved:.1%}")
    st.caption(
        "Achieved power tells us how likely this experiment design was to detect the effect we actually "
        "observed, given the realized sample sizes. Values close to 100% indicate the experiment was "
        "comfortably well-powered for this effect size."
    )
