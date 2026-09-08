"""Reusable Streamlit table/dataframe display helpers for the FinSight app."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_segment_table(segment_df: pd.DataFrame) -> None:
    """Render a segment performance table with friendly formatting and a
    warning for segments below the minimum reliable sample size."""
    if segment_df.empty:
        st.info("No segment data available.")
        return

    display_df = segment_df.copy()
    display_df["control_conversion"] = display_df["control_conversion"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    display_df["treatment_conversion"] = display_df["treatment_conversion"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    display_df["absolute_lift"] = display_df["absolute_lift"].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "—")
    display_df["relative_lift"] = display_df["relative_lift"].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "—")
    display_df["p_value"] = display_df["p_value"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "—")
    display_df["estimated_incremental_value"] = display_df["estimated_incremental_value"].map(
        lambda v: f"${v:,.0f}" if pd.notna(v) else "—"
    )
    display_df["meets_min_sample_size"] = display_df["meets_min_sample_size"].map(lambda b: "✅" if b else "⚠️ small n")

    rename = {
        "dimension": "Dimension", "segment": "Segment", "segment_size": "Size",
        "n_control": "Control n", "n_treatment": "Treatment n",
        "control_conversion": "Control Conv.", "treatment_conversion": "Treatment Conv.",
        "absolute_lift": "Abs. Lift", "relative_lift": "Rel. Lift", "p_value": "p-value",
        "estimated_incremental_value": "Est. Incremental Value", "meets_min_sample_size": "Reliable n",
    }
    display_df = display_df.rename(columns=rename)
    cols_to_show = [c for c in rename.values() if c in display_df.columns]
    st.dataframe(display_df[cols_to_show], width='stretch', hide_index=True)


def render_balance_table(balance_df: pd.DataFrame) -> None:
    if balance_df.empty:
        st.info("No balance diagnostics available.")
        return
    display_df = balance_df.copy()
    display_df["control_mean"] = display_df["control_mean"].map(lambda v: f"{v:,.2f}")
    display_df["treatment_mean"] = display_df["treatment_mean"].map(lambda v: f"{v:,.2f}")
    display_df["difference"] = display_df["difference"].map(lambda v: f"{v:+,.2f}")
    display_df["standardized_mean_diff"] = display_df["standardized_mean_diff"].map(lambda v: f"{v:+.3f}")
    display_df["p_value"] = display_df["p_value"].map(lambda v: f"{v:.3f}")
    display_df["balanced"] = display_df["balanced"].map(lambda b: "✅ balanced" if b else "⚠️ imbalanced")
    rename = {
        "variable": "Variable", "control_mean": "Control Mean", "treatment_mean": "Treatment Mean",
        "difference": "Difference", "standardized_mean_diff": "Std. Mean Diff", "p_value": "p-value",
        "balanced": "Balance Check",
    }
    display_df = display_df.rename(columns=rename)
    st.dataframe(display_df[list(rename.values())], width='stretch', hide_index=True)


def render_customer_profile(customer_row: pd.Series) -> None:
    """Render a customer's key attributes as a compact two-column table."""
    fields = {
        "Age": customer_row.get("age"),
        "Job": customer_row.get("job"),
        "Marital status": customer_row.get("marital"),
        "Education": customer_row.get("education"),
        "Income": f"${customer_row.get('income'):,.0f}" if pd.notna(customer_row.get("income")) else "—",
        "Credit score": f"{customer_row.get('credit_score'):.0f}" if pd.notna(customer_row.get("credit_score")) else "—",
        "Tenure (years)": customer_row.get("tenure_years"),
        "Credit utilization": f"{customer_row.get('credit_utilization'):.0%}" if pd.notna(customer_row.get("credit_utilization")) else "—",
        "Risk band": customer_row.get("risk_band"),
        "Engagement level": customer_row.get("engagement_level"),
        "Experiment group": customer_row.get("experiment_group"),
    }
    df = pd.DataFrame({"Attribute": fields.keys(), "Value": [str(v) for v in fields.values()]})
    st.dataframe(df, width='stretch', hide_index=True)
