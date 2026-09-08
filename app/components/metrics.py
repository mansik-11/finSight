"""Reusable Streamlit metric-row components for the FinSight app."""
from __future__ import annotations

import streamlit as st


def render_metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a row of st.metric() widgets.

    Parameters
    ----------
    metrics:
        List of (label, value, delta) tuples. delta may be None.
    """
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        if delta is not None:
            col.metric(label, value, delta)
        else:
            col.metric(label, value)


def render_recommendation_banner(decision: str, reason: str) -> None:
    """Render a color-coded recommendation banner."""
    if decision == "ROLLOUT":
        st.success(f"**Recommendation: {decision}**\n\n{reason}")
    elif decision == "HOLD / REVIEW":
        st.warning(f"**Recommendation: {decision}**\n\n{reason}")
    else:
        st.error(f"**Recommendation: {decision}**\n\n{reason}")


def render_significance_badges(is_stat_sig: bool, is_practical_sig: bool) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if is_stat_sig:
            st.success("✅ Statistically significant (p < 0.05)")
        else:
            st.warning("⚠️ Not statistically significant")
    with col2:
        if is_practical_sig:
            st.success("✅ Practically / business significant")
        else:
            st.warning("⚠️ Below practical significance threshold")
