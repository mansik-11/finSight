"""Reusable Plotly chart builders for the FinSight Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

COLOR_CONTROL = "#94A3B8"
COLOR_TREATMENT = "#2563EB"
COLOR_POSITIVE = "#16A34A"
COLOR_NEGATIVE = "#DC2626"


def control_vs_treatment_bar(control_rate: float, treatment_rate: float, ci_lower: float, ci_upper: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Treatment"], y=[control_rate, treatment_rate],
        marker_color=[COLOR_CONTROL, COLOR_TREATMENT],
        text=[f"{control_rate:.1%}", f"{treatment_rate:.1%}"], textposition="outside",
    ))
    fig.update_layout(
        title="Conversion Rate: Control vs. Treatment",
        yaxis_title="Conversion Rate", yaxis_tickformat=".0%",
        showlegend=False, height=380,
    )
    return fig


def lift_confidence_interval_chart(absolute_lift: float, ci_lower: float, ci_upper: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[ci_lower, ci_upper], y=["Absolute Lift", "Absolute Lift"],
        mode="lines", line=dict(color=COLOR_TREATMENT, width=6), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[absolute_lift], y=["Absolute Lift"], mode="markers",
        marker=dict(size=14, color=COLOR_TREATMENT), name="Point estimate",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Treatment Effect (95% Confidence Interval)",
        xaxis_title="Absolute lift in conversion rate", xaxis_tickformat=".1%",
        height=220, showlegend=False,
    )
    return fig


def segment_bar_chart(segment_df: pd.DataFrame, value_col: str, label_col: str, title: str, tickformat: str = ",.0f") -> go.Figure:
    df = segment_df.sort_values(value_col, ascending=True)
    fig = go.Figure(go.Bar(
        x=df[value_col], y=df[label_col], orientation="h",
        marker_color=COLOR_TREATMENT,
    ))
    fig.update_layout(title=title, xaxis_title=value_col.replace("_", " ").title(), xaxis_tickformat=tickformat, height=max(300, 40 * len(df)))
    return fig


def segment_lift_grouped_bar(segment_df: pd.DataFrame, label_col: str) -> go.Figure:
    df = segment_df.sort_values("estimated_incremental_value", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[label_col], y=df["control_conversion"], name="Control", marker_color=COLOR_CONTROL))
    fig.add_trace(go.Bar(x=df[label_col], y=df["treatment_conversion"], name="Treatment", marker_color=COLOR_TREATMENT))
    fig.update_layout(barmode="group", title="Conversion Rate by Segment", yaxis_tickformat=".0%", height=420)
    return fig


def shap_global_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.sort_values("mean_abs_shap", ascending=True)
    fig = go.Figure(go.Bar(x=df["mean_abs_shap"], y=df["feature"], orientation="h", marker_color=COLOR_TREATMENT))
    fig.update_layout(title="Global Feature Importance (mean |SHAP value|)", xaxis_title="Mean |SHAP value|", height=max(350, 28 * len(df)))
    return fig


def shap_individual_waterfall(top_positive: list[dict], top_negative: list[dict], base_value: float) -> go.Figure:
    combined = list(reversed(top_negative)) + list(reversed(top_positive))
    labels = [c["feature"] for c in combined]
    values = [c["shap_value"] for c in combined]
    colors = [COLOR_NEGATIVE if v < 0 else COLOR_POSITIVE for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
    ))
    fig.add_vline(x=0, line_color="gray")
    fig.update_layout(
        title="Top Contributors to This Customer's Predicted Probability",
        xaxis_title="SHAP value (impact on log-odds)", height=max(350, 32 * len(labels)),
    )
    return fig


def calibration_chart_from_curve(prob_pred: np.ndarray, prob_true: np.ndarray, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=model_name, line_color=COLOR_TREATMENT))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash", color="gray")))
    fig.update_layout(title=f"Calibration — {model_name}", xaxis_title="Mean predicted probability", yaxis_title="Observed conversion rate", height=400)
    return fig


def roc_pr_from_arrays(y_true: np.ndarray, y_proba: np.ndarray, kind: str, model_name: str) -> go.Figure:
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
    fig = go.Figure()
    if kind == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc:.3f})", line_color=COLOR_TREATMENT))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(title=f"ROC Curve — {model_name}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        baseline = float(np.mean(y_true))
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{model_name} (AP={ap:.3f})", line_color=COLOR_TREATMENT))
        fig.add_hline(y=baseline, line_dash="dash", line_color="gray", annotation_text=f"Baseline={baseline:.3f}")
        fig.update_layout(title=f"Precision-Recall Curve — {model_name}", xaxis_title="Recall", yaxis_title="Precision", height=400)
    return fig


def risk_distribution_chart(df: pd.DataFrame, risk_col: str = "risk_score") -> go.Figure:
    fig = px.histogram(df, x=risk_col, nbins=40, color_discrete_sequence=[COLOR_TREATMENT])
    fig.update_layout(title="Distribution of Customer Risk Score", xaxis_title="Risk score (0 = lowest risk, 1 = highest)", yaxis_title="Customers", height=380)
    return fig


def simulator_impact_funnel(eligible: int, targeted: int, expected_conversions: int) -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=["Eligible customers", "Targeted customers", "Expected conversions"],
        x=[eligible, targeted, expected_conversions],
        marker_color=[COLOR_CONTROL, COLOR_TREATMENT, COLOR_POSITIVE],
    ))
    fig.update_layout(title="Targeting Funnel", height=380)
    return fig
