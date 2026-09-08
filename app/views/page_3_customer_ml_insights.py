"""Page 3 — Customer & ML Insights."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import roc_pr_from_arrays, segment_lift_grouped_bar, shap_global_importance_chart, shap_individual_waterfall
from app.components.metrics import render_metric_row
from app.components.tables import render_customer_profile, render_segment_table
from app.utils.app_utils import fmt_pct, get_config, get_feature_columns, load_model_cached, load_test_split, selected_final_model_name

from src.explain import explain_individual, global_feature_importance, compute_shap_values
from src.segmentation import build_segment_report


@st.cache_data(show_spinner="Computing SHAP values for a sample of customers...")
def _cached_global_shap(model_name: str, sample_size: int = 1500):
    test_df = load_test_split()
    numeric, categorical = get_feature_columns()
    features = numeric + categorical
    pipeline = load_model_cached(model_name)
    sample_df = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    shap_values, feature_names, _ = compute_shap_values(pipeline, sample_df, features)
    return global_feature_importance(shap_values, feature_names), sample_df


def render() -> None:
    config = get_config()
    final_model_name = selected_final_model_name()
    test_df = load_test_split()
    numeric, categorical = get_feature_columns()
    features = numeric + categorical

    st.title("Customer & ML Insights")
    st.caption(f"Predictions and explanations shown use the **{final_model_name.replace('_', ' ').title()}** model.")

    st.markdown("#### Select a Customer")
    sample_ids = test_df["customer_id"].sample(min(500, len(test_df)), random_state=1).sort_values().tolist()
    selected_id = st.selectbox("Customer ID", sample_ids, index=0)
    customer_row = test_df[test_df["customer_id"] == selected_id]

    pipeline = load_model_cached(final_model_name)
    predicted_proba = float(pipeline.predict_proba(customer_row[features])[:, 1][0])
    risk_band = customer_row["risk_band"].iloc[0]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("##### Customer Profile")
        render_customer_profile(customer_row.iloc[0])
    with col2:
        st.markdown("##### Model Prediction")
        render_metric_row([
            ("Predicted conversion probability", f"{predicted_proba:.1%}", None),
            ("Risk category", str(risk_band), None),
        ])
        confidence_note = (
            "high-confidence" if abs(predicted_proba - 0.5) > 0.3 else
            "moderate-confidence" if abs(predicted_proba - 0.5) > 0.15 else
            "low-confidence (near the decision boundary)"
        )
        st.caption(
            f"This is a **{confidence_note}** prediction. Predicted probabilities are calibrated "
            "estimates, not guarantees — see the calibration curve below for how trustworthy these "
            "probabilities are in aggregate."
        )

        background = test_df.sample(min(500, len(test_df)), random_state=7)
        explanation = explain_individual(pipeline if final_model_name == "xgboost" else load_model_cached("xgboost"),
                                          customer_row, features, background)
        st.markdown("###### Why this prediction? (SHAP)")
        st.plotly_chart(
            shap_individual_waterfall(explanation["top_positive_contributors"], explanation["top_negative_contributors"], explanation["base_value_log_odds"]),
            width='stretch',
        )
        st.caption(explanation["disclaimer"])

    st.markdown("---")
    st.markdown("#### Global Feature Importance")
    st.markdown("Averaged across a sample of customers — which features most influence the XGBoost model's predictions overall.")
    importance_df, _ = _cached_global_shap("xgboost")
    st.plotly_chart(shap_global_importance_chart(importance_df), width='stretch')
    st.caption(
        "SHAP values describe how much each feature is associated with shifting model predictions, "
        "on average. This is not evidence of a causal relationship."
    )

    st.markdown("---")
    st.markdown("#### Model Performance (Held-Out Test Set)")
    import json
    from src.config import project_path
    metrics_path = project_path(config["paths"]["reports_dir"], "metrics", "model_evaluation.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            eval_metrics = json.load(f)
        col1, col2 = st.columns(2)
        for col, model_name in zip([col1, col2], ["logistic_regression", "xgboost"]):
            m = eval_metrics.get(model_name, {})
            with col:
                st.markdown(f"**{model_name.replace('_', ' ').title()}**" + (" ⭐ *(selected)*" if model_name == final_model_name else ""))
                render_metric_row([
                    ("ROC-AUC", f"{m.get('roc_auc', 0):.3f}", None),
                    ("PR-AUC", f"{m.get('pr_auc', 0):.3f}", None),
                    ("F1", f"{m.get('f1', 0):.3f}", None),
                ])
        st.caption(eval_metrics.get("selection_rationale", ""))
    else:
        st.info("Run `make evaluate` to generate model evaluation metrics.")

    st.markdown("---")
    st.markdown("#### Segment Performance")
    segment_report = build_segment_report(
        pd.concat([test_df]), config["experiment"]["treatment_column"], config["experiment"]["outcome_column"],
        financial_value_per_conversion=config["business"]["financial_value_per_conversion"],
        min_segment_size=max(30, config["segmentation"]["min_segment_size"] // 10),
    )
    render_segment_table(segment_report)
