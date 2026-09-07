"""SHAP explainability for FinSight's XGBoost model.

Provides:
  * Global explanation -- top features influencing predictions on average.
  * Individual explanation -- for one customer, predicted probability plus
    top positive/negative contributing features.

Important: SHAP values describe association with the model's prediction,
not causation. This module and any UI built on it must never claim SHAP
"proves" that a feature causes conversion.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.config import load_config
from src.evaluate import load_model, load_test_split
from src.train import get_feature_lists

logger = logging.getLogger(__name__)


def _transform_features(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Apply the pipeline's fitted preprocessor and return the transformed
    array plus the resulting (one-hot expanded) feature names."""
    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return X_transformed, feature_names


def build_shap_explainer(pipeline, background_df: pd.DataFrame, features: list[str]):
    """Build a SHAP TreeExplainer for the XGBoost model inside the pipeline."""
    model = pipeline.named_steps["model"]
    X_bg, _ = _transform_features(pipeline, background_df[features])
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return explainer


def compute_shap_values(pipeline, df: pd.DataFrame, features: list[str]):
    """Compute SHAP values for the given rows. Returns (shap_values, feature_names, X_transformed)."""
    explainer = build_shap_explainer(pipeline, df, features)
    X_transformed, feature_names = _transform_features(pipeline, df[features])
    shap_values = explainer.shap_values(X_transformed)
    return shap_values, feature_names, X_transformed


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across the sample (global importance)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    report = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    report = report.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return report


def explain_individual(
    pipeline,
    customer_row: pd.DataFrame,
    features: list[str],
    background_df: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """Explain a single customer's predicted conversion probability.

    Returns predicted probability plus the top positive and negative SHAP
    contributors, phrased as associations rather than causal claims.
    """
    predicted_proba = float(pipeline.predict_proba(customer_row[features])[:, 1][0])

    explainer = build_shap_explainer(pipeline, background_df, features)
    X_transformed, feature_names = _transform_features(pipeline, customer_row[features])
    shap_values = explainer.shap_values(X_transformed)[0]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[0])

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
        "feature_value": X_transformed[0],
    }).sort_values("shap_value", ascending=False)

    top_positive = contributions.head(top_n).to_dict(orient="records")
    top_negative = contributions.tail(top_n).sort_values("shap_value").to_dict(orient="records")

    return {
        "predicted_probability": predicted_proba,
        "base_value_log_odds": float(base_value),
        "top_positive_contributors": top_positive,
        "top_negative_contributors": top_negative,
        "disclaimer": (
            "These values describe how much each feature is associated with "
            "shifting the model's predicted probability of conversion for this "
            "customer, relative to the average customer. They are not proof "
            "that any feature causes conversion."
        ),
    }


def run_global_explanation(config: dict | None = None, sample_size: int = 2000) -> pd.DataFrame:
    """Convenience entry point: load the XGBoost model + test data and return
    the global feature importance table."""
    config = config or load_config()
    pipeline = load_model("xgboost", config)
    test_df = load_test_split(config)
    numeric, categorical = get_feature_lists(test_df)
    features = numeric + categorical

    sample_df = test_df.sample(min(sample_size, len(test_df)), random_state=config["random_seed"])
    shap_values, feature_names, _ = compute_shap_values(pipeline, sample_df, features)
    return global_feature_importance(shap_values, feature_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    importance = run_global_explanation()
    print(importance.to_string(index=False))
