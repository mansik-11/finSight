"""Model evaluation for FinSight.

Evaluates the Logistic Regression baseline and XGBoost final model on the
held-out test split. Reports ranking metrics (ROC-AUC, PR-AUC), threshold
metrics (precision/recall/F1/confusion matrix at 0.5), and a calibration
curve -- because the business simulator consumes *predicted probabilities*
directly (expected value = P(conversion) * value - cost), so well-calibrated
probabilities matter more than accuracy alone.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import load_config, project_path
from src.train import get_feature_lists

logger = logging.getLogger(__name__)


def load_model(model_name: str, config: dict | None = None):
    config = config or load_config()
    models_dir = project_path(config["paths"]["models_dir"])
    path = models_dir / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def load_test_split(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    models_dir = project_path(config["paths"]["models_dir"])
    path = models_dir / "test_split.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Test split not found at {path}. Run `python -m src.train` first.")
    return pd.read_parquet(path)


def evaluate_model(pipeline, test_df: pd.DataFrame, features: list[str], target_col: str, threshold: float = 0.5) -> dict[str, Any]:
    """Compute the full evaluation metric suite for one fitted pipeline."""
    y_true = test_df[target_col].to_numpy()
    y_proba = pipeline.predict_proba(test_df[features])[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": len(test_df),
        "positive_rate_actual": float(y_true.mean()),
        "positive_rate_predicted": float(y_pred.mean()),
    }
    return metrics, y_true, y_proba


def build_roc_curve_figure(y_true: np.ndarray, y_proba: np.ndarray, model_name: str) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
    fig.update_layout(title=f"ROC Curve — {model_name}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def build_pr_curve_figure(y_true: np.ndarray, y_proba: np.ndarray, model_name: str) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{model_name} (AP={ap:.3f})"))
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray", annotation_text=f"Baseline rate={baseline:.3f}")
    fig.update_layout(title=f"Precision-Recall Curve — {model_name}", xaxis_title="Recall", yaxis_title="Precision")
    return fig


def build_calibration_figure(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, n_bins: int = 10) -> go.Figure:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=model_name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash", color="gray")))
    fig.update_layout(
        title=f"Calibration Curve — {model_name}",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed conversion rate",
    )
    return fig


def run_evaluation(config: dict | None = None) -> dict:
    config = config or load_config()
    target_col = config["modeling"]["target_column"]
    reports_dir = project_path(config["paths"]["reports_dir"])
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_df = load_test_split(config)
    numeric, categorical = get_feature_lists(test_df)
    features = numeric + categorical

    all_metrics = {}
    for model_name in ["logistic_regression", "xgboost"]:
        logger.info("Evaluating %s", model_name)
        pipeline = load_model(model_name, config)
        metrics, y_true, y_proba = evaluate_model(pipeline, test_df, features, target_col)
        all_metrics[model_name] = metrics

        build_roc_curve_figure(y_true, y_proba, model_name).write_html(
            str(figures_dir / f"{model_name}_roc.html"), include_plotlyjs="cdn"
        )
        build_pr_curve_figure(y_true, y_proba, model_name).write_html(
            str(figures_dir / f"{model_name}_pr.html"), include_plotlyjs="cdn"
        )
        build_calibration_figure(y_true, y_proba, model_name).write_html(
            str(figures_dir / f"{model_name}_calibration.html"), include_plotlyjs="cdn"
        )

    # Final model selection: chosen based on business objective (ranking
    # quality for targeting under class imbalance -- PR-AUC) rather than
    # accuracy alone, per project requirements.
    final_model = max(all_metrics, key=lambda m: all_metrics[m]["pr_auc"])
    all_metrics["selected_final_model"] = final_model
    all_metrics["selection_rationale"] = (
        "Selected by highest PR-AUC on the held-out test set, since the business "
        "cares about ranking customers by conversion likelihood under class "
        "imbalance (~13% base rate), not raw accuracy."
    )

    with open(metrics_dir / "model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Evaluation complete. Selected final model: %s", final_model)
    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    results = run_evaluation()
    print(json.dumps(results, indent=2))
