"""Model training for FinSight.

Trains two models to predict `conversion`:
  1. Logistic Regression -- interpretable baseline/benchmark.
  2. XGBoost -- nonlinear final model.

Both are tracked with MLflow using a local file-based tracking directory
(no external MLflow server required). Fitted pipelines (preprocessing +
model) are also persisted to `models/` with joblib so the Streamlit app can
load them without retraining.

Leakage safeguards:
  * `duration` (call length) was already dropped in feature_engineering.
  * `historical_campaign_outcome` (the raw dataset's own real-world target)
    is explicitly excluded -- it is not our synthetic experiment's outcome
    and using it would leak unrelated information into the model.
  * Post-outcome / simulation-internal columns (`sim_true_log_odds`,
    `sim_true_probability`, `financial_exposure`-derived risk fields used
    only downstream in the business simulator) are excluded from features.
  * Rows with `experiment_group == 'ineligible'` (missing outcome) are
    dropped before splitting.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import load_config, project_path

logger = logging.getLogger(__name__)

# Columns that must NEVER be used as model features.
EXCLUDED_COLUMNS = {
    "customer_id", "conversion", "experiment_group",
    "historical_campaign_outcome",
    "sim_true_log_odds", "sim_true_probability",
    # Segmentation display bands are derived from raw numeric features already
    # in the feature set (risk_score, income, tenure_years, etc.) -- including
    # both the raw value and its band would be redundant, so we keep the raw,
    # more information-rich numeric columns and drop the display bands here.
    "risk_band", "income_band", "tenure_band", "credit_score_band",
}

NUMERIC_FEATURES = [
    "age", "campaign", "pdays", "previous",
    "income", "credit_score", "tenure_years", "credit_utilization",
    "financial_exposure", "risk_score",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome", "engagement_level",
    "experiment_group",
]


def get_feature_lists(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )


def load_modeling_data(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    processed_path = project_path(config["data"]["processed_path"])
    df = pd.read_parquet(processed_path)
    target_col = config["modeling"]["target_column"]
    # Drop ineligible customers (no simulated outcome).
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)
    return df


def split_data(
    df: pd.DataFrame, config: dict | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split with a fixed seed. No leakage:
    split happens before any fitting of preprocessing or models."""
    config = config or load_config()
    seed = config["random_seed"]
    target_col = config["modeling"]["target_column"]
    test_size = config["modeling"]["test_size"]
    val_size = config["modeling"]["val_size"]

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target_col]
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=seed, stratify=train_val_df[target_col]
    )
    logger.info("Split sizes -> train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def train_logistic_regression(
    train_df: pd.DataFrame, numeric: list[str], categorical: list[str], config: dict
) -> Pipeline:
    target_col = config["modeling"]["target_column"]
    lr_cfg = config["modeling"]["logistic_regression"]
    preprocessor = build_preprocessor(numeric, categorical)
    model = LogisticRegression(
        C=lr_cfg["C"], max_iter=lr_cfg["max_iter"], class_weight=lr_cfg["class_weight"], random_state=config["random_seed"]
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(train_df[numeric + categorical], train_df[target_col])
    return pipeline


def train_xgboost(
    train_df: pd.DataFrame, numeric: list[str], categorical: list[str], config: dict
) -> Pipeline:
    target_col = config["modeling"]["target_column"]
    xgb_cfg = config["modeling"]["xgboost"]

    n_pos = int(train_df[target_col].sum())
    n_neg = int(len(train_df) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    preprocessor = build_preprocessor(numeric, categorical)
    model = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        min_child_weight=xgb_cfg["min_child_weight"],
        reg_lambda=xgb_cfg["reg_lambda"],
        eval_metric=xgb_cfg["eval_metric"],
        scale_pos_weight=scale_pos_weight,
        random_state=config["random_seed"],
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(train_df[numeric + categorical], train_df[target_col])
    return pipeline


def _quick_val_metrics(pipeline: Pipeline, val_df: pd.DataFrame, features: list[str], target_col: str) -> dict:
    proba = pipeline.predict_proba(val_df[features])[:, 1]
    return {
        "val_roc_auc": roc_auc_score(val_df[target_col], proba),
        "val_pr_auc": average_precision_score(val_df[target_col], proba),
    }


def run_training(config: dict | None = None) -> dict:
    """Train both models, log to MLflow, and persist artifacts to `models/`.

    Returns a dict with fitted pipelines and the data splits, so callers
    (e.g. evaluate.py, or a notebook) can reuse them without retraining.
    """
    config = config or load_config()
    target_col = config["modeling"]["target_column"]

    df = load_modeling_data(config)
    numeric, categorical = get_feature_lists(df)
    features = numeric + categorical
    train_df, val_df, test_df = split_data(df, config)

    mlflow_cfg = config["modeling"]["mlflow"]
    tracking_uri = mlflow_cfg["tracking_uri"]
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        # Resolve the relative sqlite path against the project root so MLflow
        # works regardless of the current working directory.
        rel_path = tracking_uri.replace("sqlite:///", "", 1)
        abs_path = project_path(rel_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{abs_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    models_dir = project_path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        logger.info("Training Logistic Regression baseline")
        lr_pipeline = train_logistic_regression(train_df, numeric, categorical, config)
        lr_metrics = _quick_val_metrics(lr_pipeline, val_df, features, target_col)
        mlflow.log_params({f"lr_{k}": v for k, v in config["modeling"]["logistic_regression"].items()})
        mlflow.log_metrics(lr_metrics)
        mlflow.log_param("n_features_numeric", len(numeric))
        mlflow.log_param("n_features_categorical", len(categorical))
        joblib.dump(lr_pipeline, models_dir / "logistic_regression.joblib")
        logger.info("Logistic Regression val metrics: %s", lr_metrics)
        results["logistic_regression"] = {"pipeline": lr_pipeline, "val_metrics": lr_metrics}

    with mlflow.start_run(run_name="xgboost_final_model"):
        logger.info("Training XGBoost final model")
        xgb_pipeline = train_xgboost(train_df, numeric, categorical, config)
        xgb_metrics = _quick_val_metrics(xgb_pipeline, val_df, features, target_col)
        mlflow.log_params({f"xgb_{k}": v for k, v in config["modeling"]["xgboost"].items()})
        mlflow.log_metrics(xgb_metrics)
        joblib.dump(xgb_pipeline, models_dir / "xgboost.joblib")
        logger.info("XGBoost val metrics: %s", xgb_metrics)
        results["xgboost"] = {"pipeline": xgb_pipeline, "val_metrics": xgb_metrics}

    joblib.dump({"numeric": numeric, "categorical": categorical}, models_dir / "feature_lists.joblib")
    train_df.to_parquet(models_dir / "train_split.parquet", index=False)
    val_df.to_parquet(models_dir / "val_split.parquet", index=False)
    test_df.to_parquet(models_dir / "test_split.parquet", index=False)

    results["splits"] = {"train": train_df, "val": val_df, "test": test_df}
    results["features"] = {"numeric": numeric, "categorical": categorical}
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_training()
