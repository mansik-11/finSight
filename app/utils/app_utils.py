"""Shared utilities for the FinSight Streamlit app: cached data/model loading
and small formatting helpers. Keeping this centralized avoids retraining or
re-reading files on every widget interaction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this app from the app/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, project_path  # noqa: E402
from src.train import get_feature_lists  # noqa: E402


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_data(show_spinner="Loading processed dataset...")
def load_processed_data() -> pd.DataFrame:
    config = load_config()
    path = project_path(config["data"]["processed_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run `make data` (or `python -m src.pipeline`) first."
        )
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading held-out test split...")
def load_test_split() -> pd.DataFrame:
    config = load_config()
    path = project_path(config["paths"]["models_dir"]) / "test_split.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Test split not found at {path}. Run `make train` (or `python -m src.train`) first."
        )
    return pd.read_parquet(path)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model_cached(model_name: str):
    config = load_config()
    path = project_path(config["paths"]["models_dir"]) / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact '{model_name}' not found at {path}. Run `make train` first."
        )
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def get_feature_columns() -> tuple[list[str], list[str]]:
    df = load_processed_data()
    return get_feature_lists(df)


@st.cache_data(show_spinner="Scoring customers...")
def score_all_customers(model_name: str) -> pd.DataFrame:
    """Return the test split with an added `predicted_probability` column
    from the requested model. Cached so re-navigating pages doesn't rescore.
    """
    df = load_test_split().copy()
    numeric, categorical = get_feature_lists(df)
    pipeline = load_model_cached(model_name)
    df["predicted_probability"] = pipeline.predict_proba(df[numeric + categorical])[:, 1]
    return df


def fmt_currency(value: float) -> str:
    return f"${value:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def selected_final_model_name() -> str:
    """Read the evaluation report to find which model was selected as final.
    Falls back to 'xgboost' if evaluation hasn't been run yet."""
    config = load_config()
    metrics_path = project_path(config["paths"]["reports_dir"], "metrics", "model_evaluation.json")
    if not metrics_path.exists():
        return "xgboost"
    import json
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("selected_final_model", "xgboost")
