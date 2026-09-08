"""Tests for data ingestion, validation, and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data_ingestion import load_raw_data
from src.data_validation import validate_processed_data, validate_raw_data
from src.feature_engineering import build_processed_dataset


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="module")
def processed_df(raw_df):
    return build_processed_dataset(raw_df)


def test_raw_data_loads_with_expected_columns(raw_df):
    required = ["age", "job", "marital", "education", "default", "housing", "loan", "poutcome"]
    for col in required:
        assert col in raw_df.columns


def test_raw_data_passes_validation(raw_df):
    report = validate_raw_data(raw_df)
    assert report.passed
    assert report.errors == []


def test_raw_validation_flags_missing_columns():
    bad_df = pd.DataFrame({"age": [30, 40]})
    report = validate_raw_data(bad_df)
    assert not report.passed
    assert any("Missing required raw columns" in e for e in report.errors)


def test_raw_validation_flags_implausible_age():
    df = pd.DataFrame({
        "age": [5, 30], "job": ["admin.", "admin."], "marital": ["single", "single"],
        "education": ["high.school", "high.school"], "default": ["no", "no"],
        "housing": ["no", "no"], "loan": ["no", "no"], "contact": ["cellular", "cellular"],
        "month": ["may", "may"], "day_of_week": ["mon", "mon"], "duration": [100, 100],
        "campaign": [1, 1], "pdays": [999, 999], "previous": [0, 0],
        "poutcome": ["nonexistent", "nonexistent"], "historical_campaign_outcome": ["no", "no"],
    })
    report = validate_raw_data(df)
    assert not report.passed
    assert any("age" in e for e in report.errors)


def test_processed_dataset_has_no_leakage_columns(processed_df):
    # `duration` is a well-documented leakage variable (only known after a call
    # completes) and must never appear in the processed/modeling dataset.
    assert "duration" not in processed_df.columns


def test_processed_dataset_target_is_binary_or_nan(processed_df):
    target = processed_df["conversion"]
    valid_values = set(target.dropna().unique())
    assert valid_values.issubset({0, 1})


def test_processed_dataset_treatment_column_values(processed_df):
    valid_groups = {"control", "treatment", "ineligible"}
    assert set(processed_df["experiment_group"].unique()).issubset(valid_groups)


def test_processed_dataset_passes_validation(processed_df):
    config = load_config()
    analysis_df = processed_df.dropna(subset=[config["experiment"]["outcome_column"]])
    report = validate_processed_data(
        analysis_df,
        target_column=config["experiment"]["outcome_column"],
        treatment_column=config["experiment"]["treatment_column"],
    )
    assert report.passed


def test_credit_score_within_plausible_bounds(processed_df):
    assert processed_df["credit_score"].between(300, 850).all()


def test_risk_score_within_unit_interval(processed_df):
    assert processed_df["risk_score"].between(0, 1).all()


def test_no_missing_values_in_synthetic_financial_columns(processed_df):
    for col in ["income", "credit_score", "tenure_years", "credit_utilization", "risk_score"]:
        assert processed_df[col].isna().sum() == 0


def test_experiment_assignment_is_reproducible(raw_df):
    df1 = build_processed_dataset(raw_df)
    df2 = build_processed_dataset(raw_df)
    pd.testing.assert_series_equal(df1["experiment_group"], df2["experiment_group"])
    pd.testing.assert_series_equal(df1["conversion"], df2["conversion"])


def test_treatment_and_control_are_reasonably_balanced(processed_df):
    counts = processed_df.loc[processed_df["experiment_group"] != "ineligible", "experiment_group"].value_counts()
    ratio = counts.min() / counts.max()
    assert ratio > 0.9  # roughly balanced 50/50 split
