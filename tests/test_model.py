"""Tests for the modeling pipeline (src.train, src.evaluate, src.explain).

These tests use a small synthetic sample built the same way as the real
pipeline (via feature_engineering) to keep the test suite fast, rather than
retraining on the full 41k-row dataset for every test run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data_ingestion import load_raw_data
from src.feature_engineering import build_processed_dataset
from src.train import (
    build_preprocessor,
    get_feature_lists,
    split_data,
    train_logistic_regression,
    train_xgboost,
)


@pytest.fixture(scope="module")
def small_processed_df():
    config = load_config()
    raw_df = load_raw_data()
    processed = build_processed_dataset(raw_df, config=config)
    processed = processed.dropna(subset=[config["modeling"]["target_column"]])
    # Subsample for fast tests while keeping both classes represented.
    return processed.groupby("conversion", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 1500), random_state=0)
    )


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_split_data_produces_expected_proportions(small_processed_df, config):
    train_df, val_df, test_df = split_data(small_processed_df, config)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(small_processed_df)
    assert len(test_df) / total == pytest.approx(config["modeling"]["test_size"], abs=0.02)


def test_split_data_is_stratified(small_processed_df, config):
    train_df, val_df, test_df = split_data(small_processed_df, config)
    target = config["modeling"]["target_column"]
    overall_rate = small_processed_df[target].mean()
    for split_df in (train_df, val_df, test_df):
        assert abs(split_df[target].mean() - overall_rate) < 0.1


def test_logistic_regression_trains_and_predicts(small_processed_df, config):
    train_df, val_df, _ = split_data(small_processed_df, config)
    numeric, categorical = get_feature_lists(train_df)
    pipeline = train_logistic_regression(train_df, numeric, categorical, config)
    proba = pipeline.predict_proba(val_df[numeric + categorical])
    assert proba.shape == (len(val_df), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_xgboost_trains_and_predicts(small_processed_df, config):
    train_df, val_df, _ = split_data(small_processed_df, config)
    numeric, categorical = get_feature_lists(train_df)
    pipeline = train_xgboost(train_df, numeric, categorical, config)
    proba = pipeline.predict_proba(val_df[numeric + categorical])
    assert proba.shape == (len(val_df), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predictions_have_correct_shape_for_single_row(small_processed_df, config):
    train_df, val_df, _ = split_data(small_processed_df, config)
    numeric, categorical = get_feature_lists(train_df)
    pipeline = train_logistic_regression(train_df, numeric, categorical, config)
    single_row = val_df.iloc[[0]]
    proba = pipeline.predict_proba(single_row[numeric + categorical])
    assert proba.shape == (1, 2)


def test_get_feature_lists_excludes_leakage_columns(small_processed_df):
    numeric, categorical = get_feature_lists(small_processed_df)
    all_features = set(numeric) | set(categorical)
    forbidden = {"conversion", "customer_id", "historical_campaign_outcome", "experiment_group"}
    # experiment_group IS used as a legitimate randomized feature; only check
    # true leakage/id columns are excluded.
    assert "conversion" not in all_features
    assert "customer_id" not in all_features
    assert "historical_campaign_outcome" not in all_features
    assert "sim_true_probability" not in all_features
