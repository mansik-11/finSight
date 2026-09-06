"""Data validation for FinSight.

Performs schema, type, and sanity checks on the raw and processed datasets.
Raises informative errors rather than silently proceeding with bad data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "historical_campaign_outcome",
]

EXPECTED_CATEGORICAL_VALUES = {
    "default": {"yes", "no", "unknown"},
    "housing": {"yes", "no", "unknown"},
    "loan": {"yes", "no", "unknown"},
    "poutcome": {"nonexistent", "failure", "success"},
    "historical_campaign_outcome": {"yes", "no"},
}


@dataclass
class ValidationReport:
    """Container summarizing the results of a validation run."""

    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError(
                "Data validation failed with errors:\n" + "\n".join(self.errors)
            )


def validate_raw_data(df: pd.DataFrame) -> ValidationReport:
    """Validate the freshly-loaded raw dataset.

    Checks required columns exist, key categoricals contain only expected
    values, numeric columns don't contain impossible values, and reports
    duplicate/missing-data statistics.
    """
    report = ValidationReport()

    missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add_error(f"Missing required raw columns: {missing_cols}")
        return report  # can't continue meaningfully without required columns

    if df.empty:
        report.add_error("Raw dataset is empty.")
        return report

    # Duplicates
    n_dupes = int(df.duplicated().sum())
    report.stats["n_duplicates"] = n_dupes
    if n_dupes > 0:
        report.add_warning(f"{n_dupes} duplicate rows found in raw data.")

    # Missing values (dataset uses 'unknown' rather than NaN for categoricals,
    # but we still check for true NaNs which would indicate a load problem)
    n_missing = int(df[REQUIRED_RAW_COLUMNS].isna().sum().sum())
    report.stats["n_missing_values"] = n_missing
    if n_missing > 0:
        report.add_warning(f"{n_missing} NaN values found across required columns.")

    # Impossible values
    if (df["age"] < 17).any() or (df["age"] > 100).any():
        report.add_error("age contains implausible values outside [17, 100].")
    if (df["campaign"] < 1).any():
        report.add_error("campaign (number of contacts) must be >= 1.")
    if (df["previous"] < 0).any():
        report.add_error("previous (prior contacts) cannot be negative.")
    if (df["duration"] < 0).any():
        report.add_error("duration cannot be negative.")

    # Categorical domains
    for col, allowed in EXPECTED_CATEGORICAL_VALUES.items():
        observed = set(df[col].dropna().unique())
        unexpected = observed - allowed
        if unexpected:
            report.add_error(f"Unexpected values in '{col}': {unexpected}")

    # Dtypes
    numeric_cols = ["age", "duration", "campaign", "pdays", "previous"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.add_error(f"Column '{col}' expected to be numeric, got {df[col].dtype}.")

    report.stats["n_rows"] = len(df)
    report.stats["n_columns"] = df.shape[1]
    logger.info("Raw validation complete: passed=%s, errors=%d, warnings=%d",
                report.passed, len(report.errors), len(report.warnings))
    return report


def validate_processed_data(df: pd.DataFrame, target_column: str, treatment_column: str) -> ValidationReport:
    """Validate the fully engineered/processed dataset before it is used for
    analytics or modeling.
    """
    report = ValidationReport()

    if df.empty:
        report.add_error("Processed dataset is empty.")
        return report

    if target_column not in df.columns:
        report.add_error(f"Target column '{target_column}' missing from processed data.")
    else:
        invalid_targets = set(df[target_column].dropna().unique()) - {0, 1}
        if invalid_targets:
            report.add_error(f"Target column contains invalid values: {invalid_targets}")

    if treatment_column not in df.columns:
        report.add_error(f"Treatment column '{treatment_column}' missing from processed data.")
    else:
        invalid_groups = set(df[treatment_column].dropna().unique()) - {"control", "treatment"}
        if invalid_groups:
            report.add_error(f"Treatment column contains invalid values: {invalid_groups}")

    for col in ["income", "credit_score", "tenure_years", "risk_score"]:
        if col in df.columns and df[col].isna().any():
            report.add_error(f"Column '{col}' contains missing values in processed data.")

    if "credit_score" in df.columns:
        if (df["credit_score"] < 300).any() or (df["credit_score"] > 850).any():
            report.add_error("credit_score outside plausible [300, 850] range.")

    if "risk_score" in df.columns:
        if (df["risk_score"] < 0).any() or (df["risk_score"] > 1).any():
            report.add_error("risk_score outside plausible [0, 1] range.")

    report.stats["n_rows"] = len(df)
    logger.info("Processed validation complete: passed=%s, errors=%d",
                report.passed, len(report.errors))
    return report
