"""Feature engineering for FinSight.

This module is the heart of FinSight's documented "synthetic experimentation
layer" described in the README. It performs three distinct jobs, kept
separate for auditability:

1. **Derived financial attributes** -- the public UCI Bank Marketing dataset
   does not contain income, credit score, tenure, utilization or exposure
   figures (real financial institutions do not publish these). We derive
   plausible, internally-consistent versions of these attributes from the
   real demographic/behavioral columns using simple, documented formulas
   with reproducible noise. These are clearly synthetic and are labeled as
   such in the data dictionary.

2. **Experiment assignment** -- eligible customers are randomly split into
   control/treatment arms using a reproducible seed (`numpy.random.Generator`).

3. **Treatment response simulation** -- conversion outcomes are generated
   from a transparent logistic mechanism: a documented baseline propensity
   formula (control arm) plus a documented treatment uplift with a small
   number of named heterogeneous-effect segments (treatment arm). This
   avoids ever presenting real historical outcomes as if they came from a
   randomized experiment (the raw dataset does not contain one).

No step here uses the real dataset's historical outcome (`historical_campaign_outcome`)
to generate the synthetic `conversion` target -- that column is retained only
for provenance/EDA context and is explicitly excluded from modeling features.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

JOB_INCOME_BASE = {
    "management": 78000, "entrepreneur": 72000, "self-employed": 60000,
    "technician": 52000, "admin.": 46000, "services": 40000,
    "housemaid": 30000, "blue-collar": 38000, "retired": 32000,
    "unemployed": 18000, "student": 12000, "unknown": 40000,
}
EDUCATION_INCOME_MULT = {
    "illiterate": 0.75, "basic.4y": 0.82, "basic.6y": 0.85, "basic.9y": 0.90,
    "high.school": 1.00, "professional.course": 1.10, "university.degree": 1.28,
    "unknown": 0.95,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def add_synthetic_financial_attributes(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Derive income, credit_score, tenure_years, credit_utilization,
    financial_exposure, and risk_score from real demographic/behavioral
    columns using documented, reproducible formulas.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # --- Income: job base salary * education multiplier * mild age curve * noise
    job_base = df["job"].map(JOB_INCOME_BASE).fillna(40000)
    edu_mult = df["education"].map(EDUCATION_INCOME_MULT).fillna(0.95)
    age_curve = 1.0 + np.clip((df["age"] - 22) / 100, -0.05, 0.22)  # career growth, plateaus
    income_noise = rng.normal(loc=1.0, scale=0.14, size=n)
    df["income"] = (job_base * edu_mult * age_curve * income_noise).round(-2).clip(lower=8000)

    # --- Credit score: base 680 adjusted by default/loan history, income, age, noise
    base_score = 680.0
    default_penalty = np.select(
        [df["default"] == "yes", df["default"] == "unknown"], [-70.0, -15.0], default=0.0
    )
    poutcome_adj = np.select(
        [df["poutcome"] == "success", df["poutcome"] == "failure"], [22.0, -18.0], default=0.0
    )
    loan_penalty = np.where(df["housing"] == "yes", -8.0, 0.0) + np.where(df["loan"] == "yes", -14.0, 0.0)
    income_adj = (_zscore(df["income"]) * 18.0)
    age_adj = np.clip((df["age"] - 25) * 1.1, -20, 45)
    score_noise = rng.normal(loc=0.0, scale=35.0, size=n)
    raw_score = base_score + default_penalty + poutcome_adj + loan_penalty + income_adj + age_adj + score_noise
    df["credit_score"] = np.clip(raw_score, 300, 850).round(0)

    # --- Tenure years: proxy for length of customer relationship
    tenure_noise = rng.normal(loc=0.0, scale=2.5, size=n)
    raw_tenure = (df["age"] - 18) * 0.35 + np.where(df["previous"] > 0, 1.5, 0.0) + tenure_noise
    df["tenure_years"] = np.clip(raw_tenure, 0, 40).round(1)

    # --- Credit utilization (0-1): higher when score lower / has active loans
    util_noise = rng.normal(loc=0.0, scale=0.08, size=n)
    raw_util = (
        0.45
        - _zscore(df["credit_score"]) * 0.12
        + np.where(df["housing"] == "yes", 0.08, 0.0)
        + np.where(df["loan"] == "yes", 0.10, 0.0)
        + util_noise
    )
    df["credit_utilization"] = np.clip(raw_util, 0.02, 0.98).round(3)

    # --- Financial exposure ($): approximate outstanding obligation exposure
    exposure_mult = 0.9 + 0.6 * df["credit_utilization"]
    df["financial_exposure"] = (df["income"] * 0.35 * exposure_mult).round(-1)

    # --- Composite risk score (0-1): higher = riskier
    risk_raw = (
        0.40 * (1 - _minmax(df["credit_score"]))
        + 0.25 * df["credit_utilization"]
        + 0.20 * np.where(df["default"] == "yes", 1.0, np.where(df["default"] == "unknown", 0.4, 0.0))
        + 0.15 * np.where(df["poutcome"] == "failure", 1.0, 0.0)
    )
    df["risk_score"] = np.clip(risk_raw, 0, 1).round(3)

    return df


def _minmax(s: pd.Series) -> pd.Series:
    rng_ = s.max() - s.min()
    if rng_ == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / rng_


def assign_experiment_groups(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Randomly assign eligible customers to control/treatment arms.

    Eligibility rule (documented business rule): customers currently in
    credit default (`default == 'yes'`) are not eligible to receive a new
    personalized financial offer and are excluded from the experiment
    population entirely (`experiment_group = 'ineligible'`).
    """
    config = config or load_config()
    seed = config["random_seed"]
    treatment_share = config["experiment"]["treatment_share"]
    treatment_col = config["experiment"]["treatment_column"]

    df = df.copy()
    eligible_mask = df["default"] != "yes"

    rng = np.random.default_rng(seed)
    n_eligible = int(eligible_mask.sum())
    draws = rng.random(n_eligible)
    assignment = np.where(draws < treatment_share, "treatment", "control")

    df[treatment_col] = "ineligible"
    df.loc[eligible_mask, treatment_col] = assignment

    logger.info(
        "Experiment assignment: %d eligible (%d treatment / %d control), %d ineligible",
        n_eligible,
        int((assignment == "treatment").sum()),
        int((assignment == "control").sum()),
        int((~eligible_mask).sum()),
    )
    return df


def simulate_conversion_outcome(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Simulate the synthetic `conversion` outcome via a transparent logistic
    mechanism documented in ``config.yaml`` under `experiment.base_propensity`
    and `experiment.treatment_effect`.
    """
    config = config or load_config()
    seed = config["random_seed"] + 1  # decorrelate from assignment draws
    treatment_col = config["experiment"]["treatment_column"]
    outcome_col = config["experiment"]["outcome_column"]
    bp = config["experiment"]["base_propensity"]
    te = config["experiment"]["treatment_effect"]

    df = df.copy()

    # Standardized continuous predictors
    credit_score_z = _zscore(df["credit_score"])
    income_z = _zscore(df["income"])
    tenure_years_z = _zscore(df["tenure_years"])
    campaign_contacts_z = _zscore(df["campaign"])
    consumer_confidence_z = _zscore(df["cons_conf_idx"]) if "cons_conf_idx" in df.columns else 0.0

    # Binary indicators
    previous_success = (df["poutcome"] == "success").astype(float)
    previous_failure = (df["poutcome"] == "failure").astype(float)
    has_default = (df["default"] == "yes").astype(float)
    has_housing_loan = (df["housing"] == "yes").astype(float)
    has_personal_loan = (df["loan"] == "yes").astype(float)
    age_mid_career = df["age"].between(30, 55).astype(float)
    university_educated = (df["education"] == "university.degree").astype(float)

    coef = bp["coefficients"]
    base_log_odds = (
        bp["intercept"]
        + coef["credit_score_z"] * credit_score_z
        + coef["income_z"] * income_z
        + coef["tenure_years_z"] * tenure_years_z
        + coef["previous_success"] * previous_success
        + coef["previous_failure"] * previous_failure
        + coef["campaign_contacts_z"] * campaign_contacts_z
        + coef["has_default"] * has_default
        + coef["has_housing_loan"] * has_housing_loan
        + coef["has_personal_loan"] * has_personal_loan
        + coef["age_mid_career"] * age_mid_career
        + coef["university_educated"] * university_educated
        + coef["consumer_confidence_z"] * consumer_confidence_z
    )

    is_treatment = (df[treatment_col] == "treatment").astype(float)

    # Heterogeneous treatment bonus needs preliminary bands; compute quick
    # tertile-based bands here (final published bands are computed later in
    # `add_segmentation_bands` -- recomputing cheaply keeps this function
    # self-contained and independent of column ordering).
    risk_tertiles = pd.qcut(df["risk_score"], 3, labels=["Low", "Medium", "High"], duplicates="drop")
    income_tertiles = pd.qcut(df["income"], 3, labels=["Low", "Mid", "High"], duplicates="drop")
    engagement_high = ((df["previous"] > 0) & (df["poutcome"] != "failure")).astype(float)

    het_bonus = (
        te["heterogeneous_bonus"]["medium_risk_band"] * (risk_tertiles == "Medium").astype(float)
        + te["heterogeneous_bonus"]["high_engagement"] * engagement_high
        + te["heterogeneous_bonus"]["mid_income_band"] * (income_tertiles == "Mid").astype(float)
    )

    treatment_log_odds_bonus = is_treatment * (te["base_uplift_log_odds"] + het_bonus)
    final_log_odds = base_log_odds + treatment_log_odds_bonus
    final_prob = _sigmoid(final_log_odds)

    rng = np.random.default_rng(seed)
    draws = rng.random(len(df))
    conversion = (draws < final_prob).astype(int)

    # Ineligible customers are not part of the outcome measurement population
    ineligible_mask = df[treatment_col] == "ineligible"
    conversion = np.where(ineligible_mask, np.nan, conversion)

    df["sim_true_log_odds"] = final_log_odds
    df["sim_true_probability"] = final_prob
    df[outcome_col] = conversion

    return df


def add_segmentation_bands(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add business-friendly segment labels used for segmentation analysis
    and the risk-aware targeting layer.
    """
    df = df.copy()

    df["risk_band"] = pd.qcut(df["risk_score"], 3, labels=["Low", "Medium", "High"], duplicates="drop")
    df["income_band"] = pd.qcut(df["income"], 3, labels=["Low", "Mid", "High"], duplicates="drop")
    df["tenure_band"] = pd.cut(
        df["tenure_years"], bins=[-0.01, 3, 10, 100], labels=["New", "Established", "Loyal"]
    )

    conditions = [
        (df["previous"] == 0),
        (df["previous"] > 0) & (df["poutcome"] == "failure"),
        (df["previous"] > 0) & (df["poutcome"] != "failure"),
    ]
    choices = ["Low", "Medium", "High"]
    df["engagement_level"] = np.select(conditions, choices, default="Low")

    df["credit_score_band"] = pd.cut(
        df["credit_score"],
        bins=[299, 579, 669, 739, 799, 851],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    )
    return df


def build_processed_dataset(raw_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Run the full, reproducible feature-engineering pipeline and return the
    processed, analysis-ready dataset.
    """
    config = config or load_config()
    seed = config["random_seed"]

    df = raw_df.copy()
    df.insert(0, "customer_id", [f"CUST{100000 + i}" for i in range(len(df))])

    # Drop known leakage / not-useful-for-this-project columns.
    # `duration` (call length) is only known *after* a call completes and is
    # a well-documented leakage variable for this dataset -- it must never be
    # used as a predictive feature.
    leakage_cols = ["duration"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    df = add_synthetic_financial_attributes(df, seed=seed)
    df = assign_experiment_groups(df, config=config)
    df = simulate_conversion_outcome(df, config=config)
    df = add_segmentation_bands(df, config=config)

    return df


if __name__ == "__main__":
    import logging as _logging
    from src.data_ingestion import load_raw_data

    _logging.basicConfig(level=_logging.INFO)
    raw = load_raw_data()
    processed = build_processed_dataset(raw)
    print(processed.shape)
    print(processed[["experiment_group", "conversion"]].value_counts(dropna=False))
