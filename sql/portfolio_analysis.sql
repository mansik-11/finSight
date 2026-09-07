-- =============================================================================
-- FinSight — Portfolio SQL Analytics (DuckDB)
-- Run against the processed dataset (data/processed/finsight_dataset.parquet)
-- registered as a view called `customers`.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Q1. Overall conversion rate across the analyzed (control + treatment) population
-- -----------------------------------------------------------------------------
-- name: overall_conversion
SELECT
    COUNT(*)                                   AS analysis_population,
    SUM(conversion)                            AS total_conversions,
    ROUND(AVG(conversion), 4)                  AS overall_conversion_rate
FROM customers
WHERE experiment_group IN ('control', 'treatment');


-- -----------------------------------------------------------------------------
-- Q2. Conversion rate by experiment group (control vs. treatment)
-- -----------------------------------------------------------------------------
-- name: conversion_by_group
SELECT
    experiment_group,
    COUNT(*)                                   AS customers,
    SUM(conversion)                            AS conversions,
    ROUND(AVG(conversion), 4)                  AS conversion_rate
FROM customers
WHERE experiment_group IN ('control', 'treatment')
GROUP BY experiment_group
ORDER BY experiment_group;


-- -----------------------------------------------------------------------------
-- Q3. Highest-value customer segments (by risk band x income band),
--     ranked by estimated total financial exposure and conversion rate.
--     Uses a CTE + window function to also show each segment's rank.
-- -----------------------------------------------------------------------------
-- name: highest_value_segments
WITH segment_stats AS (
    SELECT
        risk_band,
        income_band,
        COUNT(*)                                       AS segment_size,
        ROUND(AVG(conversion), 4)                       AS conversion_rate,
        ROUND(AVG(financial_exposure), 2)               AS avg_financial_exposure,
        ROUND(SUM(financial_exposure), 2)                AS total_financial_exposure
    FROM customers
    WHERE experiment_group IN ('control', 'treatment')
    GROUP BY risk_band, income_band
    HAVING COUNT(*) >= 200          -- protect against unreliable tiny segments
)
SELECT
    *,
    RANK() OVER (ORDER BY total_financial_exposure DESC) AS exposure_rank
FROM segment_stats
ORDER BY total_financial_exposure DESC;


-- -----------------------------------------------------------------------------
-- Q4. Which segments have the largest treatment lift (absolute)?
--     Uses CASE WHEN to pivot control/treatment conversion into columns.
-- -----------------------------------------------------------------------------
-- name: segment_lift
WITH per_segment AS (
    SELECT
        engagement_level,
        income_band,
        COUNT(*) AS segment_size,
        AVG(CASE WHEN experiment_group = 'control'   THEN conversion END) AS control_rate,
        AVG(CASE WHEN experiment_group = 'treatment' THEN conversion END) AS treatment_rate,
        SUM(CASE WHEN experiment_group = 'control'   THEN 1 ELSE 0 END)   AS n_control,
        SUM(CASE WHEN experiment_group = 'treatment' THEN 1 ELSE 0 END)   AS n_treatment
    FROM customers
    WHERE experiment_group IN ('control', 'treatment')
    GROUP BY engagement_level, income_band
    HAVING SUM(CASE WHEN experiment_group = 'control' THEN 1 ELSE 0 END) >= 100
       AND SUM(CASE WHEN experiment_group = 'treatment' THEN 1 ELSE 0 END) >= 100
)
SELECT
    engagement_level,
    income_band,
    segment_size,
    ROUND(control_rate, 4)                         AS control_conversion_rate,
    ROUND(treatment_rate, 4)                        AS treatment_conversion_rate,
    ROUND(treatment_rate - control_rate, 4)          AS absolute_lift,
    ROUND((treatment_rate - control_rate) / NULLIF(control_rate, 0), 4) AS relative_lift
FROM per_segment
ORDER BY absolute_lift DESC;


-- -----------------------------------------------------------------------------
-- Q5. Average financial exposure by risk band (join-free aggregation)
-- -----------------------------------------------------------------------------
-- name: exposure_by_risk_band
SELECT
    risk_band,
    COUNT(*)                                    AS customers,
    ROUND(AVG(financial_exposure), 2)           AS avg_financial_exposure,
    ROUND(AVG(credit_utilization), 3)           AS avg_credit_utilization,
    ROUND(AVG(credit_score), 1)                 AS avg_credit_score
FROM customers
GROUP BY risk_band
ORDER BY
    CASE risk_band WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 END;


-- -----------------------------------------------------------------------------
-- Q6. Distribution of customer risk (risk_score deciles) with conversion context
--     Demonstrates NTILE window function.
-- -----------------------------------------------------------------------------
-- name: risk_distribution
WITH deciled AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY risk_score) AS risk_decile
    FROM customers
    WHERE experiment_group IN ('control', 'treatment')
)
SELECT
    risk_decile,
    COUNT(*)                                AS customers,
    ROUND(MIN(risk_score), 3)               AS min_risk_score,
    ROUND(MAX(risk_score), 3)               AS max_risk_score,
    ROUND(AVG(conversion), 4)               AS conversion_rate
FROM deciled
GROUP BY risk_decile
ORDER BY risk_decile;


-- -----------------------------------------------------------------------------
-- Q7. Job categories joined against a small in-SQL lookup of engagement
--     labels, illustrating an explicit JOIN even though the base table is
--     already denormalized (demonstrates JOIN syntax on a realistic use
--     case: attaching a business-friendly display label).
-- -----------------------------------------------------------------------------
-- name: job_engagement_join
WITH engagement_labels(engagement_level, display_label) AS (
    VALUES
        ('Low',    'Cold / No prior response'),
        ('Medium', 'Warm / Prior contact, no success'),
        ('High',   'Hot / Prior success or engaged')
)
SELECT
    c.job,
    e.display_label,
    COUNT(*)                        AS customers,
    ROUND(AVG(c.conversion), 4)     AS conversion_rate
FROM customers c
JOIN engagement_labels e ON c.engagement_level = e.engagement_level
WHERE c.experiment_group IN ('control', 'treatment')
GROUP BY c.job, e.display_label
ORDER BY c.job, conversion_rate DESC;
