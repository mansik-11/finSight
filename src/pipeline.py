"""End-to-end reproducible data pipeline for FinSight.

Run with: `python -m src.pipeline`  (or `make data`)

Raw data -> validation -> feature engineering -> experiment assignment
-> processed dataset validation -> save to data/processed/
"""
from __future__ import annotations

import logging

from src.config import load_config, project_path
from src.data_ingestion import load_raw_data
from src.data_validation import validate_processed_data, validate_raw_data
from src.feature_engineering import build_processed_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    config = load_config()

    logger.info("Step 1/4: Loading raw data")
    raw_df = load_raw_data()

    logger.info("Step 2/4: Validating raw data")
    raw_report = validate_raw_data(raw_df)
    for w in raw_report.warnings:
        logger.warning(w)
    raw_report.raise_if_failed()

    logger.info("Step 3/4: Feature engineering + experiment simulation")
    processed_df = build_processed_dataset(raw_df, config=config)

    logger.info("Step 4/4: Validating processed data")
    processed_report = validate_processed_data(
        processed_df.dropna(subset=[config["experiment"]["outcome_column"]]),
        target_column=config["experiment"]["outcome_column"],
        treatment_column=config["experiment"]["treatment_column"],
    )
    for w in processed_report.warnings:
        logger.warning(w)
    processed_report.raise_if_failed()

    processed_path = project_path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(processed_path, index=False)

    csv_path = project_path(config["data"]["processed_csv_path"])
    processed_df.to_csv(csv_path, index=False)

    logger.info("Saved processed dataset (%s rows, %s cols) to %s and %s",
                processed_df.shape[0], processed_df.shape[1], processed_path, csv_path)


if __name__ == "__main__":
    run_pipeline()
