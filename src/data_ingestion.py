"""Data ingestion for FinSight.

Loads the raw public dataset (UCI Bank Marketing) and performs light,
non-lossy normalization: column renaming and dtype coercion only.
No feature engineering happens here -- see ``feature_engineering.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, project_path

logger = logging.getLogger(__name__)

# Rename raw UCI columns (which contain dots) to snake_case, and give
# business-friendly names where it improves readability downstream.
RAW_COLUMN_RENAME = {
    "emp.var.rate": "emp_var_rate",
    "cons.price.idx": "cons_price_idx",
    "cons.conf.idx": "cons_conf_idx",
    "euribor3m": "euribor3m",
    "nr.employed": "nr_employed",
    "y": "historical_campaign_outcome",
}


def load_raw_data(raw_path: str | Path | None = None, delimiter: str | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset from disk.

    Parameters
    ----------
    raw_path:
        Optional override for the raw dataset path. Defaults to the path in
        ``config.yaml``.
    delimiter:
        Optional override for the CSV delimiter.

    Returns
    -------
    pd.DataFrame
        Raw dataset with normalized column names.
    """
    config = load_config()
    raw_path = Path(raw_path) if raw_path else project_path(config["data"]["raw_path"])
    delimiter = delimiter or config["data"]["raw_delimiter"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            "Run `make data` or see README 'Local Setup' to download it."
        )

    logger.info("Loading raw dataset from %s", raw_path)
    df = pd.read_csv(raw_path, sep=delimiter)
    df = df.rename(columns=RAW_COLUMN_RENAME)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info("Loaded raw dataset with shape %s", df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_raw_data()
    print(data.head())
    print(data.shape)
