"""SQL analytics layer for FinSight.

Executes the real SQL queries in ``sql/portfolio_analysis.sql`` against the
processed dataset using DuckDB. Queries are parsed out of the .sql file by
their `-- name: <query_name>` marker comments so the SQL itself stays the
single source of truth (no SQL duplicated into Python strings).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

from src.config import load_config, project_path

SQL_FILE = project_path("sql", "portfolio_analysis.sql")


def parse_named_queries(sql_path: str | Path = SQL_FILE) -> Dict[str, str]:
    """Parse a .sql file containing `-- name: <query_name>` markers into a dict."""
    text = Path(sql_path).read_text(encoding="utf-8")
    # Split on the marker comment, keeping the name of each subsequent block.
    pattern = re.compile(r"--\s*name:\s*(\w+)\s*\n(.*?)(?=(?:--\s*name:\s*\w+)|\Z)", re.DOTALL)
    queries = {}
    for match in pattern.finditer(text):
        name, body = match.group(1).strip(), match.group(2).strip()
        queries[name] = body
    if not queries:
        raise ValueError(f"No named queries (-- name: ...) found in {sql_path}")
    return queries


def get_connection(processed_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with the processed dataset registered as `customers`."""
    config = load_config()
    processed_path = Path(processed_path) if processed_path else project_path(config["data"]["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. Run `make data` first."
        )
    con = duckdb.connect(database=":memory:")
    con.execute(
        f"CREATE OR REPLACE VIEW customers AS SELECT * FROM read_parquet('{processed_path.as_posix()}')"
    )
    return con


def run_named_query(name: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Run a single named query from ``sql/portfolio_analysis.sql`` and return a DataFrame."""
    own_con = con is None
    con = con or get_connection()
    try:
        queries = parse_named_queries()
        if name not in queries:
            raise KeyError(f"Query '{name}' not found. Available: {list(queries.keys())}")
        return con.execute(queries[name]).fetchdf()
    finally:
        if own_con:
            con.close()


def run_all_queries() -> Dict[str, pd.DataFrame]:
    """Run every named query in the SQL file and return a dict of DataFrames."""
    con = get_connection()
    try:
        queries = parse_named_queries()
        return {name: con.execute(sql).fetchdf() for name, sql in queries.items()}
    finally:
        con.close()


if __name__ == "__main__":
    results = run_all_queries()
    for name, frame in results.items():
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False))
