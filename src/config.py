"""Central configuration loader for FinSight.

All modules should obtain configuration through :func:`load_config` rather
than hardcoding values, so that assumptions stay auditable in ``config.yaml``.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@functools.lru_cache(maxsize=None)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and cache the project configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file. Defaults to the project's
        top-level ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Ensure config.yaml exists at the project root."
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not config:
        raise ValueError(f"Configuration file at {config_path} is empty or invalid.")
    return config


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root (avoids hardcoded machine paths)."""
    return PROJECT_ROOT.joinpath(*parts)
