"""Shared CSV loading and target-column helpers for pipeline stages."""

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_csv_directory(input_dir: Path) -> pd.DataFrame:
    """Concatenate CSV files from an existing directory."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    csv_paths = list(input_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {input_dir}.")
    logger.info("Loading %d CSV file(s) from %s", len(csv_paths), input_dir)
    return pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)


def split_target(
    data: pd.DataFrame,
    target_column: str,
    source: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature columns and the target without mutating the input frame."""
    if target_column not in data.columns:
        location = f" in {source}" if source is not None else ""
        raise ValueError(f"Target column {target_column!r} not found{location}.")
    return data.drop(columns=[target_column]), data[target_column]


def load_labeled_data(input_path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a labeled CSV and return its features and target separately."""
    return split_target(pd.read_csv(input_path), target_column, source=input_path)