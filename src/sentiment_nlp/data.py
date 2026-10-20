"""Dataset loading, validation, and splitting."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

VALID_LABELS = {"positive", "negative"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV with `text` and `label` columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.loc[:, ["text", "label"]].copy()
    frame["text"] = frame["text"].astype("string")
    frame["label"] = frame["label"].astype("string").str.strip().str.lower()
    frame = frame.dropna(subset=["text", "label"])
    frame["text"] = frame["text"].str.strip()
    frame = frame.loc[frame["text"].ne("")].copy()

    invalid = set(frame["label"].dropna().unique()).difference(VALID_LABELS)
    if invalid:
        raise ValueError(f"Unsupported labels: {sorted(invalid)}")
    if frame["label"].nunique() != 2:
        raise ValueError("Dataset must contain both 'positive' and 'negative' labels.")
    if len(frame) < 10:
        raise ValueError("At least 10 valid rows are required for a train/test split.")
    # Remove exact duplicate text/label pairs. Keep contradictory-label duplicates
    # visible for investigation rather than silently resolving them.
    frame = frame.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if frame["label"].nunique() != 2 or len(frame) < 10:
        raise ValueError("Not enough valid rows/classes remain after duplicate removal.")
    return frame


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified train, validation, and test DataFrames.

    Validation and test proportions are fractions of the complete dataset.
    """
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Split sizes must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")
    train_val, test = train_test_split(
        frame, test_size=test_size, random_state=random_state, stratify=frame["label"]
    )
    relative_val_size = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
