
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def generate_observability_profile(df, dataset_name="dataset"):
    """Generate dataset and column-level observability statistics."""

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    column_profiles = []

    for column in df.columns:
        series = df[column]
        valid = series.dropna()

        profile = {
            "column": str(column),
            "data_type": str(series.dtype),
            "total_values": len(series),
            "missing_values": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100), 2
            ),
            "unique_values": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            profile.update({
                "minimum": float(valid.min()) if not valid.empty else None,
                "maximum": float(valid.max()) if not valid.empty else None,
                "mean": float(valid.mean()) if not valid.empty else None,
                "median": float(valid.median()) if not valid.empty else None,
                "standard_deviation": (
                    float(valid.std()) if len(valid) > 1 else None
                )
            })
        else:
            profile["top_values"] = {
                str(key): int(value)
                for key, value in
                valid.astype(str).value_counts().head(5).items()
            }

        column_profiles.append(profile)

    return {
        "dataset_name": dataset_name,
        "captured_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": column_profiles
    }


def save_observability_snapshot(profile, history_dir="reports/observability"):
    """Save a profile snapshot for historical monitoring."""

    folder = Path(history_dir)
    folder.mkdir(parents=True, exist_ok=True)

    dataset_name = profile["dataset_name"]

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        dataset_name
    )

    file_path = folder / f"{safe_name}.jsonl"

    with file_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(profile, allow_nan=False) + "\n"
        )

    return str(file_path)


def load_observability_history(
    dataset_name,
    history_dir="reports/observability"
):
    """Load previous dataset monitoring snapshots."""

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        dataset_name
    )

    file_path = Path(history_dir) / f"{safe_name}.jsonl"

    if not file_path.exists():
        return []

    history = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                history.append(json.loads(line))

    return history


def compare_observability_profiles(previous, current):
    """Compare two dataset profiles."""

    changes = []

    metrics = [
        ("Total Rows", "total_rows"),
        ("Missing Cells", "missing_cells"),
        ("Duplicate Rows", "duplicate_rows")
    ]

    for label, key in metrics:
        old_value = previous[key]
        new_value = current[key]

        changes.append({
            "Metric": label,
            "Previous": old_value,
            "Current": new_value,
            "Change": new_value - old_value
        })

    old_columns = {
        item["column"]: item
        for item in previous["columns"]
    }

    for item in current["columns"]:

        column = item["column"]

        if column not in old_columns:
            continue

        old = old_columns[column]

        changes.append({
            "Metric": f"{column} - Missing Values",
            "Previous": old["missing_values"],
            "Current": item["missing_values"],
            "Change": (
                item["missing_values"] - old["missing_values"]
            )
        })

        if old.get("mean") is not None and item.get("mean") is not None:
            changes.append({
                "Metric": f"{column} - Mean",
                "Previous": round(old["mean"], 3),
                "Current": round(item["mean"], 3),
                "Change": round(
                    item["mean"] - old["mean"], 3
                )
            })

    return pd.DataFrame(changes)
