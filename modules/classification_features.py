
import re
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype
)


def normalise_text(value):
    """Convert text into a standardised format."""

    value = str(value)

    value = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        value
    )

    value = re.sub(
        r"[^a-zA-Z0-9]+",
        " ",
        value
    )

    return " ".join(value.lower().split())


def identify_column_type(series):
    """Identify the basic data type of a column."""

    if is_bool_dtype(series):
        return "boolean"

    if is_datetime64_any_dtype(series):
        return "datetime"

    if is_numeric_dtype(series):
        return "numeric"

    return "text categorical"


def identify_value_patterns(series):
    """Detect common patterns in sample values."""

    values = (
        series.dropna()
        .astype(str)
        .str.strip()
        .head(30)
        .tolist()
    )

    if not values:
        return []

    patterns = []

    email_count = sum(
        bool(re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            value
        ))
        for value in values
    )

    phone_count = sum(
        bool(re.fullmatch(
            r"\+?[\d\s().-]{8,20}",
            value
        ))
        for value in values
    )

    if email_count / len(values) >= 0.8:
        patterns.append("email values")

    if phone_count / len(values) >= 0.8:
        patterns.append("phone number values")

    if is_numeric_dtype(series):
        patterns.append("numerical measurements")

    elif is_datetime64_any_dtype(series):
        patterns.append("date and time values")

    # Look for date-like strings
    elif values:
        parsed = pd.to_datetime(
            pd.Series(values),
            errors="coerce",
            format="mixed"
        )

        if parsed.notna().mean() >= 0.8:
            patterns.append("date values")

    # Look for currency-like values
    currency_symbols = ("₹", "$", "€", "£")

    if any(
        value.startswith(currency_symbols)
        for value in values
    ):
        patterns.append("currency values")

    return patterns


def build_dataset_profile(
    dataframe: pd.DataFrame,
    sample_values_per_column: int = 5
) -> dict:
    """
    Analyse an uploaded dataframe and create
    machine-learning features for dataset classification.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a Pandas DataFrame."
        )

    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError(
            "Cannot classify an empty dataset."
        )

    column_profiles = []
    feature_parts = []

    for column in dataframe.columns:

        series = dataframe[column]

        column_name = normalise_text(column)

        column_type = identify_column_type(series)

        missing_percentage = round(
            series.isna().mean() * 100,
            2
        )

        unique_count = int(
            series.nunique(dropna=True)
        )

        sample_values = (
            series.dropna()
            .astype(str)
            .drop_duplicates()
            .head(sample_values_per_column)
            .tolist()
        )

        # Keep sample values reasonably short
        sample_values = [
            value[:80]
            for value in sample_values
        ]

        normalised_samples = [
            normalise_text(value)
            for value in sample_values
        ]

        patterns = identify_value_patterns(series)

        column_profile = {
            "original_name": str(column),
            "normalised_name": column_name,
            "data_type": column_type,
            "missing_percentage": missing_percentage,
            "unique_count": unique_count,
            "sample_values": sample_values,
            "patterns": patterns
        }

        column_profiles.append(column_profile)

        # Build ML feature representation
        column_features = (
            f"column {column_name} "
            f"type {column_type} "
            f"patterns {' '.join(patterns)} "
            f"samples {' '.join(normalised_samples)} "
        )

        # Add useful numerical statistics
        if is_numeric_dtype(series) and not is_bool_dtype(series):

            numeric_values = pd.to_numeric(
                series,
                errors="coerce"
            ).dropna()

            if not numeric_values.empty:

                column_features += (
                    f"minimum {numeric_values.min()} "
                    f"maximum {numeric_values.max()} "
                    f"average {round(numeric_values.mean(), 2)} "
                )

        feature_parts.append(column_features)

    combined_text = (
        f"dataset with {len(dataframe.columns)} columns "
        f"and {len(dataframe)} records "
        + " ".join(feature_parts)
    )

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": column_profiles,
        "classifier_text": combined_text
    }
