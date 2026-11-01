
import pandas as pd


def clean_dataset(
    df,
    remove_duplicates=True,
    correct_invalid=True
):
    """
    Clean a dataset without modifying the original.

    Operations:
    1. Normalize empty strings.
    2. Remove duplicate rows.
    3. Identify and correct invalid numerical values.
    4. Fill missing numerical values using median.
    5. Fill missing categorical values using mode.

    Returns:
        cleaned_df: Cleaned copy of the dataset.
        summary: Cleaning operations and statistics.
    """

    if df is None or df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    cleaned = df.copy()

    # --------------------------------
    # 1. NORMALIZE EMPTY VALUES
    # --------------------------------

    for column in cleaned.select_dtypes(
        include=["object", "string"]
    ).columns:

        cleaned[column] = cleaned[column].replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

    original_rows = len(cleaned)

    # --------------------------------
    # 2. REMOVE DUPLICATE RECORDS
    # --------------------------------

    duplicates_removed = 0

    if remove_duplicates:

        duplicates_removed = int(
            cleaned.duplicated().sum()
        )

        cleaned = cleaned.drop_duplicates().copy()

    # --------------------------------
    # 3. DEFINE VALIDATION RANGES
    # --------------------------------

    valid_ranges = {}

    for column in cleaned.columns:

        name = str(column).lower()

        if "age" in name:
            valid_ranges[column] = (1, 100)

        elif any(
            word in name
            for word in ["mark", "score", "attendance"]
        ):
            valid_ranges[column] = (0, 100)

        elif any(
            word in name
            for word in ["price", "quantity", "salary", "amount"]
        ):
            valid_ranges[column] = (0, None)

    # --------------------------------
    # 4. CORRECT INVALID VALUES
    # --------------------------------

    invalid_corrected = 0

    if correct_invalid:

        for column, (minimum, maximum) in valid_ranges.items():

            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce"
            )

            invalid_mask = cleaned[column].notna()

            if minimum is not None:
                invalid_mask &= cleaned[column] < minimum

            if maximum is not None:
                invalid_mask |= (
                    cleaned[column].notna()
                    & (cleaned[column] > maximum)
                )

            invalid_corrected += int(invalid_mask.sum())

            cleaned.loc[invalid_mask, column] = float("nan")

    # --------------------------------
    # 5. COUNT MISSING VALUES
    # --------------------------------

    missing_before_fill = int(
        cleaned.isna().sum().sum()
    )

    # --------------------------------
    # 6. IMPUTE NUMERICAL VALUES
    # --------------------------------

    numeric_columns = cleaned.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:

        median_value = cleaned[column].median()

        if pd.notna(median_value):
            cleaned[column] = cleaned[column].fillna(
                median_value
            )

    # --------------------------------
    # 7. IMPUTE CATEGORICAL VALUES
    # --------------------------------

    categorical_columns = cleaned.select_dtypes(
        exclude=["number"]
    ).columns

    for column in categorical_columns:

        mode_values = cleaned[column].mode(
            dropna=True
        )

        if not mode_values.empty:
            cleaned[column] = cleaned[column].fillna(
                mode_values.iloc[0]
            )

    # --------------------------------
    # 8. GENERATE CLEANING SUMMARY
    # --------------------------------

    remaining_missing = int(
        cleaned.isna().sum().sum()
    )

    summary = {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned),
        "original_columns": len(df.columns),
        "duplicates_removed": duplicates_removed,
        "invalid_values_corrected": invalid_corrected,
        "missing_values_filled": (
            missing_before_fill - remaining_missing
        ),
        "remaining_missing_values": remaining_missing
    }

    return cleaned, summary
