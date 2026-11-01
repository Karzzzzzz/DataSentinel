
import pandas as pd


def analyze_quality(df):
    """
    Analyze a dataset for:
    1. Missing values
    2. Duplicate records
    3. Invalid numerical values
    """

    # Work on a copy so the original dataset is unchanged
    data = df.copy()

    # --------------------------------
    # 1. MISSING VALUE DETECTION
    # --------------------------------

    missing_mask = data.isna()

    # Also treat empty or whitespace-only strings as missing
    for column in data.select_dtypes(include=["object", "string"]).columns:
        missing_mask[column] = (
            missing_mask[column]
            | data[column].astype("string").str.strip().eq("").fillna(False)
        )

    missing_by_column = missing_mask.sum()
    total_missing = int(missing_mask.sum().sum())

    # --------------------------------
    # 2. DUPLICATE RECORD DETECTION
    # --------------------------------

    duplicate_mask = data.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    duplicate_rows = [
        int(index) + 1
        for index in data.index[duplicate_mask]
        if isinstance(index, int)
    ]

    # --------------------------------
    # 3. INVALID VALUE DETECTION
    # --------------------------------

    invalid_records = []

    for column in data.columns:

        column_name = str(column).lower().strip()
        numeric_values = pd.to_numeric(
            data[column], errors="coerce"
        )

        # Identify columns with known value ranges
        if "age" in column_name:
            invalid_mask = (
                numeric_values.notna()
                & ((numeric_values < 0) | (numeric_values > 120))
            )
            reason = "Age must be between 0 and 120"

        elif any(word in column_name for word in
                 ["mark", "score", "percentage", "attendance"]):
            invalid_mask = (
                numeric_values.notna()
                & ((numeric_values < 0) | (numeric_values > 100))
            )
            reason = "Value must be between 0 and 100"

        elif any(word in column_name for word in
                 ["quantity", "price", "salary", "amount"]):
            invalid_mask = (
                numeric_values.notna()
                & (numeric_values < 0)
            )
            reason = "Value cannot be negative"

        else:
            continue

        for index in data.index[invalid_mask]:

            invalid_records.append({
                "Row": int(index) + 1,
                "Column": column,
                "Invalid Value": data.loc[index, column],
                "Reason": reason
            })

    invalid_df = pd.DataFrame(
        invalid_records,
        columns=["Row", "Column", "Invalid Value", "Reason"]
    )

    # --------------------------------
    # 4. RETURN ANALYSIS RESULTS
    # --------------------------------

    return {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "missing_count": total_missing,
        "missing_by_column": missing_by_column,
        "duplicate_count": duplicate_count,
        "duplicate_rows": duplicate_rows,
        "invalid_count": len(invalid_records),
        "invalid_records": invalid_df
    }
