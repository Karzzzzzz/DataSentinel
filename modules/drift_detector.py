
import os
import tempfile
import pandas as pd


def generate_drift_report(reference_df, current_df):

    if reference_df.empty or current_df.empty:
        raise ValueError("Both datasets must contain records.")

    # Keep columns present in both datasets
    common_columns = [
        col for col in reference_df.columns
        if col in current_df.columns
    ]

    if not common_columns:
        raise ValueError("No common columns found between datasets.")

    reference = reference_df[common_columns].copy()
    current = current_df[common_columns].copy()

    # Align column types
    for column in common_columns:
        if (
            pd.api.types.is_numeric_dtype(reference[column])
            and pd.api.types.is_numeric_dtype(current[column])
        ):
            reference[column] = pd.to_numeric(
                reference[column], errors="coerce"
            )
            current[column] = pd.to_numeric(
                current[column], errors="coerce"
            )
        else:
            reference[column] = reference[column].astype("string")
            current[column] = current[column].astype("string")

    # Remove columns that are entirely empty
    usable_columns = [
        col for col in common_columns
        if not (
            reference[col].isna().all()
            and current[col].isna().all()
        )
    ]

    reference = reference[usable_columns]
    current = current[usable_columns]

    if reference.empty or current.empty or not usable_columns:
        raise ValueError("No usable common columns to compare.")

    # Support modern and older Evidently imports
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

    report = Report(
        metrics=[DataDriftPreset()]
    )

    result = report.run(
        reference_data=reference,
        current_data=current
    )

    # Generate HTML report
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".html",
        delete=False
    )

    report_path = temp_file.name
    temp_file.close()

    try:
        save_function = getattr(result, "save_html", None)

        if not callable(save_function):
            save_function = getattr(report, "save_html", None)

        if not callable(save_function):
            raise RuntimeError("Evidently could not export the report.")

        save_function(report_path)

        with open(report_path, "r", encoding="utf-8") as file:
            html = file.read()

        return html

    finally:
        if os.path.exists(report_path):
            os.remove(report_path)
