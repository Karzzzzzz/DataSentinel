
from io import BytesIO
from pathlib import Path

import pandas as pd


def extract_tabular_data(
    filename: str,
    file_bytes: bytes
) -> dict:
    """
    Extract data from CSV and Excel files.

    Excel workbooks are processed sheet by sheet.
    CSV files are processed as one dataset.
    The original file bytes are not modified.
    """

    extension = Path(filename).suffix.lower()

    if not file_bytes:
        return {
            "success": False,
            "filename": filename,
            "sheets": [],
            "error": "The uploaded file is empty."
        }

    sheets = []

    try:

        # ======================================
        # EXCEL EXTRACTION
        # ======================================

        if extension in (".xlsx", ".xls"):

            workbook = pd.ExcelFile(
                BytesIO(file_bytes)
            )

            try:
                for sheet_name in workbook.sheet_names:

                    dataframe = workbook.parse(sheet_name)

                    sheets.append({
                        "sheet_name": sheet_name,
                        "dataframe": dataframe,
                        "row_count": len(dataframe),
                        "column_count": len(dataframe.columns),
                        "columns": [
                            str(column)
                            for column in dataframe.columns
                        ],
                        "data_types": {
                            str(column): str(dtype)
                            for column, dtype
                            in dataframe.dtypes.items()
                        }
                    })

            finally:
                workbook.close()

            detected_format = "excel"

        # ======================================
        # CSV EXTRACTION
        # ======================================

        elif extension == ".csv":

            dataframe = None
            last_error = None

            encodings = [
                "utf-8-sig",
                "utf-8",
                "cp1252",
                "latin1"
            ]

            for encoding in encodings:

                try:

                    dataframe = pd.read_csv(
                        BytesIO(file_bytes),
                        encoding=encoding,
                        sep=None,
                        engine="python"
                    )

                    break

                except (
                    UnicodeDecodeError,
                    pd.errors.ParserError
                ) as error:

                    last_error = error

            if dataframe is None:
                raise ValueError(
                    f"Unable to read CSV: {last_error}"
                )

            sheets.append({
                "sheet_name": "CSV Data",
                "dataframe": dataframe,
                "row_count": len(dataframe),
                "column_count": len(dataframe.columns),
                "columns": [
                    str(column)
                    for column in dataframe.columns
                ],
                "data_types": {
                    str(column): str(dtype)
                    for column, dtype
                    in dataframe.dtypes.items()
                }
            })

            detected_format = "csv"

        else:

            return {
                "success": False,
                "filename": filename,
                "sheets": [],
                "error": "Only CSV and Excel files are supported."
            }

        # ======================================
        # RETURN EXTRACTION RESULTS
        # ======================================

        return {
            "success": True,
            "filename": filename,
            "format": detected_format,
            "sheet_count": len(sheets),
            "sheets": sheets,
            "error": None
        }

    except Exception as error:

        return {
            "success": False,
            "filename": filename,
            "sheets": [],
            "error": str(error)
        }
