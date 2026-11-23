
import re
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype
)


FIELD_PATTERNS = {
    "person_name": [
        "name", "full name", "student name",
        "employee name", "customer name",
        "patient name", "owner name"
    ],
    "age": ["age", "student age", "employee age"],
    "gender": ["gender", "sex"],
    "pronouns": ["pronoun", "pronouns"],
    "email": ["email", "email address"],
    "phone_number": [
        "phone", "mobile", "contact number",
        "phone number", "mobile number"
    ],
    "address": ["address", "home address", "residential address"],
    "student_id": ["student id", "usn", "roll number", "roll no"],
    "employee_id": ["employee id", "staff id", "emp id"],
    "customer_id": ["customer id", "client id"],
    "product_id": ["product id", "item id"],
    "order_id": ["order id", "invoice id"],
    "member_id": ["member id", "membership id"],
    "patient_id": ["patient id", "medical record id"],
    "date_of_birth": ["date of birth", "dob", "birth date"],
    "date": [
        "date", "order date", "joining date",
        "payment date", "transaction date",
        "appointment date"
    ],
    "salary": [
        "salary", "monthly salary", "annual salary",
        "basic pay", "wages"
    ],
    "marks": ["marks", "score", "exam score", "internal marks"],
    "cgpa": ["cgpa", "sgpa", "gpa"],
    "attendance": ["attendance", "attendance percentage"],
    "department": ["department", "dept"],
    "designation": ["designation", "job title", "position"],
    "quantity": ["quantity", "qty", "units"],
    "unit_price": ["unit price", "selling price", "purchase price"],
    "revenue": ["revenue", "sales revenue", "total revenue"],
    "property_price": ["house price", "property price"],
    "area": ["area", "property area", "land area", "square feet"],
    "location": ["location", "city", "state", "country"],
    "product": ["product", "product name", "item name"],
    "currency_amount": ["amount", "total amount", "payment amount"],
}


GENDER_VALUES = {
    "male", "female", "non binary",
    "nonbinary", "other", "prefer not to say"
}

PRONOUN_VALUES = {
    "he", "him", "his", "she", "her",
    "hers", "they", "them", "theirs"
}


def normalize_column_name(name):
    """Standardise column names for matching."""

    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    name = re.sub(r"[^a-zA-Z0-9]+", " ", name)

    return " ".join(name.lower().split())


def match_column_pattern(column_name):
    """Identify a field using known column-name patterns."""

    normalized = normalize_column_name(column_name)

    for field_type, aliases in FIELD_PATTERNS.items():
        for alias in aliases:
            if normalized == normalize_column_name(alias):
                return field_type

    return None


def inspect_sample_values(series):
    """Infer common field types from actual sample values."""

    values = (
        series.dropna()
        .astype(str)
        .str.strip()
        .head(30)
        .tolist()
    )

    if not values:
        return "unknown", 0.0

    # Email detection
    email_count = sum(
        bool(re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            value
        ))
        for value in values
    )

    if email_count / len(values) >= 0.8:
        return "email", 0.95

    # Explicit gender values; never infer gender from names
    normalized = {
        re.sub(r"[^a-z ]", "", value.lower()).strip()
        for value in values
    }

    if normalized and normalized.issubset(GENDER_VALUES):
        return "gender", 0.95

    # Explicit pronouns
    if normalized and normalized.issubset(PRONOUN_VALUES):
        return "pronouns", 0.95

    # Phone-like values
    phone_count = sum(
        bool(re.fullmatch(
            r"\+?[\d\s().-]{8,20}",
            value
        ))
        for value in values
    )

    if phone_count / len(values) >= 0.8:
        return "phone_number", 0.85

    # Date-like text values
    if not is_numeric_dtype(series):
        parsed_dates = pd.to_datetime(
            pd.Series(values),
            errors="coerce",
            format="mixed"
        )

        if parsed_dates.notna().mean() >= 0.8:
            return "date", 0.85

    return "unknown", 0.0


def classify_single_field(column_name, series):
    """Classify an individual DataFrame column."""

    matched_type = match_column_pattern(column_name)

    if matched_type:
        return {
            "column": str(column_name),
            "field_type": matched_type,
            "confidence": 0.95,
            "method": "column_name_matching",
            "needs_review": False
        }

    detected_type, confidence = inspect_sample_values(series)

    if detected_type != "unknown":
        return {
            "column": str(column_name),
            "field_type": detected_type,
            "confidence": confidence,
            "method": "sample_value_analysis",
            "needs_review": False
        }

    if is_bool_dtype(series):
        field_type = "boolean"
    elif is_datetime64_any_dtype(series):
        field_type = "date"
    elif is_numeric_dtype(series):
        field_type = "numeric_measure"
    elif series.notna().any():
        field_type = "text_or_category"
    else:
        field_type = "unknown"

    return {
        "column": str(column_name),
        "field_type": field_type,
        "confidence": 0.5 if field_type != "unknown" else 0.0,
        "method": "general_data_type_analysis",
        "needs_review": True
    }


def classify_fields(dataframe):
    """Classify every column in a Pandas DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if len(dataframe.columns) == 0:
        raise ValueError("Dataset has no columns.")

    fields = []

    for column in dataframe.columns:
        fields.append(
            classify_single_field(
                column,
                dataframe[column]
            )
        )

    return {
        "success": True,
        "total_fields": len(fields),
        "classified_fields": sum(
            field["field_type"] != "unknown"
            for field in fields
        ),
        "fields_needing_review": sum(
            field["needs_review"]
            for field in fields
        ),
        "fields": fields
    }
