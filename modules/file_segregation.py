"""File classification, domain segregation and conservative tabular repair tools.

This module deliberately does not execute uploaded files. Unknown or unsupported
binary formats are assigned to "Unknown / Needs Review" rather than guessed.
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


UNKNOWN_DOMAIN = "Unknown / Needs Review"
MIXED_DOMAIN = "Mixed Domains"

DOMAIN_OPTIONS = [
    "Education",
    "Employment",
    "Real Estate",
    "Finance",
    "Business",
    "Food & Beverage",
    "Healthcare",
    "Household",
    "Transport",
    "Agriculture",
    "Government",
    "Legal",
    "Insurance",
    "Tourism & Hospitality",
    "Sports",
    "Construction",
    "Energy",
    "Environment",
    "IT & Telecom",
    "Research & Science",
    "Media",
    "Events",
    "Automotive",
    "Unknown / Needs Review",
    "Mixed Domains",
]

# Distinctive phrases score more strongly than generic words.
DOMAIN_TERMS: dict[str, list[tuple[str, float]]] = {
    "Education": [
        ("student", 2.0), ("college", 2.5), ("university", 2.5),
        ("school", 2.0), ("semester", 2.0), ("cgpa", 2.5),
        ("gpa", 1.5), ("marksheet", 2.5), ("mark sheet", 2.5),
        ("attendance", 1.2), ("roll number", 2.0),
        ("registration number", 1.5), ("course", 0.8),
        ("faculty", 1.3), ("exam result", 2.0), ("transcript", 2.0),
        ("admission form", 2.0), ("tuition fee", 1.5),
    ],
    "Employment": [
        ("employee", 2.0), ("payroll", 2.5), ("salary", 1.8),
        ("payslip", 2.5), ("pay slip", 2.5), ("human resources", 2.0),
        ("hr department", 1.5), ("designation", 1.2),
        ("joining date", 1.8), ("leave balance", 1.8),
        ("performance appraisal", 2.0), ("employee id", 2.0),
        ("department", 0.7), ("resume", 1.2), ("curriculum vitae", 1.8),
    ],
    "Real Estate": [
        ("real estate", 3.0), ("property id", 2.5), ("property listing", 2.5),
        ("property type", 1.8), ("property", 1.2), ("listing price", 2.0),
        ("square feet", 2.0), ("sqft", 2.0), ("carpet area", 2.0),
        ("built up area", 2.0), ("bedroom", 1.3), ("bathroom", 1.0),
        ("rent amount", 1.8), ("lease agreement", 2.0),
        ("tenant", 1.4), ("landlord", 1.4), ("property price", 2.0),
        ("house price", 2.0), ("apartment", 1.0), ("flat number", 1.2),
    ],
    "Finance": [
        ("bank statement", 2.5), ("account number", 1.2),
        ("ifsc", 2.0), ("transaction id", 2.0), ("debit", 1.0),
        ("credit", 0.8), ("account balance", 2.0), ("loan amount", 2.0),
        ("interest rate", 1.8), ("tax return", 2.0),
        ("gstin", 2.0), ("invoice number", 1.2), ("payment status", 1.5),
        ("investment", 1.0), ("mutual fund", 2.0),
    ],
    "Business": [
        ("customer id", 1.8), ("customer", 0.8), ("order id", 1.8),
        ("sales", 1.0), ("product sku", 2.0), ("sku", 1.2),
        ("inventory", 2.0), ("stock quantity", 2.0),
        ("purchase order", 2.0), ("supplier", 1.2),
        ("revenue", 1.2), ("unit price", 1.2), ("e-commerce", 1.5),
    ],
    "Food & Beverage": [
        ("food order", 2.0), ("restaurant", 1.8), ("menu item", 2.0),
        ("recipe", 1.5), ("ingredients", 1.2), ("nutrition", 1.5),
        ("calories", 1.5), ("food delivery", 2.0),
        ("beverage", 1.5), ("grocery item", 1.5),
    ],
    "Healthcare": [
        ("patient id", 2.0), ("patient", 1.2), ("diagnosis", 1.8),
        ("prescription", 2.0), ("hospital", 1.8), ("medical record", 2.5),
        ("blood pressure", 1.8), ("lab result", 1.8),
        ("discharge summary", 2.5), ("doctor", 0.8),
        ("dosage", 1.3), ("clinical", 1.3),
    ],
    "Household": [
        ("household", 2.0), ("grocery bill", 2.0), ("electricity bill", 2.0),
        ("water bill", 1.8), ("family member", 1.5),
        ("monthly expenses", 1.8), ("home expense", 2.0),
        ("utility bill", 1.8), ("domestic budget", 1.8),
    ],
    "Transport": [
        ("vehicle number", 1.8), ("vehicle id", 1.8), ("route id", 1.8),
        ("shipment", 1.8), ("tracking number", 1.8), ("fleet", 1.5),
        ("dispatch", 1.3), ("odometer", 1.8), ("fuel consumption", 1.8),
        ("transport", 1.2), ("logistics", 1.6),
    ],
    "Agriculture": [
        ("crop", 1.4), ("harvest", 1.8), ("farm", 1.2),
        ("yield per acre", 2.2), ("irrigation", 1.8),
        ("soil type", 1.4), ("fertilizer", 1.8),
        ("seed variety", 1.8), ("livestock", 1.6),
    ],
    "Government": [
        ("government department", 2.2), ("municipality", 1.8),
        ("citizen id", 2.0), ("public service", 1.5),
        ("permit number", 1.6), ("passport number", 1.6),
        ("voter id", 1.8),
    ],
    "Legal": [
        ("legal notice", 2.2), ("court case", 2.2), ("affidavit", 2.2),
        ("agreement", 1.0), ("contract clause", 1.8),
        ("party of the first part", 2.2), ("law firm", 1.8),
        ("case number", 1.4),
    ],
    "Insurance": [
        ("insurance policy", 2.5), ("policy number", 2.0),
        ("claim number", 2.0), ("premium amount", 1.8),
        ("coverage amount", 1.8), ("insurer", 1.8),
        ("insurance claim", 2.5),
    ],
    "Tourism & Hospitality": [
        ("hotel booking", 2.2), ("check in", 1.2), ("check-out", 1.2),
        ("guest name", 1.4), ("reservation", 1.0),
        ("room number", 1.0), ("travel itinerary", 2.0),
        ("tour package", 2.0),
    ],
    "Sports": [
        ("player id", 1.8), ("team name", 1.6), ("match score", 2.0),
        ("tournament", 1.4), ("coach", 0.8), ("points table", 1.8),
    ],
    "Construction": [
        ("construction site", 2.0), ("building permit", 2.0),
        ("contractor", 1.2), ("project cost", 1.2),
        ("cement", 1.2), ("site engineer", 1.8),
    ],
    "Energy": [
        ("kilowatt hour", 2.0), ("kwh", 1.5), ("power generation", 2.0),
        ("solar panel", 1.8), ("energy consumption", 2.0),
    ],
    "Environment": [
        ("carbon emission", 2.0), ("air quality", 1.8),
        ("water quality", 1.8), ("waste management", 2.0),
        ("recycling", 1.3), ("biodiversity", 1.5),
    ],
    "IT & Telecom": [
        ("ip address", 1.5), ("server name", 1.5),
        ("software license", 2.0), ("network device", 1.8),
        ("telecom", 1.8), ("bandwidth", 1.5),
        ("operating system", 1.2),
    ],
    "Research & Science": [
        ("research paper", 2.2), ("methodology", 1.2),
        ("literature review", 1.8), ("hypothesis", 1.2),
        ("experiment results", 1.8), ("references", 0.7),
        ("abstract", 0.7), ("research dataset", 2.0),
    ],
    "Media": [
        ("publication date", 1.2), ("editorial", 1.3),
        ("news article", 2.0), ("broadcast", 1.4),
        ("media outlet", 1.5),
    ],
    "Events": [
        ("event name", 1.5), ("event venue", 1.8),
        ("event registration", 2.0), ("attendee", 1.2),
        ("conference", 1.2),
    ],
    "Automotive": [
        ("vehicle model", 1.8), ("vehicle identification number", 2.2),
        ("vin", 1.0), ("engine number", 1.8),
        ("chassis number", 1.8), ("service history", 1.2),
    ],
}

DOCUMENT_TYPES: dict[str, list[str]] = {
    "Invoice": ["invoice", "bill to", "invoice number", "subtotal", "tax amount", "amount due"],
    "Receipt": ["receipt", "payment received", "transaction receipt", "thank you for your purchase"],
    "Resume / CV": ["resume", "curriculum vitae", "work experience", "professional summary", "skills"],
    "Salary Slip": ["payslip", "pay slip", "salary slip", "basic pay", "net salary", "gross salary"],
    "Academic Record": ["transcript", "marksheet", "mark sheet", "cgpa", "semester results", "grade point"],
    "Attendance Sheet": ["attendance sheet", "attendance report", "present days", "absent days"],
    "Admission Form": ["admission form", "application form", "applicant name", "course applied"],
    "Certificate": ["certificate", "certified that", "has successfully completed", "awarded to"],
    "Medical Report": ["medical report", "patient id", "diagnosis", "clinical findings", "discharge summary"],
    "Prescription": ["prescription", "medication", "dosage", "take once", "refills"],
    "Bank Statement": ["bank statement", "opening balance", "closing balance", "debit", "credit"],
    "Contract / Agreement": ["agreement", "hereby agrees", "terms and conditions", "party of the first part", "effective date"],
    "Lease Agreement": ["lease agreement", "tenant", "landlord", "monthly rent", "security deposit"],
    "Insurance Document": ["insurance policy", "policy number", "claim number", "coverage", "premium"],
    "Research Paper": ["abstract", "methodology", "literature review", "references", "conclusion"],
    "Report": ["executive summary", "findings", "recommendations", "report period", "analysis"],
    "Presentation": ["agenda", "slide", "speaker notes", "key takeaways"],
    "Identity Document": ["identity card", "date of birth", "document number", "national id", "passport number"],
    "Purchase Order": ["purchase order", "po number", "supplier", "quantity ordered"],
    "Tax Document": ["tax return", "gstin", "taxpayer", "tax period", "taxable income"],
    "Travel / Booking Document": ["itinerary", "booking reference", "reservation", "check-in", "check-out"],
    "Unknown Document": [],
}


def _normalise_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(name)).name).strip("._") or "file"


def detect_uploaded_file(filename: str, file_bytes: bytes) -> dict[str, str]:
    """Identify common file formats by signatures first, then extension/content."""
    name = str(filename or "uploaded_file")
    ext = Path(name).suffix.lower()
    data = bytes(file_bytes or b"")
    fmt = "unknown"
    category = "unknown"
    message = "File format is not recognised. It will be placed in Needs Review."

    # Signature-based identification.
    if data.startswith(b"%PDF-"):
        fmt, category = "pdf", "document"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        fmt, category = "png", "image"
    elif data.startswith(b"\xff\xd8\xff"):
        fmt, category = "jpg", "image"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        fmt, category = "gif", "image"
    elif data.startswith(b"BM"):
        fmt, category = "bmp", "image"
    elif data.startswith((b"II*\x00", b"MM\x00*")):
        fmt, category = "tiff", "image"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        fmt, category = "webp", "image"
    elif data.startswith(b"PAR1"):
        fmt, category = "parquet", "tabular"
    elif data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        if ext == ".xls":
            fmt, category = "xls", "tabular"
        else:
            fmt, category = "ole", "unknown"
    elif data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                if "xl/workbook.xml" in names:
                    fmt, category = "xlsx", "tabular"
                elif "word/document.xml" in names:
                    fmt, category = "docx", "document"
                elif "ppt/presentation.xml" in names:
                    fmt, category = "pptx", "document"
                else:
                    fmt, category = "zip", "archive"
        except Exception:
            fmt, category = "zip", "archive"

    # Extension fallback, only when the signature did not identify another type.
    if fmt == "unknown":
        extension_map = {
            ".csv": ("csv", "tabular"), ".tsv": ("tsv", "tabular"),
            ".xls": ("xls", "tabular"), ".xlsx": ("xlsx", "tabular"),
            ".json": ("json", "tabular"), ".jsonl": ("jsonl", "tabular"),
            ".ndjson": ("jsonl", "tabular"), ".parquet": ("parquet", "tabular"),
            ".pdf": ("pdf", "document"), ".docx": ("docx", "document"),
            ".pptx": ("pptx", "document"), ".txt": ("txt", "text"),
            ".md": ("md", "text"), ".markdown": ("md", "text"),
            ".html": ("html", "text"), ".htm": ("html", "text"),
            ".xml": ("xml", "text"), ".rtf": ("rtf", "text"),
            ".yaml": ("yaml", "text"), ".yml": ("yaml", "text"),
            ".log": ("log", "text"), ".sql": ("sql", "text"),
            ".png": ("png", "image"), ".jpg": ("jpg", "image"),
            ".jpeg": ("jpg", "image"), ".webp": ("webp", "image"),
            ".gif": ("gif", "image"), ".bmp": ("bmp", "image"),
            ".tif": ("tiff", "image"), ".tiff": ("tiff", "image"),
            ".zip": ("zip", "archive"),
        }
        if ext in extension_map:
            fmt, category = extension_map[ext]

    if category == "unknown" and data:
        # Last resort: accept valid text content, even if the extension is unusual.
        try:
            sample = data[:65536].decode("utf-8-sig")
            printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in sample)
            if sample and printable / max(len(sample), 1) > 0.92:
                fmt, category = "text", "text"
        except UnicodeDecodeError:
            pass

    if category != "unknown":
        message = "File format identified."
    if category == "archive":
        message = "Archive detected. Upload its contained files individually; archive extraction is not enabled."

    return {
        "filename": name,
        "format": fmt,
        "category": category,
        "supported": str(category in {"image", "tabular", "document", "text"}),
        "message": message,
    }


def classify_domain_text(text: str, filename: str = "") -> dict[str, Any]:
    """Classify document/plain text using transparent domain keyword evidence."""
    text = str(text or "")
    searchable = f"{filename}\n{text[:250000]}".lower()
    searchable = re.sub(r"[_\-]+", " ", searchable)
    scores: dict[str, float] = {domain: 0.0 for domain in DOMAIN_TERMS}
    evidence: dict[str, list[str]] = {domain: [] for domain in DOMAIN_TERMS}

    for domain, terms in DOMAIN_TERMS.items():
        for phrase, weight in terms:
            if phrase in searchable:
                scores[domain] += weight
                evidence[domain].append(phrase)

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Confidence is a heuristic score, not a calibrated probability.
    confidence = 0.0
    if best_score > 0:
        confidence = best_score / (best_score + second_score + 1.0)
        confidence = min(0.99, confidence)

    recognised = best_score >= 2.0 and (best_score - second_score) >= 0.5
    domain = best_domain if recognised else UNKNOWN_DOMAIN

    type_scores = {
        label: sum(1 for phrase in phrases if phrase in searchable)
        for label, phrases in DOCUMENT_TYPES.items()
    }
    document_type = max(type_scores, key=type_scores.get) if type_scores else "Unknown Document"
    if not type_scores or type_scores[document_type] == 0:
        document_type = "Unknown Document"

    return {
        "domain": domain,
        "subcategory": document_type if document_type != "Unknown Document" else "General Document",
        "document_type": document_type,
        "status": "recognised" if recognised else "needs_review",
        "confidence": round(confidence, 3),
        "evidence": evidence.get(best_domain, [])[:8] if recognised else [],
        "top_candidates": [
            {"domain": label, "score": round(score, 2)}
            for label, score in ranked[:5] if score > 0
        ],
    }


def read_text_file(file_bytes: bytes, file_format: str) -> str:
    """Extract readable text from common text-based formats."""
    raw = bytes(file_bytes or b"")
    text = ""
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        return ""

    if file_format == "html":
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(text)
    elif file_format == "xml":
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(text)
    elif file_format == "rtf":
        text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
        text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
        text = text.replace("{", " ").replace("}", " ")
    elif file_format in {"json", "jsonl"}:
        try:
            obj = json.loads(text) if file_format == "json" else [json.loads(line) for line in text.splitlines() if line.strip()]
            text = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            pass
    return re.sub(r"\s+", " ", text).strip()


def load_tabular_file(filename: str, file_bytes: bytes, file_format: str, nrows: int | None = None) -> list[dict[str, Any]]:
    """Load supported non-Excel tabular formats into a sheet-like structure."""
    stream = io.BytesIO(file_bytes)
    if file_format in {"csv", "tsv"}:
        if file_format == "tsv":
            df = pd.read_csv(stream, sep="\t", low_memory=False, nrows=nrows)
        else:
            # Sniff delimiter from a sample; default to comma.
            sample = file_bytes[:8192].decode("utf-8-sig", errors="replace")
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except Exception:
                delimiter = ","
            stream.seek(0)
            df = pd.read_csv(stream, sep=delimiter, low_memory=False, nrows=nrows)
        sheet_name = "Data"
    elif file_format in {"json", "jsonl"}:
        df = pd.read_json(stream, lines=(file_format == "jsonl"))
        if len(df.columns) == 1 and df.shape[0] == 1 and isinstance(df.iloc[0, 0], (dict, list)):
            df = pd.json_normalize(df.iloc[0, 0])
        sheet_name = "Data"
    elif file_format == "parquet":
        df = pd.read_parquet(stream)
        sheet_name = "Data"
    else:
        raise ValueError(f"No tabular reader is configured for {file_format!r}.")

    return [{
        "sheet_name": sheet_name,
        "dataframe": df,
        "row_count": len(df),
        "column_count": len(df.columns),
    }]


def dataframe_to_text(df: pd.DataFrame, max_rows: int = 15) -> str:
    """Create a compact representation for fallback domain classification."""
    if df is None:
        return ""
    sample = df.head(max_rows).fillna("").astype(str).to_string(index=False)
    return "Columns: " + ", ".join(map(str, df.columns)) + "\n" + sample


def repair_dataframe(
    dataframe: pd.DataFrame,
    *,
    remove_duplicates: bool = False,
    fill_missing: bool = False,
) -> dict[str, Any]:
    """Apply conservative, user-selected repairs and return a change report.

    Always trims outer whitespace and converts whitespace-only strings to NA.
    Duplicate removal and median/mode filling happen only when selected.
    Outliers are reported for review and are never silently changed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    cleaned = dataframe.copy(deep=True)
    rows_before, cols_before = cleaned.shape
    missing_before = int(cleaned.isna().sum().sum())
    duplicates_before = int(cleaned.duplicated().sum())
    trimmed_cells = 0

    for column in cleaned.columns:
        series = cleaned[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            as_string = series.astype("string")
            stripped = as_string.str.strip()
            trimmed_cells += int((as_string.notna() & (as_string != stripped)).sum())
            stripped = stripped.replace(r"^\s*$", pd.NA, regex=True)
            cleaned[column] = stripped

    duplicates_after_trim = int(cleaned.duplicated().sum())
    removed_duplicates = 0
    if remove_duplicates and duplicates_after_trim:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().copy()
        removed_duplicates = before - len(cleaned)

    filled_cells = 0
    if fill_missing:
        identifier_markers = (
            "id", "identifier", "roll", "registration", "email", "phone",
            "mobile", "name", "address", "dob", "birth", "date",
            "serial", "code", "reference", "account", "passport",
            "invoice", "transaction", "student", "employee",
        )
        for column in cleaned.columns:
            column_name = str(column).lower().replace("_", " ").replace("-", " ")
            if any(marker in column_name for marker in identifier_markers):
                continue
            missing_mask = cleaned[column].isna()
            count = int(missing_mask.sum())
            if not count:
                continue
            series = cleaned[column]
            replacement = None
            if pd.api.types.is_numeric_dtype(series):
                non_missing = series.dropna()
                if not non_missing.empty:
                    replacement = non_missing.median()
            else:
                non_missing = series.dropna()
                # Avoid filling free-text/high-cardinality categories with a repeated mode.
                if not non_missing.empty and non_missing.nunique() <= max(20, int(len(non_missing) * 0.5)):
                    modes = non_missing.mode()
                    if not modes.empty:
                        replacement = modes.iloc[0]
            if replacement is not None and not (isinstance(replacement, float) and math.isnan(replacement)):
                cleaned.loc[missing_mask, column] = replacement
                filled_cells += count

    missing_after = int(cleaned.isna().sum().sum())
    duplicates_after = int(cleaned.duplicated().sum())

    email_invalid = 0
    email_columns = [c for c in cleaned.columns if "email" in str(c).lower()]
    for column in email_columns:
        values = cleaned[column].dropna().astype(str)
        email_invalid += int((~values.str.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", na=False)).sum())

    outlier_counts: dict[str, int] = {}
    invalid_ranges: dict[str, int] = {}
    for column in cleaned.select_dtypes(include="number").columns:
        values = cleaned[column].dropna()
        if len(values) < 1:
            continue
        column_name = str(column).lower().replace("_", " ").replace("-", " ")
        if "cgpa" in column_name or re.search(r"\bgpa\b", column_name):
            invalid_count = int(((values < 0) | (values > 10)).sum())
            if invalid_count:
                invalid_ranges[str(column)] = invalid_count
        elif "attendance" in column_name or "percentage" in column_name or "percent" in column_name:
            invalid_count = int(((values < 0) | (values > 100)).sum())
            if invalid_count:
                invalid_ranges[str(column)] = invalid_count
        elif re.search(r"\bage\b", column_name):
            invalid_count = int(((values < 0) | (values > 120)).sum())
            if invalid_count:
                invalid_ranges[str(column)] = invalid_count

        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        if count:
            outlier_counts[str(column)] = count

    suggestions: list[str] = []
    if missing_after:
        suggestions.append(f"{missing_after} missing cells remain. Review them before filling; no values were invented.")
    if duplicates_after:
        suggestions.append(f"{duplicates_after} exact duplicate rows remain. Enable duplicate removal only if duplicates are unwanted.")
    if email_invalid:
        suggestions.append(f"{email_invalid} email value(s) do not match a basic email format. Review manually.")
    if invalid_ranges:
        details = ", ".join(f"{column}: {count}" for column, count in invalid_ranges.items())
        suggestions.append("Potential domain-range violations were found (CGPA/GPA 0–10, attendance/percentage 0–100, age 0–120): " + details + ". Values were not automatically changed.")
    if outlier_counts:
        suggestions.append("Potential numeric outliers were detected. They were flagged, not automatically changed.")
    if not suggestions:
        suggestions.append("No remaining issues were found by the current basic checks. Domain-specific review may still be needed.")

    return {
        "dataframe": cleaned,
        "summary": {
            "rows_before": rows_before,
            "rows_after": len(cleaned),
            "columns": cols_before,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "duplicates_before": duplicates_before,
            "duplicates_after": duplicates_after,
            "whitespace_cells_trimmed": trimmed_cells,
            "duplicates_removed": removed_duplicates,
            "missing_cells_filled": filled_cells,
            "invalid_email_values": email_invalid,
            "potential_outliers": outlier_counts,
            "invalid_range_values": invalid_ranges,
        },
        "suggestions": suggestions,
    }


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(sheets: list[dict[str, Any]]) -> bytes:
    output = io.BytesIO()
    used_names: set[str] = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for index, sheet in enumerate(sheets, start=1):
            raw_name = str(sheet.get("sheet_name") or f"Sheet{index}")
            safe_name = re.sub(r"[\\/*?:\[\]]", "_", raw_name)[:31] or f"Sheet{index}"
            base = safe_name
            suffix = 1
            while safe_name.lower() in used_names:
                extra = f"_{suffix}"
                safe_name = base[:31 - len(extra)] + extra
                suffix += 1
            used_names.add(safe_name.lower())
            sheet["dataframe"].to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()


def create_segregation_archive(records: list[dict[str, Any]]) -> str:
    """Create a disk-backed ZIP with originals grouped by domain and cleaned outputs.

    Original files are never modified. Cleaned table copies and extracted text are
    placed in separate folders within the archive.
    """
    handle = tempfile.NamedTemporaryFile(prefix="datasentinel_segregated_", suffix=".zip", delete=False)
    archive_path = handle.name
    handle.close()

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        used_paths: set[str] = set()

        def write_unique(path: str, payload: bytes) -> None:
            base = path
            counter = 1
            while path.lower() in used_paths:
                stem, ext = os.path.splitext(base)
                path = f"{stem}_{counter}{ext}"
                counter += 1
            used_paths.add(path.lower())
            archive.writestr(path, payload)

        manifest = []
        for record in records:
            filename = _normalise_name(record.get("filename", "file"))
            domain = _normalise_name(record.get("domain", UNKNOWN_DOMAIN))
            original = record.get("file_bytes") or b""
            if original:
                write_unique(f"Original Files/{domain}/{filename}", original)

            manifest.append({
                "filename": record.get("filename", ""),
                "format": record.get("format", "unknown"),
                "file_category": record.get("category", "unknown"),
                "domain": record.get("domain", UNKNOWN_DOMAIN),
                "subcategory": record.get("subcategory", ""),
                "document_type": record.get("document_type", ""),
                "status": record.get("status", "unknown"),
                "confidence_or_ranking_score": record.get("confidence", 0.0),
                "content_similarity_group": record.get("similarity_group"),
                "visual_similarity_group": record.get("visual_similarity_group"),
                "sheet_categories": [
                    {"sheet_name": sheet.get("sheet_name"), "domain": sheet.get("domain")}
                    for sheet in (record.get("sheets", []) or [])
                ],
            })

            # Save a compact description of image analysis; omit the raw embedding.
            image_analysis = record.get("image_analysis") or {}
            if image_analysis:
                safe_metadata = {
                    key: image_analysis.get(key)
                    for key in (
                        "visual_group", "predicted_category", "relative_score", "group_relative_score",
                        "needs_review", "top_predictions", "caption", "visible_expression",
                        "detected_objects", "tags", "custom_model_prediction", "custom_model_used", "notes",
                    )
                }
                write_unique(
                    f"AI Image Analysis/{domain}/{Path(filename).stem}_analysis.json",
                    json.dumps(safe_metadata, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                )

            # Write corrected table data under the category of each sheet.
            sheet_exports = record.get("sheets", []) or []
            for index, sheet in enumerate(sheet_exports, start=1):
                df = sheet.get("cleaned_dataframe")
                if not isinstance(df, pd.DataFrame):
                    continue
                sheet_domain = _normalise_name(sheet.get("domain", domain))
                sheet_name = _normalise_name(sheet.get("sheet_name", f"Sheet{index}"))
                stem = Path(filename).stem
                csv_bytes = dataframe_to_csv_bytes(df)
                sampled_suffix = "_sampled_first_20k_rows" if sheet.get("sampled") else "_cleaned"
                write_unique(
                    f"Cleaned Data/{sheet_domain}/{stem}__{sheet_name}{sampled_suffix}.csv",
                    csv_bytes,
                )

            # Add a rebuilt Excel workbook with all cleaned sheets as a convenience.
            if record.get("format") in {"xlsx", "xls"} and sheet_exports:
                cleaned_sheet_payload = [
                    {"sheet_name": s.get("sheet_name", f"Sheet{i}"), "dataframe": s["cleaned_dataframe"]}
                    for i, s in enumerate(sheet_exports, start=1)
                    if isinstance(s.get("cleaned_dataframe"), pd.DataFrame)
                ]
                if cleaned_sheet_payload:
                    workbook_bytes = dataframe_to_xlsx_bytes(cleaned_sheet_payload)
                    workbook_sampled = any(s.get("sampled") for s in sheet_exports)
                    workbook_suffix = "_sampled_first_20k_rows" if workbook_sampled else "_cleaned"
                    write_unique(
                        f"Cleaned Workbooks/{domain}/{Path(filename).stem}{workbook_suffix}.xlsx",
                        workbook_bytes,
                    )

            extracted_text = record.get("extracted_text")
            if extracted_text:
                text_bytes = str(extracted_text).encode("utf-8")
                write_unique(
                    f"Extracted Text/{domain}/{Path(filename).stem}_extracted.txt",
                    text_bytes,
                )

        write_unique(
            "Segregation Summary/analysis_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        )

    return archive_path
