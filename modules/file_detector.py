
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, BadZipFile
import csv


def detect_file_type(file_name: str, file_bytes: bytes) -> dict:
    """
    Detect the actual format of an uploaded file using
    its binary signature and internal structure.
    """

    extension = Path(file_name).suffix.lower()

    result = {
        "filename": file_name,
        "format": "unknown",
        "category": "unknown",
        "supported": False,
        "message": "Unrecognised or unsupported file format."
    }

    if not isinstance(file_bytes, bytes) or not file_bytes:
        result["format"] = "empty"
        result["message"] = "The uploaded file is empty."
        return result

    detected = None
    category = "unknown"

    # PDF
    if file_bytes.startswith(b"%PDF-"):
        detected, category = "pdf", "document"

    # PNG
    elif file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        detected, category = "png", "image"

    # JPEG
    elif file_bytes.startswith(b"\xff\xd8\xff"):
        detected, category = "jpg", "image"

    # GIF
    elif file_bytes.startswith((b"GIF87a", b"GIF89a")):
        detected, category = "gif", "image"

    # WEBP
    elif (
        len(file_bytes) >= 12
        and file_bytes[:4] == b"RIFF"
        and file_bytes[8:12] == b"WEBP"
    ):
        detected, category = "webp", "image"

    # BMP
    elif file_bytes.startswith(b"BM"):
        detected, category = "bmp", "image"

    # TIFF
    elif file_bytes.startswith((b"II*\x00", b"MM\x00*")):
        detected, category = "tiff", "image"

    # Modern Office formats (ZIP containers)
    elif file_bytes.startswith(b"PK"):
        try:
            with ZipFile(BytesIO(file_bytes)) as archive:
                names = set(archive.namelist())

                if "xl/workbook.xml" in names:
                    detected, category = "xlsx", "spreadsheet"

                elif "word/document.xml" in names:
                    detected, category = "docx", "document"

                elif "ppt/presentation.xml" in names:
                    detected, category = "pptx", "document"

        except BadZipFile:
            pass

    # Legacy Excel format
    elif file_bytes.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        if extension == ".xls":
            detected, category = "xls", "spreadsheet"

    # CSV and TXT
    if detected is None:

        try:
            text = file_bytes.decode("utf-8-sig")

            if "\x00" not in text:

                if extension == ".txt":
                    detected, category = "txt", "text"

                elif extension == ".csv":
                    try:
                        rows = list(csv.reader(text.splitlines()[:20]))

                        if len(rows) >= 2 and len(rows[0]) >= 2:
                            detected, category = "csv", "spreadsheet"
                        else:
                            detected, category = "csv", "spreadsheet"

                    except csv.Error:
                        detected, category = None, "unknown"

                elif extension == ".csv":
                    detected, category = "csv", "spreadsheet"

                elif extension == ".txt":
                    detected, category = "txt", "text"

        except UnicodeDecodeError:
            pass

    # Verify detected format against supported formats
    supported_formats = {
        "pdf", "png", "jpg", "gif", "webp",
        "bmp", "tiff", "xls", "xlsx", "csv",
        "docx", "pptx", "txt"
    }

    if detected:
        result["format"] = detected
        result["category"] = category
        result["supported"] = detected in supported_formats
        result["message"] = "File format identified successfully."

    return result
