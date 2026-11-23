"""Batch upload validation for DataSentinel.

Limits are defined in bytes and kept as aliases for compatibility with
older versions of the module.
"""

from __future__ import annotations

import hashlib
from typing import Any


# ---------------------------------------------------------------------------
# UPLOAD LIMITS
# ---------------------------------------------------------------------------

MAX_FILES_PER_BATCH = 100

# 2 GiB per file and 2 GiB combined per batch.
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BATCH_SIZE = 2 * 1024 * 1024 * 1024

# Compatibility aliases used by earlier versions of DataSentinel.
MAX_TOTAL_BATCH_SIZE_BYTES = MAX_TOTAL_BATCH_SIZE
MAX_TOTAL_SIZE_BYTES = MAX_TOTAL_BATCH_SIZE
MAX_SINGLE_FILE_SIZE_MB = 2048
MAX_TOTAL_BATCH_SIZE_MB = 2048

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}


def _get_file_bytes(uploaded_file: Any) -> bytes:
    """Read a Streamlit UploadedFile safely into bytes."""
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        # Fallback for file-like objects used in tests or other callers.
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        data = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("Uploaded file did not return binary data.")

    return bytes(data)


def process_uploaded_files(uploaded_files: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    """Validate an uploaded batch and return ready files plus per-file status.

    Each ready result contains the fields used by app.py:
    ``filename``, ``status``, ``message``, and ``file_bytes``.
    """
    if uploaded_files is None:
        uploaded_files = []

    uploaded_files = list(uploaded_files)
    results: list[dict[str, Any]] = []

    if not uploaded_files:
        return {
            "results": results,
            "total_files": 0,
            "ready_files": 0,
            "rejected_files": 0,
            "total_size_bytes": 0,
        }

    if len(uploaded_files) > MAX_FILES_PER_BATCH:
        for index, uploaded_file in enumerate(uploaded_files):
            filename = getattr(uploaded_file, "name", f"file_{index + 1}")
            if index >= MAX_FILES_PER_BATCH:
                results.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "message": (
                            f"Batch limit is {MAX_FILES_PER_BATCH} files. "
                            "This file was not processed."
                        ),
                        "file_bytes": None,
                    }
                )
                continue
            # Continue validating the first permitted files below.
        uploaded_files = uploaded_files[:MAX_FILES_PER_BATCH]

    total_size_bytes = 0
    seen_hashes: set[str] = set()
    processed: list[dict[str, Any]] = []

    for index, uploaded_file in enumerate(uploaded_files):
        filename = getattr(uploaded_file, "name", f"file_{index + 1}")
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if suffix not in SUPPORTED_EXTENSIONS:
            processed.append(
                {
                    "filename": filename,
                    "status": "unsupported",
                    "message": f"Unsupported file type: {suffix or 'no extension'}.",
                    "file_bytes": None,
                }
            )
            continue

        try:
            # Use Streamlit's size metadata first when available, to avoid
            # loading an oversized file into memory unnecessarily.
            declared_size = getattr(uploaded_file, "size", None)
            if declared_size is not None and declared_size > MAX_FILE_SIZE_BYTES:
                processed.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "message": "File exceeds the 2 GB per-file limit.",
                        "file_bytes": None,
                    }
                )
                continue

            file_bytes = _get_file_bytes(uploaded_file)
            file_size = len(file_bytes)

            if file_size > MAX_FILE_SIZE_BYTES:
                processed.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "message": "File exceeds the 2 GB per-file limit.",
                        "file_bytes": None,
                    }
                )
                continue

            if total_size_bytes + file_size > MAX_TOTAL_BATCH_SIZE:
                processed.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "message": "The batch would exceed the 2 GB total upload limit.",
                        "file_bytes": None,
                    }
                )
                continue

            total_size_bytes += file_size
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if file_hash in seen_hashes:
                processed.append(
                    {
                        "filename": filename,
                        "status": "duplicate",
                        "message": "Duplicate file content detected; already uploaded in this batch.",
                        "file_bytes": None,
                    }
                )
                continue

            seen_hashes.add(file_hash)
            processed.append(
                {
                    "filename": filename,
                    "status": "ready",
                    "message": "File is ready for analysis.",
                    "file_bytes": file_bytes,
                    "size_bytes": file_size,
                    "sha256": file_hash,
                }
            )

        except Exception as exc:
            processed.append(
                {
                    "filename": filename,
                    "status": "error",
                    "message": f"Could not read uploaded file: {exc}",
                    "file_bytes": None,
                }
            )

    # Preserve any over-limit-count messages added above.
    results.extend(processed)
    ready_count = sum(item["status"] == "ready" for item in results)

    return {
        "results": results,
        "total_files": len(results),
        "ready_files": ready_count,
        "rejected_files": len(results) - ready_count,
        "total_size_bytes": total_size_bytes,
    }
