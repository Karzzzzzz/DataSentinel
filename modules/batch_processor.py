"""Batch upload validation for DataSentinel 2.0.

Accepts files regardless of extension. Classification and parsing are handled
later; unknown formats are retained and routed to Needs Review. Files are never
executed. Limits: 100 files, 2 GiB per file, 2 GiB total per batch.
"""

from __future__ import annotations

import hashlib
from typing import Any

MAX_FILES_PER_BATCH = 100
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BATCH_SIZE = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BATCH_SIZE_BYTES = MAX_TOTAL_BATCH_SIZE
MAX_TOTAL_SIZE_BYTES = MAX_TOTAL_BATCH_SIZE
MAX_SINGLE_FILE_SIZE_MB = 2048
MAX_TOTAL_BATCH_SIZE_MB = 2048


def _get_file_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        data = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("Uploaded file did not return binary data.")
    return bytes(data)


def process_uploaded_files(uploaded_files: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    if uploaded_files is None:
        uploaded_files = []
    uploaded_files = list(uploaded_files)
    results: list[dict[str, Any]] = []

    if not uploaded_files:
        return {"results": [], "total_files": 0, "ready_files": 0,
                "rejected_files": 0, "total_size_bytes": 0}

    extra_files = uploaded_files[MAX_FILES_PER_BATCH:]
    uploaded_files = uploaded_files[:MAX_FILES_PER_BATCH]
    total_size_bytes = 0
    seen_hashes: set[str] = set()

    for index, uploaded_file in enumerate(uploaded_files):
        filename = getattr(uploaded_file, "name", f"file_{index + 1}")
        try:
            declared_size = getattr(uploaded_file, "size", None)
            if declared_size is not None and declared_size > MAX_FILE_SIZE_BYTES:
                results.append({"filename": filename, "status": "error",
                                "message": "File exceeds the 2 GiB per-file limit.",
                                "file_bytes": None})
                continue

            file_bytes = _get_file_bytes(uploaded_file)
            file_size = len(file_bytes)
            if file_size > MAX_FILE_SIZE_BYTES:
                results.append({"filename": filename, "status": "error",
                                "message": "File exceeds the 2 GiB per-file limit.",
                                "file_bytes": None})
                continue
            if total_size_bytes + file_size > MAX_TOTAL_BATCH_SIZE:
                results.append({"filename": filename, "status": "error",
                                "message": "The batch would exceed the 2 GiB total limit.",
                                "file_bytes": None})
                continue

            file_hash = hashlib.sha256(file_bytes).hexdigest()
            if file_hash in seen_hashes:
                results.append({"filename": filename, "status": "duplicate",
                                "message": "Duplicate file content detected in this batch.",
                                "file_bytes": None})
                continue

            seen_hashes.add(file_hash)
            total_size_bytes += file_size
            results.append({
                "filename": filename,
                "status": "ready",
                "message": "File accepted for format detection and classification.",
                "file_bytes": file_bytes,
                "size_bytes": file_size,
                "sha256": file_hash,
            })
        except Exception as exc:
            results.append({"filename": filename, "status": "error",
                            "message": f"Could not read uploaded file: {exc}",
                            "file_bytes": None})

    for index, uploaded_file in enumerate(extra_files, start=MAX_FILES_PER_BATCH + 1):
        filename = getattr(uploaded_file, "name", f"file_{index}")
        results.append({"filename": filename, "status": "error",
                        "message": f"Batch limit is {MAX_FILES_PER_BATCH} files.",
                        "file_bytes": None})

    ready_count = sum(item["status"] == "ready" for item in results)
    return {
        "results": results,
        "total_files": len(results),
        "ready_files": ready_count,
        "rejected_files": len(results) - ready_count,
        "total_size_bytes": total_size_bytes,
    }
