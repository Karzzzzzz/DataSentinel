"""Similarity grouping for mixed DataSentinel uploads.

Groups semantically similar extracted text/captions across file types and computes
separate visual-similarity groups for images using their CLIP embeddings.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _record_text(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("semantic_text", "")),
        str(record.get("extracted_text", "") or "")[:4000],
    ]
    image = record.get("image_analysis") or {}
    parts.extend([
        str(image.get("caption", "")),
        " ".join(map(str, image.get("tags", []) or [])),
        " ".join(str(x.get("label", "")) for x in image.get("detected_objects", []) or []),
    ])
    for sheet in record.get("sheets", []) or []:
        parts.append(str(sheet.get("sheet_name", "")))
        parts.append(str(sheet.get("subcategory", "")))
        parts.extend(str(x.get("Field", "")) for x in sheet.get("field_rows", [])[:30])
    return " ".join(x for x in parts if x).strip()


def _components(similarity: np.ndarray, threshold: float) -> list[int]:
    count = similarity.shape[0]
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(count):
        for j in range(i + 1, count):
            if similarity[i, j] >= threshold:
                union(i, j)
    roots: dict[int, int] = {}
    groups = []
    for i in range(count):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots) + 1
        groups.append(roots[root])
    return groups


def assign_similarity_groups(
    records: list[dict[str, Any]],
    *,
    text_threshold: float = 0.22,
    visual_threshold: float = 0.84,
) -> dict[str, Any]:
    """Attach cross-file text similarity and image-only visual similarity groups."""
    if not records:
        return {"text_groups": 0, "visual_groups": 0, "method": "none"}

    texts = [_record_text(record) for record in records]
    valid = [i for i, text in enumerate(texts) if len(text) >= 8]
    method = "TF-IDF lexical similarity"
    text_group_count = 0
    if len(valid) >= 2:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000, stop_words="english")
            matrix = vectorizer.fit_transform([texts[i] for i in valid])
            similarities = cosine_similarity(matrix)
            groups = _components(similarities, text_threshold)
            group_sizes: dict[int, int] = {}
            for group_id in groups:
                group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
            group_map = {
                original: (groups[pos] if group_sizes[groups[pos]] > 1 else None)
                for pos, original in enumerate(valid)
            }
            for idx, record in enumerate(records):
                record["similarity_group"] = group_map.get(idx, None)
                record["similarity_method"] = method if group_map.get(idx) is not None else "no close text match"
            text_group_count = len({group for group in groups if group_sizes[group] > 1})
        except Exception as exc:
            for record in records:
                record["similarity_group"] = None
                record["similarity_method"] = f"unavailable: {exc}"
    else:
        for record in records:
            record["similarity_group"] = None
            record["similarity_method"] = "insufficient text"

    image_indices = []
    image_vectors = []
    for idx, record in enumerate(records):
        vector = record.get("visual_embedding")
        if vector is None:
            continue
        try:
            arr = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(arr)
            if norm > 0:
                image_indices.append(idx)
                image_vectors.append(arr / norm)
        except Exception:
            continue

    visual_group_count = 0
    if len(image_vectors) >= 2:
        matrix = np.vstack(image_vectors)
        similarities = matrix @ matrix.T
        visual_groups = _components(similarities, visual_threshold)
        visual_sizes: dict[int, int] = {}
        for group_id in visual_groups:
            visual_sizes[group_id] = visual_sizes.get(group_id, 0) + 1
        visual_map = {
            original: (visual_groups[pos] if visual_sizes[visual_groups[pos]] > 1 else None)
            for pos, original in enumerate(image_indices)
        }
        for idx, record in enumerate(records):
            record["visual_similarity_group"] = visual_map.get(idx)
        visual_group_count = len({group for group in visual_groups if visual_sizes[group] > 1})
    else:
        for record in records:
            record["visual_similarity_group"] = None

    return {
        "text_groups": text_group_count,
        "visual_groups": visual_group_count,
        "method": method,
        "text_threshold": text_threshold,
        "visual_threshold": visual_threshold,
    }
