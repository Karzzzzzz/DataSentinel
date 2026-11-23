"""Human-feedback collection and CLIP-embedding classifier training."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

import numpy as np
import pandas as pd

from modules.image_ai.advanced_image_analyzer import _pil_image, _clip_rank, MODEL_NAME


def count_image_training_examples() -> dict[str, int]:
    path = Path("training/image_labels.csv")
    if not path.exists():
        return {"total": 0, "classes": 0}
    try:
        df = pd.read_csv(path)
        if "label" not in df.columns:
            return {"total": 0, "classes": 0}
        return {"total": int(len(df)), "classes": int(df["label"].nunique())}
    except Exception:
        return {"total": 0, "classes": 0}


def train_custom_image_classifier(min_examples_per_class: int = 5) -> dict[str, Any]:
    """Train a supervised classifier on user-confirmed CLIP embeddings.

    CLIP is kept frozen; LogisticRegression learns the user's visual categories.
    A held-out split is reported before the final model is refit on all examples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib

    metadata_path = Path("training/image_labels.csv")
    if not metadata_path.exists():
        raise ValueError("No confirmed image labels found. Correct and save image examples first.")
    metadata = pd.read_csv(metadata_path)
    required = {"file_path", "label"}
    if not required.issubset(metadata.columns):
        raise ValueError("training/image_labels.csv is missing file_path or label columns.")

    rows = []
    for row in metadata.to_dict("records"):
        path = Path(str(row["file_path"]))
        if path.exists() and str(row.get("label", "")).strip():
            rows.append((path, str(row["label"]).strip()))
    if not rows:
        raise ValueError("No saved training images were found. Check training/image_examples.")

    counts = pd.Series([label for _, label in rows]).value_counts()
    eligible = set(counts[counts >= min_examples_per_class].index)
    rows = [(path, label) for path, label in rows if label in eligible]
    if len(eligible) < 2:
        need = ", ".join(f"{label}: {count}" for label, count in counts.items())
        raise ValueError(
            f"At least two categories need {min_examples_per_class} images each before training. "
            f"Current counts: {need or 'none'}. Aim for 50–200 varied examples per category for stronger results."
        )

    embeddings, labels = [], []
    for index, (path, label) in enumerate(rows, start=1):
        try:
            image_bytes = path.read_bytes()
            image = _pil_image(image_bytes)
            _, embedding = _clip_rank(image, [f"a photograph of {label}"])
            embeddings.append(embedding)
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels)
    counts = pd.Series(y).value_counts()
    if len(counts[counts >= min_examples_per_class]) < 2:
        raise ValueError("Too few valid image examples remain after reading the training images.")

    # Keep only categories with enough successfully encoded examples.
    keep = set(counts[counts >= min_examples_per_class].index)
    mask = np.asarray([label in keep for label in y])
    X, y = X[mask], y[mask]
    class_count = len(set(y))
    test_count = max(class_count, int(np.ceil(0.2 * len(y))))
    if len(y) - test_count < class_count:
        raise ValueError("Not enough images for a stratified train/test split. Add more examples per category.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=42, stratify=y
    )
    evaluation_model = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    final_model = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
    final_model.fit(X, y)
    output = Path("training/saved_models/image_classifier.joblib")
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "encoder_name": MODEL_NAME, "labels": sorted(set(y))}, output)

    return {
        "accuracy": accuracy,
        "test_samples": int(len(y_test)),
        "training_samples": int(len(y_train)),
        "total_labelled_samples": int(len(y)),
        "classes": sorted(set(y)),
        "report": report,
        "model_path": output.as_posix(),
        "note": "The score is a held-out test estimate for the labelled categories, not a guarantee on every image in the world.",
    }
