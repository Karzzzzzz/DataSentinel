
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from modules.classification_features import (
    build_dataset_profile,
    normalise_text
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "training"
    / "saved_models"
    / "dataset_classifier.joblib"
)


def load_classifier():
    """Load the saved dataset classification model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Classifier model not found. Run "
            "python training/train_classifier.py first."
        )

    return joblib.load(MODEL_PATH)


def classify_dataset(
    dataframe: pd.DataFrame,
    top_k: int = 7,
    min_similarity: float = 0.10,
    min_domain_support: float = 0.45
) -> dict:
    """
    Classify an uploaded dataset using TF-IDF similarity.

    Improvements:
    - Aligns uploaded headers with training features.
    - Aggregates evidence by domain and subcategory.
    - Returns alternative classifications.
    - Supports Unknown / Needs Review.
    """

    try:

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        if dataframe.empty or len(dataframe.columns) == 0:
            raise ValueError("Dataset is empty.")

        model = load_classifier()

        vectorizer = model["vectorizer"]
        training_matrix = model["training_matrix"]
        training_records = model["training_records"]

        if training_matrix.shape[0] != len(training_records):
            raise ValueError("Training model is inconsistent.")

        # Extract uploaded dataset structure
        profile = build_dataset_profile(dataframe)

        # IMPORTANT:
        # Training uses sample_text column names.
        # Therefore, query features use column names first,
        # avoiding unrelated generic descriptive words.
        query_parts = []

        for column in profile["columns"]:

            query_parts.append(
                normalise_text(column["original_name"])
            )

            # Add descriptive sample values where useful.
            query_parts.extend(
                normalise_text(value)
                for value in column["sample_values"]
                if str(value).strip()
            )

        query_text = " ".join(query_parts)

        # Convert query to TF-IDF features
        query_vector = vectorizer.transform([query_text])

        if query_vector.nnz == 0:
            return {
                "success": True,
                "status": "unknown",
                "domain": "Unknown",
                "subcategory": "Unknown",
                "dataset_type": "Unknown",
                "similarity_score": 0.0,
                "alternatives": [],
                "message": "No matching classification patterns found."
            }

        # Compare against training examples
        similarities = cosine_similarity(
            query_vector,
            training_matrix
        ).flatten()

        count = min(top_k, len(training_records))

        top_indices = similarities.argsort()[::-1][:count]

        candidates = []

        for index in top_indices:

            record = training_records[index]
            score = float(similarities[index])

            candidates.append({
                "domain": record["domain"],
                "subcategory": record["subcategory"],
                "dataset_type": record["dataset_type"],
                "similarity_score": score
            })

        if not candidates:
            raise ValueError("No classification candidates found.")

        best_similarity = candidates[0]["similarity_score"]

        # Aggregate evidence by main domain
        domain_scores = {}

        for item in candidates:

            domain = item["domain"]

            weight = item["similarity_score"] ** 2

            domain_scores[domain] = (
                domain_scores.get(domain, 0.0) + weight
            )

        ranked_domains = sorted(
            domain_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_domain = ranked_domains[0][0]

        total_domain_weight = sum(domain_scores.values())

        domain_support = (
            domain_scores[best_domain] / total_domain_weight
            if total_domain_weight > 0 else 0.0
        )

        # Consider only candidates belonging to the selected domain
        domain_candidates = [
            item for item in candidates
            if item["domain"] == best_domain
        ]

        # Aggregate subcategory evidence
        subcategory_scores = {}

        for item in domain_candidates:

            key = item["subcategory"]

            subcategory_scores[key] = (
                subcategory_scores.get(key, 0.0)
                + item["similarity_score"] ** 2
            )

        best_subcategory = max(
            subcategory_scores,
            key=subcategory_scores.get
        )

        # Find the strongest matching dataset type
        type_candidates = [
            item for item in domain_candidates
            if item["subcategory"] == best_subcategory
        ]

        best_match = max(
            type_candidates,
            key=lambda item: item["similarity_score"]
        )

        # Determine whether the result is sufficiently supported
        needs_review = (
            best_similarity < min_similarity
            or domain_support < min_domain_support
        )

        if needs_review:

            status = "needs_review"

            predicted_domain = "Unknown / Needs Review"
            predicted_subcategory = "Unknown"
            predicted_type = "Unknown"

        else:

            status = "recognised"

            predicted_domain = best_domain
            predicted_subcategory = best_subcategory
            predicted_type = best_match["dataset_type"]

        # Return alternative predictions
        alternatives = []

        for item in candidates:

            alternatives.append({
                "domain": item["domain"],
                "subcategory": item["subcategory"],
                "dataset_type": item["dataset_type"],
                "similarity_score": round(
                    item["similarity_score"], 4
                )
            })

        return {
            "success": True,
            "status": status,

            "domain": predicted_domain,
            "subcategory": predicted_subcategory,
            "dataset_type": predicted_type,

            "similarity_score": round(
                best_similarity, 4
            ),

            "domain_support": round(
                domain_support, 4
            ),

            "alternatives": alternatives,

            "dataset_profile": profile,

            "message": (
                "Dataset needs manual classification review."
                if needs_review
                else "Dataset classification completed."
            )
        }

    except Exception as error:

        return {
            "success": False,
            "status": "error",
            "error": str(error)
        }
