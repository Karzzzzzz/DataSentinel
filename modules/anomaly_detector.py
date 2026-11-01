
import numpy as np
import pandas as pd

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    df,
    contamination=0.05,
    algorithm="Isolation Forest"
):
    """Detect unusual records using selectable PyOD algorithms."""

    data = df.copy()

    numeric_data = data.select_dtypes(
        include=["number"]
    ).copy()

    excluded_words = [
        "id", "target", "label", "outcome", "status"
    ]

    feature_columns = [
        col for col in numeric_data.columns
        if not any(
            word in str(col).lower()
            for word in excluded_words
        )
    ]

    numeric_data = numeric_data[feature_columns]

    if numeric_data.empty:
        raise ValueError("No suitable numerical features found.")

    numeric_data = numeric_data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(axis=1, how="all")

    numeric_data = numeric_data.fillna(
        numeric_data.median()
    )

    numeric_data = numeric_data.loc[
        :, numeric_data.nunique() > 1
    ]

    feature_columns = numeric_data.columns.tolist()

    if len(feature_columns) == 0:
        raise ValueError("No usable numerical features found.")

    if len(data) < 6:
        raise ValueError("At least 6 records are required.")

    # Distance-based algorithms benefit from scaling
    if algorithm in ["LOF", "KNN"]:
        model_data = StandardScaler().fit_transform(numeric_data)
    else:
        model_data = numeric_data

    # Select model
    if algorithm == "Isolation Forest":
        model = IForest(
            contamination=contamination,
            random_state=42
        )

    elif algorithm == "LOF":
        model = LOF(
            contamination=contamination,
            n_neighbors=min(20, len(data) - 1)
        )

    elif algorithm == "KNN":
        model = KNN(
            contamination=contamination,
            n_neighbors=min(5, len(data) - 1)
        )

    else:
        raise ValueError("Unsupported anomaly detection algorithm.")

    model.fit(model_data)

    predictions = model.predict(model_data)
    scores = model.decision_scores_

    data["Anomaly Status"] = np.where(
        predictions == 1, "Anomaly", "Normal"
    )

    data["Anomaly Score"] = scores
    data["Original Row"] = np.arange(1, len(data) + 1)
    data["Detection Algorithm"] = algorithm

    return data, feature_columns
