
import pandas as pd

from modules.anomaly_detector import detect_anomalies


def compare_algorithms(df, contamination=0.05):
    """
    Run Isolation Forest, LOF and KNN on the same dataset.
    Compare anomaly counts and record-level agreement.
    """

    algorithms = [
        "Isolation Forest",
        "LOF",
        "KNN"
    ]

    results = {}
    summary = []

    # Run all three algorithms
    for algorithm in algorithms:

        detected, features = detect_anomalies(
            df,
            contamination=contamination,
            algorithm=algorithm
        )

        results[algorithm] = detected

        anomaly_count = int(
            (detected["Anomaly Status"] == "Anomaly").sum()
        )

        normal_count = len(detected) - anomaly_count

        summary.append({
            "Algorithm": algorithm,
            "Total Records": len(detected),
            "Anomalies Detected": anomaly_count,
            "Normal Records": normal_count,
            "Anomaly Percentage": round(
                anomaly_count / len(detected) * 100, 2
            )
        })

    summary_df = pd.DataFrame(summary)

    # Compare which records each algorithm identifies
    comparison = df.copy()

    for algorithm in algorithms:

        detected = results[algorithm]

        comparison[f"{algorithm} Result"] = (
            detected["Anomaly Status"].values
        )

    result_columns = [
        f"{algorithm} Result"
        for algorithm in algorithms
    ]

    # Count how many algorithms flag each record
    comparison["Anomaly Votes"] = (
        comparison[result_columns] == "Anomaly"
    ).sum(axis=1)

    # Two or more algorithms must agree
    comparison["Consensus Status"] = comparison[
        "Anomaly Votes"
    ].apply(
        lambda votes: (
            "Consensus Anomaly"
            if votes >= 2 else "Normal / Disagreement"
        )
    )

    comparison["Original Row"] = range(
        1, len(comparison) + 1
    )

    return summary_df, comparison, results
