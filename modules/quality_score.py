
from modules.quality_checker import analyze_quality


def calculate_quality_score(df):

    if df is None or df.empty:
        raise ValueError("Cannot score an empty dataset.")

    results = analyze_quality(df)

    rows = len(df)
    total_cells = max(df.size, 1)

    missing_rate = results["missing_count"] / total_cells
    duplicate_rate = results["duplicate_count"] / rows
    invalid_rate = results["invalid_count"] / rows

    # Calculate weighted penalties
    missing_penalty = 40 * min(missing_rate, 1)
    duplicate_penalty = 25 * min(duplicate_rate, 1)
    invalid_penalty = 35 * min(invalid_rate, 1)

    score = round(
        max(
            0,
            100 - missing_penalty
                - duplicate_penalty
                - invalid_penalty
        ),
        2
    )

    # Determine dataset status
    if score >= 90:
        if (
            results["missing_count"] == 0
            and results["duplicate_count"] == 0
            and results["invalid_count"] == 0
        ):
            status = "Excellent"
        else:
            status = "Good - Minor Issues Detected"

    elif score >= 75:
        status = "Needs Cleaning"

    elif score >= 60:
        status = "Needs Improvement"

    else:
        status = "Poor"

    # Generate recommendations
    recommendations = []

    if results["missing_count"] > 0:
        recommendations.append(
            f"Handle {results['missing_count']} missing values."
        )

    if results["duplicate_count"] > 0:
        recommendations.append(
            f"Review {results['duplicate_count']} duplicate records."
        )

    if results["invalid_count"] > 0:
        recommendations.append(
            f"Correct {results['invalid_count']} invalid values."
        )

    if not recommendations:
        recommendations.append(
            "No major data quality issues detected."
        )

    return {
        "score": score,
        "status": status,
        "missing_penalty": round(missing_penalty, 2),
        "duplicate_penalty": round(duplicate_penalty, 2),
        "invalid_penalty": round(invalid_penalty, 2),
        "missing_count": results["missing_count"],
        "duplicate_count": results["duplicate_count"],
        "invalid_count": results["invalid_count"],
        "recommendations": recommendations
    }
