
import streamlit as st
import pandas as pd
import plotly.express as px

from modules.quality_checker import analyze_quality
from modules.anomaly_detector import detect_anomalies
from modules.profiler import generate_profile
from modules.validation_engine import validate_dataset
from modules.drift_detector import generate_drift_report
from modules.observability import (
    generate_observability_profile,
    save_observability_snapshot,
    load_observability_history,
    compare_observability_profiles,
)
from modules.cleaning_engine import clean_dataset
from modules.quality_score import calculate_quality_score
from modules.model_comparison import compare_algorithms
from modules.pdf_report import generate_pdf_report

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="DataSentinel",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------
# DATASENTINEL PROFESSIONAL UI
# --------------------------------

st.markdown("""
<style>

/* Main application background */

.stApp {
    background: linear-gradient(
        135deg,
        #0b1120 0%,
        #111827 55%,
        #172033 100%
    );
    color: #e5e7eb;
}

/* Main content spacing */

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar styling */

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] h1 {
    color: #60a5fa;
    font-weight: 800;
}

/* Headings */

h1 {
    color: #f8fafc !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background: #182438;
    border: 1px solid #293a52;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: #60a5fa;
    font-weight: 800;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #4f46e5
    );
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 18px;
    font-weight: 650;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #4338ca
    );
    border: none;
    color: white;
    transform: translateY(-1px);
}

/* Dataframes */

div[data-testid="stDataFrame"] {
    border: 1px solid #293a52;
    border-radius: 12px;
    overflow: hidden;
}

/* File uploader */

section[data-testid="stFileUploader"] {
    background: #182438;
    border: 1px dashed #3b82f6;
    border-radius: 14px;
    padding: 18px;
}

/* Input controls */

.stSelectbox > div > div,
.stNumberInput input,
.stTextInput input {
    background-color: #182438;
    border-radius: 9px;
}

/* Information messages */

div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */

hr {
    border-color: #293a52;
}

/* Captions */

.stCaption {
    color: #94a3b8;
}

/* Progress bar */

div[data-testid="stProgressBar"] > div > div {
    background-color: #3b82f6;
}

/* Sidebar radio navigation */

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 6px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1e293b;
}

/* Smooth appearance */

* {
    transition: background-color 0.2s ease;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("🛡️ DataSentinel")
st.caption(
    "AI-Powered Data Quality Monitoring and Anomaly Detection Platform"
)

st.divider()

# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("DataSentinel")
st.sidebar.caption("Dataset Intelligence Platform")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Data Explorer",
        "Data Validation",
        "Anomaly Detection",
        "Data Drift",
        "Data Observability",
        "Cleaning Studio",
        "Model Comparison",
        "Reports",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Version 1.0 | DataSentinel")

# --------------------------------
# DASHBOARD
# --------------------------------

if page == "Dashboard":
    st.header("Dataset Health Overview")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.session_state["dataset"] = df.copy()
            st.session_state["dataset_name"] = uploaded_file.name

            total_rows = len(df)
            total_columns = len(df.columns)
            missing_values = int(df.isnull().sum().sum())
            duplicates = int(df.duplicated().sum())

            st.success("Dataset uploaded successfully!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", total_rows)
            col2.metric("Total Columns", total_columns)
            col3.metric("Missing Values", missing_values)
            col4.metric("Duplicate Rows", duplicates)

            quality_results = analyze_quality(df)
            quality_score = calculate_quality_score(df)

            st.divider()
            st.subheader("Overall Data Quality Score")

            score_col1, score_col2 = st.columns(2)
            score_col1.metric("Data Quality Score", f"{quality_score['score']}/100")
            score_col2.metric("Dataset Status", quality_score["status"])

            st.progress(quality_score["score"] / 100)

            st.subheader("Quality Penalty Breakdown")
            penalty_data = pd.DataFrame(
                {
                    "Category": ["Missing Values", "Duplicates", "Invalid Values"],
                    "Penalty": [
                        quality_score["missing_penalty"],
                        quality_score["duplicate_penalty"],
                        quality_score["invalid_penalty"],
                    ],
                }
            )
            st.bar_chart(penalty_data.set_index("Category"))

            st.subheader("Cleaning Recommendations")
            for recommendation in quality_score["recommendations"]:
                st.write("•", recommendation)

            st.divider()
            st.subheader("Data Quality Detection")

            q1, q2, q3 = st.columns(3)
            q1.metric("Missing Values Detected", quality_results["missing_count"])
            q2.metric("Duplicate Records", quality_results["duplicate_count"])
            q3.metric("Invalid Values", quality_results["invalid_count"])

            st.subheader("Invalid Records")
            if not quality_results["invalid_records"].empty:
                st.dataframe(quality_results["invalid_records"], use_container_width=True)
            else:
                st.success("No invalid values detected!")

            st.divider()
            st.subheader("Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)

            st.subheader("Missing Value Analysis")
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ["Column", "Missing Values"]

            fig = px.bar(
                missing_data,
                x="Column",
                y="Missing Values",
                title="Missing Values by Column",
                color="Missing Values",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Column Information")
            column_info = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str).values,
                    "Missing Values": df.isnull().sum().values,
                    "Unique Values": df.nunique().values,
                }
            )
            st.dataframe(column_info, use_container_width=True)

        except Exception as e:
            st.error(f"Unable to process dataset: {e}")
    else:
        st.info("Upload a CSV dataset to begin your analysis.")
        st.markdown(
            """
            ### Welcome to DataSentinel

            Your intelligent data quality monitoring platform.

            **What DataSentinel will help you discover:**

            - Missing values
            - Duplicate records
            - Invalid data
            - Statistical outliers
            - Machine learning anomalies
            - Data drift
            """
        )

# --------------------------------
# OTHER MODULES
# --------------------------------

elif page == "Data Explorer":
    st.header("Advanced Data Explorer")
    st.caption("Detailed statistical profiling powered by YData")

    dataset = st.session_state.get("dataset")

    if dataset is None:
        st.warning("Please upload a dataset in the Dashboard first.")
    else:
        st.write(
            f"Currently analysing: **{st.session_state.get('dataset_name', 'Dataset')}**"
        )

        st.divider()
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(dataset))
        col2.metric("Total Columns", len(dataset.columns))
        col3.metric("Missing Values", int(dataset.isnull().sum().sum()))

        st.divider()

        if st.button(
            "Generate Detailed Dataset Profile",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Generating detailed statistical report..."):
                    profile = generate_profile(dataset)
                    profile_html = profile.to_html()
                    st.session_state["profile_html"] = profile_html

                st.success("Dataset profiling completed successfully!")
            except Exception as e:
                st.error(f"Profiling failed: {e}")

        if "profile_html" in st.session_state:
            st.subheader("Detailed Dataset Analysis")
            st.components.v1.html(
                st.session_state["profile_html"],
                height=1200,
                scrolling=True,
            )

            st.download_button(
                label="Download Complete Profiling Report",
                data=st.session_state["profile_html"],
                file_name="DataSentinel_Profiling_Report.html",
                mime="text/html",
            )

elif page == "Data Validation":
    st.header("Data Validation Engine")
    st.caption("Powered by Great Expectations")

    dataset = st.session_state.get("dataset")

    if dataset is None:
        st.warning("Please upload a dataset in the Dashboard first.")
    else:
        st.write(
            f"Validating dataset: **{st.session_state.get('dataset_name', 'Dataset')}**"
        )

        st.divider()
        st.subheader("Validation Rules")
        st.markdown(
            """
            DataSentinel will check:

            - Missing values
            - Age between 1 and 100
            - Marks between 0 and 100
            - Attendance between 0 and 100
            - Unique student IDs
            """
        )

        if st.button(
            "Run Data Validation",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Running validation checks..."):
                    validation_results = validate_dataset(dataset)
                    st.session_state["validation_results"] = validation_results

                st.success("Data validation completed!")
            except Exception as e:
                st.error(f"Validation failed: {e}")

        results = st.session_state.get("validation_results")
        if results is not None:
            st.divider()
            st.subheader("Validation Results")

            passed = int((results["Status"] == "PASS").sum())
            failed = int((results["Status"] == "FAIL").sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Checks", len(results))
            col2.metric("Passed Checks", passed)
            col3.metric("Failed Checks", failed)

            st.subheader("Detailed Validation Report")
            st.dataframe(results, use_container_width=True, hide_index=True)

            if failed == 0:
                st.success("All validation checks passed!")
            else:
                st.warning(
                    f"{failed} validation checks failed. Review the unexpected values."
                )

            st.download_button(
                "Download Validation Report",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Validation_Report.csv",
                mime="text/csv",
            )

elif page == "Anomaly Detection":
    st.header("AI-Powered Anomaly Detection")
    st.caption("Powered by PyOD — Isolation Forest")

    dataset = st.session_state.get("dataset")

    if dataset is None:
        st.warning("Please upload a dataset in the Dashboard first.")
    else:
        st.write(
            f"Analysing dataset: **{st.session_state.get('dataset_name', 'Dataset')}**"
        )

        st.divider()
        st.subheader("Model Configuration")

        contamination = st.slider(
            "Expected anomaly proportion",
            min_value=0.01,
            max_value=0.30,
            value=0.05,
            step=0.01,
            help="Estimated proportion of unusual records.",
        )
        algorithm = st.selectbox(
            "Select Anomaly Detection Algorithm",
            ["Isolation Forest", "LOF", "KNN"],
        )

        st.info(
            "Isolation Forest identifies unusual records without requiring labelled training data."
        )

        if st.button(
            "Run Anomaly Detection",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Training anomaly detection model..."):
                    results, features = detect_anomalies(
                        dataset,
                        contamination=contamination,
                        algorithm=algorithm,
                    )
                    st.session_state["anomaly_results"] = results
                    st.session_state["anomaly_features"] = features

                st.success("Anomaly detection completed!")
            except Exception as e:
                st.error(f"Anomaly detection failed: {e}")

        results = st.session_state.get("anomaly_results")

        if results is not None:
            anomalies = results[results["Anomaly Status"] == "Anomaly"]
            normal_records = results[results["Anomaly Status"] == "Normal"]

            st.divider()
            st.subheader("Detection Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", len(results))
            c2.metric("Anomalies Detected", len(anomalies))
            c3.metric("Normal Records", len(normal_records))

            st.subheader("Anomaly Distribution")
            fig = px.pie(
                names=["Normal", "Anomaly"],
                values=[len(normal_records), len(anomalies)],
                title="Normal vs Anomalous Records",
                hole=0.5,
                color_discrete_sequence=["#16A34A", "#EF4444"],
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Anomalous Records")
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)
            else:
                st.success("No anomalies detected.")

            st.subheader("Anomaly Score Distribution")
            score_fig = px.histogram(
                results,
                x="Anomaly Score",
                color="Anomaly Status",
                title="Distribution of Anomaly Scores",
                barmode="overlay",
            )
            st.plotly_chart(score_fig, use_container_width=True)

            st.download_button(
                "Download Anomaly Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="datasentinel_anomaly_report.csv",
                mime="text/csv",
            )

elif page == "Data Drift":
    st.header("Data Drift Monitoring")
    st.caption("Powered by Evidently AI")

    reference_df = st.session_state.get("dataset")

    if reference_df is None:
        st.warning("Upload a reference dataset in Dashboard first.")
    else:
        st.success("Reference dataset loaded successfully.")
        st.write(
            f"Reference dataset: **{st.session_state.get('dataset_name', 'Dataset')}**"
        )
        st.metric("Reference Records", len(reference_df))

        st.divider()
        st.subheader("Upload Current Dataset")
        current_file = st.file_uploader(
            "Upload a second dataset for comparison",
            type=["csv"],
            key="drift_current_upload",
        )

        if current_file is not None:
            try:
                current_df = pd.read_csv(current_file)
                st.metric("Current Records", len(current_df))
                st.write("Current dataset preview")
                st.dataframe(current_df.head(10), use_container_width=True)

                if st.button(
                    "Run Data Drift Analysis",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Analysing dataset drift..."):
                        html = generate_drift_report(reference_df, current_df)
                        st.session_state["drift_html"] = html
                        st.session_state["drift_filename"] = current_file.name

                    st.success("Data drift analysis completed!")
            except Exception as e:
                st.error(f"Drift analysis failed: {e}")

        if "drift_html" in st.session_state:
            st.divider()
            st.subheader("Evidently Data Drift Report")
            st.components.v1.html(
                st.session_state["drift_html"],
                height=1200,
                scrolling=True,
            )

            st.download_button(
                "Download Data Drift Report",
                data=st.session_state["drift_html"],
                file_name="DataSentinel_Drift_Report.html",
                mime="text/html",
            )

elif page == "Data Observability":
    st.header("Data Observability")
    st.caption("Dataset Health Tracking and Historical Monitoring")

    dataset = st.session_state.get("dataset")
    dataset_name = st.session_state.get("dataset_name", "dataset")

    if dataset is None:
        st.warning("Please upload a dataset in Dashboard first.")
    else:
        st.write(f"Monitoring dataset: **{dataset_name}**")

        st.divider()
        st.subheader("Dataset Monitoring")

        if st.button(
            "Generate Observability Snapshot",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Analysing dataset statistics..."):
                    profile = generate_observability_profile(dataset, dataset_name)
                    history = load_observability_history(dataset_name)
                    previous = history[-1] if history else None

                    snapshot_path = save_observability_snapshot(profile)

                    st.session_state["observability_profile"] = profile
                    st.session_state["observability_snapshot_path"] = snapshot_path
                    st.session_state["observability_changes"] = (
                        compare_observability_profiles(previous, profile)
                        if previous is not None
                        else None
                    )
                    st.session_state["observability_history"] = (
                        load_observability_history(dataset_name)
                    )

                st.success("Observability snapshot generated successfully!")
            except Exception as e:
                st.error(f"Observability failed: {e}")

        profile = st.session_state.get("observability_profile")

        if profile is not None:
            st.divider()
            st.subheader("Current Dataset Health")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows", profile["total_rows"])
            c2.metric("Total Columns", profile["total_columns"])
            c3.metric("Missing Cells", profile["missing_cells"])
            c4.metric("Duplicate Rows", profile["duplicate_rows"])

            st.subheader("Column-Level Statistics")
            column_data = pd.DataFrame(profile["columns"])
            st.dataframe(column_data, use_container_width=True, hide_index=True)

            changes = st.session_state.get("observability_changes")

            st.divider()
            st.subheader("Historical Monitoring")
            if changes is not None:
                st.write("Comparison with the previous snapshot")
                st.dataframe(changes, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "This is your first monitoring snapshot. It has been saved as the initial baseline. "
                    "Generate another snapshot after updating the dataset to compare changes."
                )

            history = st.session_state.get("observability_history", [])
            st.subheader("Monitoring History")
            history_data = pd.DataFrame(
                [
                    {
                        "Captured At": item["captured_at"],
                        "Rows": item["total_rows"],
                        "Columns": item["total_columns"],
                        "Missing Cells": item["missing_cells"],
                        "Duplicates": item["duplicate_rows"],
                    }
                    for item in history
                ]
            )

            if not history_data.empty:
                st.dataframe(
                    history_data.sort_values("Captured At", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download Observability Report",
                data=pd.DataFrame(profile["columns"]).to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Observability_Report.csv",
                mime="text/csv",
            )

elif page == "Cleaning Studio":
    st.header("Data Cleaning Studio")
    st.caption("Automatically clean and improve dataset quality")

    dataset = st.session_state.get("dataset")

    if dataset is None:
        st.warning("Please upload a dataset in Dashboard first.")
    else:
        st.write(f"Dataset: **{st.session_state.get('dataset_name', 'Dataset')}**")

        st.divider()
        st.subheader("Cleaning Configuration")

        remove_duplicates = st.checkbox("Remove exact duplicate records", value=True)
        correct_invalid = st.checkbox("Correct invalid numerical values", value=True)

        st.info(
            "Missing numerical values are filled using the median. Missing categorical values are filled using the mode."
        )

        if st.button(
            "Clean My Dataset",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Cleaning dataset..."):
                    cleaned_df, summary = clean_dataset(
                        dataset,
                        remove_duplicates=remove_duplicates,
                        correct_invalid=correct_invalid,
                    )
                    st.session_state["cleaned_dataset"] = cleaned_df
                    st.session_state["cleaning_summary"] = summary

                st.success("Dataset cleaning completed successfully!")
            except Exception as e:
                st.error(f"Cleaning failed: {e}")

        cleaned = st.session_state.get("cleaned_dataset")
        summary = st.session_state.get("cleaning_summary")

        if cleaned is not None and summary is not None:
            st.divider()
            st.subheader("Cleaning Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Rows", summary["original_rows"])
            c2.metric("Cleaned Rows", summary["cleaned_rows"])
            c3.metric("Duplicates Removed", summary["duplicates_removed"])
            c4.metric("Invalid Values Corrected", summary["invalid_values_corrected"])

            c5, c6 = st.columns(2)
            c5.metric("Missing Values Filled", summary["missing_values_filled"])
            c6.metric("Remaining Missing Values", summary["remaining_missing_values"])

            st.divider()
            st.subheader("Before vs After Cleaning")

            left, right = st.columns(2)
            with left:
                st.markdown("**Original Dataset**")
                st.dataframe(dataset.head(15), use_container_width=True)

            with right:
                st.markdown("**Cleaned Dataset**")
                st.dataframe(cleaned.head(15), use_container_width=True)

            st.download_button(
                "Download Cleaned Dataset",
                data=cleaned.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Cleaned_Dataset.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

elif page == "Model Comparison":
    st.header("ML Algorithm Comparison")
    st.caption("Compare Isolation Forest, LOF and KNN on the same dataset.")

    dataset = st.session_state.get("dataset")

    if dataset is None:
        st.warning("Please upload a dataset in Dashboard first.")
    else:
        st.write(f"Dataset: **{st.session_state.get('dataset_name', 'Dataset')}**")

        st.divider()
        contamination = st.slider(
            "Expected anomaly proportion",
            min_value=0.01,
            max_value=0.30,
            value=0.05,
            step=0.01,
            key="comparison_contamination",
        )

        if st.button(
            "Compare All Algorithms",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Running all three algorithms..."):
                    summary, comparison, model_results = compare_algorithms(
                        dataset,
                        contamination=contamination,
                    )
                    st.session_state["model_summary"] = summary
                    st.session_state["model_comparison"] = comparison

                st.success("Model comparison completed!")
            except Exception as e:
                st.error(f"Model comparison failed: {e}")

        summary = st.session_state.get("model_summary")
        comparison = st.session_state.get("model_comparison")

        if summary is not None and comparison is not None:
            st.divider()
            st.subheader("Algorithm Performance")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.subheader("Anomaly Detection Comparison")
            fig = px.bar(
                summary,
                x="Algorithm",
                y="Anomalies Detected",
                color="Algorithm",
                text="Anomalies Detected",
                title="Anomalies Detected by Each Algorithm",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                summary,
                x="Algorithm",
                y="Anomaly Percentage",
                color="Algorithm",
                text="Anomaly Percentage",
                title="Anomaly Detection Rate",
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.divider()
            st.subheader("Consensus Anomaly Detection")

            consensus = comparison[comparison["Anomaly Votes"] >= 2]
            unanimous = comparison[comparison["Anomaly Votes"] == 3]

            col1, col2, col3 = st.columns(3)
            col1.metric("Consensus Anomalies", len(consensus))
            col2.metric("Unanimous Anomalies", len(unanimous))
            col3.metric("Total Records", len(comparison))

            st.caption(
                "Consensus anomalies are records flagged by at least two of the three algorithms."
            )

            if not consensus.empty:
                st.dataframe(consensus, use_container_width=True, hide_index=True)
            else:
                st.info("No consensus anomalies detected.")

            st.divider()
            st.subheader("Download Comparison Results")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Algorithm Summary",
                    data=summary.to_csv(index=False).encode("utf-8"),
                    file_name="DataSentinel_Model_Comparison.csv",
                    mime="text/csv",
                )

            with col2:
                st.download_button(
                    "Download Consensus Results",
                    data=comparison.to_csv(index=False).encode("utf-8"),
                    file_name="DataSentinel_Consensus_Results.csv",
                    mime="text/csv",
                )

elif page == "Reports":
    st.header("DataSentinel Reports Center")
    st.caption("Download and manage your dataset analysis reports.")

    dataset = st.session_state.get("dataset")

    if dataset is None:
        st.warning("Upload a dataset in Dashboard to access reports.")
    else:
        dataset_name = st.session_state.get("dataset_name", "dataset")
        st.write(f"Current dataset: **{dataset_name}**")

        st.divider()
        st.subheader("Dataset Quality Summary")

        quality = calculate_quality_score(dataset)
        quality_results = analyze_quality(dataset)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Quality Score", f"{quality['score']}/100")
        col2.metric("Dataset Status", quality["status"])
        col3.metric("Missing Values", quality_results["missing_count"])
        col4.metric("Duplicate Rows", quality_results["duplicate_count"])

        st.divider()
        st.subheader("Available Reports")

        if "profile_html" in st.session_state:
            st.markdown("**1. Dataset Profiling Report**")
            st.download_button(
                "Download Profiling Report",
                data=st.session_state["profile_html"],
                file_name="DataSentinel_Profiling_Report.html",
                mime="text/html",
            )

        validation = st.session_state.get("validation_results")
        if validation is not None:
            st.markdown("**2. Data Validation Report**")
            st.download_button(
                "Download Validation Report",
                data=validation.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Validation_Report.csv",
                mime="text/csv",
            )

        anomaly = st.session_state.get("anomaly_results")
        if anomaly is not None:
            st.markdown("**3. Anomaly Detection Report**")
            st.download_button(
                "Download Anomaly Report",
                data=anomaly.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Anomaly_Report.csv",
                mime="text/csv",
            )

        if "drift_html" in st.session_state:
            st.markdown("**4. Data Drift Report**")
            st.download_button(
                "Download Drift Report",
                data=st.session_state["drift_html"],
                file_name="DataSentinel_Drift_Report.html",
                mime="text/html",
            )

        profile = st.session_state.get("observability_profile")
        if profile is not None:
            st.markdown("**5. Data Observability Report**")
            observability_data = pd.DataFrame(profile["columns"])
            st.download_button(
                "Download Observability Report",
                data=observability_data.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Observability_Report.csv",
                mime="text/csv",
            )

        cleaned = st.session_state.get("cleaned_dataset")
        if cleaned is not None:
            st.markdown("**6. Cleaned Dataset**")
            st.download_button(
                "Download Cleaned Dataset",
                data=cleaned.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Cleaned_Dataset.csv",
                mime="text/csv",
            )

        summary = st.session_state.get("model_summary")
        comparison = st.session_state.get("model_comparison")

        if summary is not None:
            st.markdown("**7. ML Model Comparison Report**")
            st.download_button(
                "Download Model Comparison",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Model_Comparison.csv",
                mime="text/csv",
            )

        if comparison is not None:
            st.download_button(
                "Download Consensus Analysis",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="DataSentinel_Consensus_Report.csv",
                mime="text/csv",
            )

        quality_report = pd.DataFrame(
            [
                {
                    "Dataset": dataset_name,
                    "Rows": len(dataset),
                    "Columns": len(dataset.columns),
                    "Quality Score": quality["score"],
                    "Status": quality["status"],
                    "Missing Values": quality_results["missing_count"],
                    "Duplicate Rows": quality_results["duplicate_count"],
                    "Invalid Values": quality_results["invalid_count"],
                }
            ]
        )

        st.divider()
        st.download_button(
            "Download Overall Quality Summary",
            data=quality_report.to_csv(index=False).encode("utf-8"),
            file_name="DataSentinel_Quality_Summary.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
        
        # CONSOLIDATED PDF REPORT

        st.divider()
        st.subheader("Complete PDF Report")

        if st.button(
            "Generate Complete PDF Report",
            type="primary",
            use_container_width=True
        ):

            try:

                pdf_data = generate_pdf_report(
                    df=dataset,
                    dataset_name=dataset_name,
                    quality_score=quality,
                    quality_results=quality_results,
                    validation_results=validation,
                    anomaly_results=anomaly,
                    model_summary=summary
                )

                st.session_state["consolidated_pdf"] = pdf_data

                st.success("PDF report generated successfully!")

            except Exception as e:

                st.error(f"PDF generation failed: {e}")

        if "consolidated_pdf" in st.session_state:

            st.download_button(
                "Download Complete DataSentinel PDF",
                data=st.session_state["consolidated_pdf"],
                file_name="DataSentinel_Complete_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )


        st.caption("DataSentinel | AI-Powered Data Quality Monitoring Platform")
