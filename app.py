
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
from modules.domain_classifier import classify_dataset
from modules.field_classifier import classify_fields

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
        "File Segregation & Correction",
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
    index=0,
    key="datasentinel_main_navigation_v2",
)

st.sidebar.divider()
st.sidebar.caption("Version 1.0 | DataSentinel")

# --------------------------------
# DASHBOARD
# --------------------------------

if page == "Dashboard":
    st.header("Dataset Health Overview")
    st.info(
        "This dashboard analyses CSV and Excel datasets. For JPEG/JPG images, "
        "PDFs, Word files, mixed batches, or other formats, use "
        "**File Segregation & Correction** in the sidebar."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel dataset (Dashboard only)",
        type=["csv", "xlsx", "xls"],
        help="For images and mixed file types, use File Segregation & Correction.",
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

elif page == "File Segregation & Correction":
    from pathlib import Path

    st.header("AI File Segregation & Correction")
    st.caption(
        "Upload mixed files in one batch. DataSentinel identifies their formats, "
        "groups them by likely domain, checks tabular data, and creates a "
        "downloadable folder structure with original and corrected copies."
    )

    st.info(
        "You can select files with any extension. Common images, spreadsheets, "
        "PDFs, Word/PowerPoint files, text, JSON, TSV and Parquet are recognised. "
        "Unknown or unsupported binaries are kept in **Needs Review** rather than guessed."
    )

    repair_col1, repair_col2 = st.columns(2)
    with repair_col1:
        remove_duplicates = st.checkbox(
            "Remove exact duplicate rows in corrected copies",
            value=False,
            key="seg_remove_duplicates",
            help="Off by default because identical rows can sometimes be legitimate.",
        )
    with repair_col2:
        fill_missing = st.checkbox(
            "Fill missing cells using median / most common value",
            value=False,
            key="seg_fill_missing",
            help="Off by default. This fills values statistically and may not be appropriate for every dataset.",
        )

    st.subheader("Advanced Visual Intelligence")
    vision_col1, vision_col2 = st.columns(2)
    with vision_col1:
        generate_image_descriptions = st.checkbox(
            "Generate an AI description for every image (optional, slower)",
            value=False,
            key="seg_generate_image_descriptions_v2",
            help="Uses an additional image-captioning model. Enable it when you need natural-language descriptions; its model downloads on first use.",
        )
    with vision_col2:
        detect_multiple_objects = st.checkbox(
            "Detect multiple visible objects (optional, slower)",
            value=False,
            key="seg_detect_multiple_objects_v2",
            help="Uses an additional object-detection model to find several common objects in one image. Enable it for crowded scenes; first use downloads the model.",
        )
    st.caption(
        "Image analysis includes broad category and subtype suggestions, visible expression cues, optional object detection, "
        "and visual similarity grouping. It does not guess a person's age, gender identity, or nationality from appearance."
    )

    uploaded_files = st.file_uploader(
        "Upload multiple files (mixed formats are allowed)",
        accept_multiple_files=True,
        key="file_segregation_upload",
        help="Select CSV, Excel, PDF, DOCX, PPTX, images, JSON, text files and other formats together.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected. Batch limit: 100 files and 2 GiB total.")

        if st.button(
            "Analyse, Classify & Check Files",
            type="primary",
            use_container_width=True,
            key="run_file_segregation",
        ):
            try:
                from modules.batch_processor import process_uploaded_files
                from modules.file_segregation import (
                    UNKNOWN_DOMAIN,
                    MIXED_DOMAIN,
                    classify_domain_text,
                    dataframe_to_text,
                    detect_uploaded_file,
                    load_tabular_file,
                    read_text_file,
                    repair_dataframe,
                )

                analysis_records = []
                batch_result = process_uploaded_files(uploaded_files)
                max_full_analysis_bytes = 250 * 1024 * 1024

                with st.spinner("Detecting file types and analysing content..."):
                    for item in batch_result.get("results", []):
                        filename = item.get("filename", "Unnamed file")
                        record = {
                            "filename": filename,
                            "status": item.get("status", "error"),
                            "message": item.get("message", ""),
                            "file_bytes": item.get("file_bytes"),
                            "format": "unknown",
                            "category": "unknown",
                            "domain": UNKNOWN_DOMAIN,
                            "subcategory": "Needs manual review",
                            "document_type": "Unknown",
                            "confidence": 0.0,
                            "evidence": [],
                            "sheets": [],
                            "extracted_text": "",
                            "semantic_text": "",
                            "image_analysis": {},
                            "visual_embedding": None,
                            "similarity_group": None,
                            "visual_similarity_group": None,
                            "notes": [],
                        }

                        if item.get("status") != "ready":
                            analysis_records.append(record)
                            continue

                        file_bytes = item.get("file_bytes") or b""
                        file_size = int(item.get("size_bytes", len(file_bytes)))

                        try:
                            detection = detect_uploaded_file(filename, file_bytes)
                            file_format = detection.get("format", "unknown")
                            file_category = detection.get("category", "unknown")
                            record["format"] = file_format
                            record["category"] = file_category
                            record["message"] = detection.get("message", "")

                            # Advanced image understanding: open-vocabulary CLIP classification,
                            # optional captioning / object detection, and visual embeddings.
                            if file_category == "image":
                                from modules.image_ai.advanced_image_analyzer import analyze_image_bytes

                                image_result = analyze_image_bytes(
                                    file_bytes,
                                    filename=filename,
                                    generate_caption=generate_image_descriptions,
                                    detect_objects=detect_multiple_objects,
                                )
                                if image_result.get("success"):
                                    record["image_analysis"] = image_result
                                    record["domain"] = image_result.get("domain") or UNKNOWN_DOMAIN
                                    record["subcategory"] = image_result.get("subcategory") or "Image"
                                    record["document_type"] = image_result.get("predicted_category") or "Image"
                                    record["confidence"] = image_result.get("relative_score", 0.0)
                                    record["visual_embedding"] = image_result.get("visual_embedding")
                                    record["semantic_text"] = " ".join([
                                        str(image_result.get("caption", "")),
                                        str(image_result.get("visual_group", "")),
                                        str(image_result.get("predicted_category", "")),
                                        " ".join(image_result.get("tags", []) or []),
                                        " ".join(x.get("label", "") for x in image_result.get("detected_objects", []) or []),
                                    ])
                                    record["status"] = "needs_review" if image_result.get("needs_review") else "analysed"
                                    record["notes"].extend(image_result.get("notes", []))
                                    if image_result.get("needs_review"):
                                        record["notes"].append(
                                            "The leading visual match is uncertain. Confirm or change the category before segregation."
                                        )
                                else:
                                    record["notes"].append(
                                        image_result.get("message", "Image model could not identify this image.")
                                    )
                                    record["status"] = "needs_review"
                                record["notes"].append(
                                    "Image content is analysed; the original image pixels are never modified."
                                )

                            # CSV / Excel / JSON / TSV / Parquet.
                            elif file_category == "tabular":
                                if file_size > max_full_analysis_bytes:
                                    if file_format in {"csv", "tsv"}:
                                        sheets = load_tabular_file(
                                            filename, file_bytes, file_format, nrows=20000
                                        )
                                        record["status"] = "sampled"
                                        record["notes"].append(
                                            "File exceeds 250 MiB. The first 20,000 rows were sampled for classification; "
                                            "full-file correction is not performed in this in-memory workflow."
                                        )
                                    else:
                                        record["status"] = "needs_review"
                                        record["notes"].append(
                                            "File exceeds 250 MiB. It was accepted, but full spreadsheet parsing "
                                            "is deferred to a disk-based processing mode to protect laptop memory."
                                        )
                                        analysis_records.append(record)
                                        continue
                                elif file_format in {"xlsx", "xls"}:
                                    from modules.extractors.tabular_extractor import extract_tabular_data

                                    extraction = extract_tabular_data(filename, file_bytes)
                                    if not extraction.get("success"):
                                        raise ValueError(extraction.get("error", "Spreadsheet extraction failed."))
                                    sheets = extraction.get("sheets", [])
                                else:
                                    sheets = load_tabular_file(filename, file_bytes, file_format)

                                for sheet in sheets:
                                    dataframe = sheet.get("dataframe")
                                    if dataframe is None:
                                        continue

                                    sheet_text_for_similarity = dataframe_to_text(dataframe, max_rows=20)
                                    record["semantic_text"] += "\n" + sheet_text_for_similarity
                                    fallback = classify_domain_text(
                                        sheet_text_for_similarity, filename=filename
                                    )
                                    prediction = {}
                                    try:
                                        prediction = classify_dataset(dataframe) or {}
                                    except Exception as model_error:
                                        record["notes"].append(
                                            f"Dataset ML classifier unavailable for this sheet: {model_error}"
                                        )

                                    predicted_domain = prediction.get("domain")
                                    predicted_status = prediction.get("status", "recognised")
                                    if predicted_domain and predicted_status != "needs_review":
                                        sheet_domain = predicted_domain
                                        classification_status = "recognised"
                                        confidence = prediction.get(
                                            "similarity_score", prediction.get("confidence", 0.0)
                                        )
                                    elif fallback.get("domain") != UNKNOWN_DOMAIN:
                                        sheet_domain = fallback["domain"]
                                        classification_status = fallback["status"]
                                        confidence = fallback.get("confidence", 0.0)
                                    else:
                                        sheet_domain = UNKNOWN_DOMAIN
                                        classification_status = "needs_review"
                                        confidence = fallback.get("confidence", 0.0)

                                    try:
                                        field_results = classify_fields(dataframe) or {}
                                    except Exception:
                                        field_results = {"fields": []}

                                    repair = repair_dataframe(
                                        dataframe,
                                        remove_duplicates=remove_duplicates,
                                        fill_missing=fill_missing,
                                    )
                                    record["sheets"].append({
                                        "sheet_name": sheet.get("sheet_name", "Data"),
                                        "row_count": int(sheet.get("row_count", len(dataframe))),
                                        "column_count": int(sheet.get("column_count", len(dataframe.columns))),
                                        "domain": sheet_domain,
                                        "subcategory": prediction.get(
                                            "subcategory", fallback.get("subcategory", "Dataset")
                                        ),
                                        "dataset_type": prediction.get("dataset_type", "Tabular dataset"),
                                        "classification_status": classification_status,
                                        "confidence": confidence,
                                        "field_rows": [
                                            {
                                                "Field": field.get("column", ""),
                                                "Identified As": str(field.get("field_type", "unknown")).replace("_", " ").title(),
                                            }
                                            for field in field_results.get("fields", [])
                                        ],
                                        "cleaned_dataframe": repair["dataframe"],
                                        "repair_summary": repair["summary"],
                                        "suggestions": repair["suggestions"],
                                        "sampled": record["status"] == "sampled",
                                    })

                                if not record["sheets"]:
                                    raise ValueError("No readable table or worksheet was found.")

                                sheet_domains = list(dict.fromkeys(
                                    sheet["domain"] for sheet in record["sheets"]
                                ))
                                recognised_domains = [
                                    domain for domain in sheet_domains if domain != UNKNOWN_DOMAIN
                                ]
                                if len(recognised_domains) > 1:
                                    record["domain"] = MIXED_DOMAIN
                                elif len(recognised_domains) == 1:
                                    record["domain"] = recognised_domains[0]
                                else:
                                    record["domain"] = UNKNOWN_DOMAIN
                                record["subcategory"] = "Spreadsheet / structured dataset"
                                record["document_type"] = "Tabular Dataset"
                                if record["status"] == "ready":
                                    record["status"] = "analysed"

                            # PDF / DOCX / PPTX. TXT and other text types use the text reader.
                            elif file_category == "document":
                                if file_format in {"pdf", "docx", "pptx"}:
                                    from modules.extractors.document_extractor import extract_document_data
                                    extraction = extract_document_data(filename, file_bytes)
                                    if not extraction.get("success"):
                                        raise ValueError(extraction.get("error", "Document extraction failed."))
                                    content = extraction.get("content", {}) or {}
                                    text_content = content.get("full_text", "") or ""
                                else:
                                    text_content = read_text_file(file_bytes, file_format)

                                record["extracted_text"] = text_content
                                record["semantic_text"] = text_content[:6000]
                                classification = classify_domain_text(text_content, filename=filename)
                                record.update({
                                    "domain": classification["domain"],
                                    "subcategory": classification["subcategory"],
                                    "document_type": classification["document_type"],
                                    "confidence": classification["confidence"],
                                    "evidence": classification["evidence"],
                                    "status": classification["status"],
                                })
                                if not text_content.strip():
                                    record["notes"].append(
                                        "No readable text was extracted. Scanned PDFs and image-only documents need OCR."
                                    )
                                    record["domain"] = UNKNOWN_DOMAIN
                                    record["status"] = "needs_review"

                            elif file_category == "text":
                                text_content = read_text_file(file_bytes, file_format)
                                record["extracted_text"] = text_content
                                record["semantic_text"] = text_content[:6000]
                                classification = classify_domain_text(text_content, filename=filename)
                                record.update({
                                    "domain": classification["domain"],
                                    "subcategory": classification["subcategory"],
                                    "document_type": classification["document_type"],
                                    "confidence": classification["confidence"],
                                    "evidence": classification["evidence"],
                                    "status": classification["status"],
                                })

                            elif file_category == "archive":
                                record["notes"].append(
                                    "Archives are not unpacked automatically for security. Extract the archive and upload its files."
                                )
                                record["status"] = "needs_review"

                            else:
                                # The original is retained; no unsupported binary is executed or guessed.
                                record["notes"].append(
                                    "Format not currently readable by DataSentinel. It will be placed in Unknown / Needs Review."
                                )
                                record["status"] = "needs_review"

                        except Exception as file_error:
                            record["domain"] = UNKNOWN_DOMAIN
                            record["status"] = "error"
                            record["notes"].append(str(file_error))

                        analysis_records.append(record)

                # Group files by shared content semantics and images by visual embeddings.
                try:
                    from modules.semantic_grouping import assign_similarity_groups
                    similarity_summary = assign_similarity_groups(analysis_records)
                except Exception as similarity_error:
                    similarity_summary = {"warning": str(similarity_error)}
                    for result_record in analysis_records:
                        result_record.setdefault("similarity_group", None)
                        result_record.setdefault("visual_similarity_group", None)

                st.session_state["file_segregation_similarity_summary"] = similarity_summary
                st.session_state["file_segregation_records"] = analysis_records
                st.session_state["file_segregation_batch"] = {
                    "total_files": batch_result.get("total_files", len(analysis_records)),
                    "ready_files": batch_result.get("ready_files", 0),
                    "rejected_files": batch_result.get("rejected_files", 0),
                    "total_size_bytes": batch_result.get("total_size_bytes", 0),
                }
                old_zip = st.session_state.pop("file_segregation_zip_path", None)
                if old_zip and Path(old_zip).exists():
                    try:
                        Path(old_zip).unlink()
                    except OSError:
                        pass
                st.success("Batch analysis finished. Review the suggested categories and corrections below.")

            except Exception as e:
                st.error(f"File segregation failed: {e}")
                with st.expander("Technical details"):
                    st.exception(e)

    records = st.session_state.get("file_segregation_records", [])
    if records:
        st.divider()
        st.subheader("Segregation Results")

        summary_rows = []
        for record in records:
            summary_rows.append({
                "File": record.get("filename", ""),
                "Detected Format": str(record.get("format", "unknown")).upper(),
                "Suggested Category": record.get("domain", "Unknown / Needs Review"),
                "Content Group": record.get("similarity_group") or "—",
                "Visual Group": record.get("visual_similarity_group") or "—",
                "Status": record.get("status", "unknown").replace("_", " ").title(),
            })
        summary_df = pd.DataFrame(summary_rows)
        group_counts = (
            summary_df.groupby("Suggested Category", dropna=False)
            .size()
            .reset_index(name="Files")
            .sort_values("Files", ascending=False)
        )
        st.markdown("**Files grouped by suggested domain**")
        st.dataframe(group_counts, use_container_width=True, hide_index=True)
        st.markdown("**Individual file results**")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        similarity_summary = st.session_state.get("file_segregation_similarity_summary", {})
        if similarity_summary:
            st.caption(
                "Similarity grouping: "
                f"{similarity_summary.get('text_groups', 0)} content group(s), "
                f"{similarity_summary.get('visual_groups', 0)} visual group(s). "
                "Groups are based on extracted text/captions and image embeddings; review them before relying on them."
            )

        st.markdown("**Confirm or change each file's category before creating the segregated ZIP.**")
        from modules.file_segregation import (
            DOMAIN_OPTIONS,
            UNKNOWN_DOMAIN,
            MIXED_DOMAIN,
            create_segregation_archive,
            dataframe_to_csv_bytes,
        )

        for file_index, record in enumerate(records):
            with st.container(border=True):
                st.markdown(f"### {record.get('filename', 'Unnamed file')}")
                st.caption(
                    f"Format: {str(record.get('format', 'unknown')).upper()} · "
                    f"Type: {record.get('category', 'unknown')} · "
                    f"Status: {record.get('status', 'unknown')}"
                )

                current_domain = record.get("domain", UNKNOWN_DOMAIN)
                domain_options = list(dict.fromkeys([current_domain] + DOMAIN_OPTIONS))
                selected_domain = st.selectbox(
                    "Confirm file category",
                    options=domain_options,
                    index=domain_options.index(current_domain),
                    key=f"seg_file_domain_{file_index}",
                )
                record["domain"] = selected_domain

                if record.get("subcategory"):
                    st.write(f"**Detected subtype:** {record['subcategory']}")
                if record.get("document_type"):
                    st.write(f"**File / document type:** {record['document_type']}")
                if record.get("confidence") not in (None, 0, 0.0, ""):
                    st.caption(
                        f"Model / keyword match indicator: {record['confidence']}. "
                        "This is a ranking aid, not a guaranteed probability."
                    )
                if record.get("evidence"):
                    st.caption("Evidence found: " + ", ".join(record["evidence"]))

                for note in record.get("notes", []):
                    st.info(note)

                for sheet_index, sheet in enumerate(record.get("sheets", [])):
                    st.divider()
                    st.markdown(
                        f"**Sheet: {sheet.get('sheet_name', 'Data')}** — "
                        f"{sheet.get('row_count', 0):,} rows × {sheet.get('column_count', 0)} columns"
                    )
                    sheet_domain = sheet.get("domain", UNKNOWN_DOMAIN)
                    sheet_options = list(dict.fromkeys([sheet_domain] + DOMAIN_OPTIONS))
                    sheet["domain"] = st.selectbox(
                        f"Confirm category for sheet: {sheet.get('sheet_name', 'Data')}",
                        options=sheet_options,
                        index=sheet_options.index(sheet_domain),
                        key=f"seg_sheet_domain_{file_index}_{sheet_index}",
                    )

                    repair_summary = sheet.get("repair_summary", {})
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Missing cells", f"{repair_summary.get('missing_before', 0):,}")
                    metric_cols[1].metric("Duplicate rows", f"{repair_summary.get('duplicates_before', 0):,}")
                    metric_cols[2].metric("Whitespace trimmed", f"{repair_summary.get('whitespace_cells_trimmed', 0):,}")
                    metric_cols[3].metric("Rows after repair", f"{repair_summary.get('rows_after', 0):,}")

                    if sheet.get("field_rows"):
                        with st.expander("Detected fields"):
                            st.dataframe(
                                pd.DataFrame(sheet["field_rows"]),
                                use_container_width=True,
                                hide_index=True,
                            )
                    with st.expander("Preview corrected data"):
                        st.dataframe(
                            sheet["cleaned_dataframe"].head(15),
                            use_container_width=True,
                        )
                    for suggestion in sheet.get("suggestions", []):
                        st.caption("• " + suggestion)

                image_analysis = record.get("image_analysis") or {}
                if record.get("category") == "image" and image_analysis:
                    st.divider()
                    st.markdown("**Visual understanding**")
                    preview_col, details_col = st.columns([1, 2])
                    with preview_col:
                        st.image(record.get("file_bytes"), use_container_width=True)
                    with details_col:
                        if image_analysis.get("caption"):
                            st.write("**AI description:** " + image_analysis["caption"])
                        st.write("**Visual family:** " + str(image_analysis.get("visual_group", "Unclear")))
                        st.write("**Best matching label:** " + str(image_analysis.get("predicted_category", "Unclear")))
                        if image_analysis.get("imagenet_top_label"):
                            imagenet_score = float(image_analysis.get("imagenet_top_score", 0.0))
                            st.write(
                                "**Everyday-object model cross-check:** "
                                + str(image_analysis.get("imagenet_top_label"))
                                + f" (ranking score: {imagenet_score:.1%})"
                            )
                        if image_analysis.get("visible_expression"):
                            st.write(
                                "**Visible expression cue:** "
                                + str(image_analysis["visible_expression"].get("visible_cue", "not clear"))
                            )
                        if image_analysis.get("detected_objects"):
                            object_labels = [
                                f"{obj.get('label')} ({obj.get('score', 0):.2f})"
                                for obj in image_analysis["detected_objects"]
                            ]
                            st.write("**Detected objects:** " + ", ".join(object_labels))
                        if image_analysis.get("top_predictions"):
                            with st.expander("Alternative visual matches"):
                                st.dataframe(
                                    pd.DataFrame(image_analysis["top_predictions"]),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                        if record.get("visual_similarity_group"):
                            st.caption(f"Visual similarity group: {record['visual_similarity_group']}")
                        if record.get("similarity_group"):
                            st.caption(f"Cross-file content similarity group: {record['similarity_group']}")

                    from modules.image_ai.advanced_image_analyzer import image_training_labels, save_image_training_example
                    image_labels = image_training_labels()
                    current_label = image_analysis.get("predicted_category", "Other / Unclear")
                    label_index = image_labels.index(current_label) if current_label in image_labels else 0
                    selected_training_label = st.selectbox(
                        "Correct / confirm the visual label (optional training feedback)",
                        options=image_labels,
                        index=label_index,
                        key=f"image_training_label_{file_index}",
                    )
                    if st.button(
                        "Save this confirmed label as a training example",
                        key=f"save_image_training_{file_index}",
                    ):
                        try:
                            saved_example = save_image_training_example(
                                record.get("file_bytes", b""),
                                record.get("filename", "image"),
                                selected_training_label,
                            )
                            st.success(f"Saved training example: {saved_example}")
                        except Exception as feedback_error:
                            st.error(f"Could not save training example: {feedback_error}")

                if record.get("extracted_text"):
                    with st.expander("Preview extracted document text"):
                        st.text(record["extracted_text"][:4000])

        st.divider()
        st.subheader("Improve the Image Model with Your Verified Labels")
        st.write(
            "Each saved correction becomes a labelled image example. After collecting varied examples "
            "for multiple categories, DataSentinel can train a project-specific classifier on top of CLIP embeddings."
        )
        try:
            from modules.image_ai.image_training import count_image_training_examples, train_custom_image_classifier
            training_counts = count_image_training_examples()
            train_col1, train_col2 = st.columns(2)
            train_col1.metric("Verified image examples", training_counts.get("total", 0))
            train_col2.metric("Labelled categories", training_counts.get("classes", 0))
            st.caption(
                "For a useful first model, collect at least 5 examples in each of at least 2 categories; "
                "for stronger generalisation, aim for 50–200 varied examples per category and keep a separate real-world test set."
            )
            if st.button(
                "Train / Refresh Custom Image Classifier",
                type="secondary",
                use_container_width=True,
                key="train_custom_image_classifier",
            ):
                with st.spinner("Encoding labelled images and evaluating the custom classifier..."):
                    training_result = train_custom_image_classifier()
                st.success(
                    f"Custom image classifier trained. Held-out accuracy: "
                    f"{training_result['accuracy']:.1%} on {training_result['test_samples']} test images."
                )
                st.caption(
                    f"Categories: {', '.join(training_result['classes'])}. Model saved to {training_result['model_path']}. "
                    + training_result["note"]
                )
        except Exception as training_panel_error:
            st.warning(f"Image training tools are not available yet: {training_panel_error}")

        st.divider()
        st.subheader("Create Segregated & Corrected Download")
        st.write(
            "The ZIP preserves every original file in its assigned category. "
            "Corrected table copies are placed under **Cleaned Data** by sheet category; "
            "Excel files also receive a rebuilt cleaned workbook. Extracted document text "
            "is stored separately. The original uploads are never overwritten."
        )

        if st.button(
            "Build Segregated ZIP",
            type="primary",
            use_container_width=True,
            key="build_segregated_zip",
        ):
            try:
                with st.spinner("Building category folders and corrected copies..."):
                    archive_path = create_segregation_archive(records)
                    st.session_state["file_segregation_zip_path"] = archive_path
                st.success("Segregated ZIP is ready.")
            except Exception as e:
                st.error(f"Could not build the segregated ZIP: {e}")
                with st.expander("Technical details"):
                    st.exception(e)

        archive_path = st.session_state.get("file_segregation_zip_path")
        if archive_path and Path(archive_path).is_file():
            with open(archive_path, "rb") as archive_file:
                st.download_button(
                    "Download Segregated & Corrected Files",
                    data=archive_file,
                    file_name="DataSentinel_Segregated_and_Corrected.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="download_segregated_zip",
                )


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
