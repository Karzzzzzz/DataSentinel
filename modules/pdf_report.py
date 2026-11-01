
from io import BytesIO
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from xml.sax.saxutils import escape


def generate_pdf_report(
    df,
    dataset_name,
    quality_score=None,
    quality_results=None,
    validation_results=None,
    anomaly_results=None,
    model_summary=None
):
    """Generate a downloadable DataSentinel PDF report."""

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        textColor=colors.HexColor("#183153"),
        alignment=TA_CENTER,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#183153"),
        spaceBefore=14,
        spaceAfter=8
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        wordWrap="CJK"
    )

    story = []

    story.append(Paragraph("DATASENTINEL", title_style))
    story.append(
        Paragraph(
            "AI-Powered Data Quality Monitoring and Anomaly Detection",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Dataset: {escape(str(dataset_name))}",
            styles["Heading3"]
        )
    )

    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 12))

    # DATASET OVERVIEW

    story.append(
        Paragraph("1. Dataset Overview", heading_style)
    )

    overview = [
        ["Metric", "Value"],
        ["Total Records", str(len(df))],
        ["Total Columns", str(len(df.columns))],
        ["Missing Values", str(int(df.isna().sum().sum()))],
        ["Duplicate Rows", str(int(df.duplicated().sum()))]
    ]

    if quality_results is not None:
        overview.extend([
            ["Invalid Values", str(quality_results.get("invalid_count", 0))]
        ])

    table = Table(overview, colWidths=[75 * mm, 75 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F6FA"))
    ]))
    story.append(table)

    # QUALITY SCORE

    if quality_score is not None:

        story.append(
            Paragraph("2. Data Quality Score", heading_style)
        )

        score_data = [
            ["Quality Score", "Dataset Status"],
            [
                f"{quality_score.get('score', 0)}/100",
                str(quality_score.get("status", "Not calculated"))
            ]
        ]

        table = Table(score_data, colWidths=[75 * mm, 75 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ]))
        story.append(table)

        recommendations = quality_score.get("recommendations", [])

        if recommendations:
            story.append(Spacer(1, 8))
            story.append(
                Paragraph("Cleaning Recommendations", styles["Heading3"])
            )

            for recommendation in recommendations:
                story.append(
                    Paragraph(
                        "• " + escape(str(recommendation)),
                        styles["Normal"]
                    )
                )

    # VALIDATION RESULTS

    if validation_results is not None:

        story.append(
            Paragraph("3. Data Validation Results", heading_style)
        )

        validation_data = [[
            "Validation Check",
            "Column",
            "Status",
            "Unexpected Count"
        ]]

        for _, row in validation_results.head(25).iterrows():
            validation_data.append([
                Paragraph(escape(str(row["Validation Check"])), cell_style),
                Paragraph(escape(str(row["Column"])), cell_style),
                str(row["Status"]),
                str(row["Unexpected Count"])
            ])

        table = Table(
            validation_data,
            colWidths=[65 * mm, 35 * mm, 25 * mm, 35 * mm],
            repeatRows=1
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))

        story.append(table)

    # ANOMALY RESULTS

    if anomaly_results is not None:

        story.append(
            Paragraph("4. Anomaly Detection Results", heading_style)
        )

        anomaly_count = int(
            (anomaly_results["Anomaly Status"] == "Anomaly").sum()
        )

        anomaly_data = [
            ["Metric", "Value"],
            ["Total Records", str(len(anomaly_results))],
            ["Anomalies Detected", str(anomaly_count)],
            ["Normal Records", str(len(anomaly_results) - anomaly_count)]
        ]

        table = Table(anomaly_data, colWidths=[75 * mm, 75 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ]))
        story.append(table)

    # MODEL COMPARISON

    if model_summary is not None:

        story.append(
            Paragraph("5. ML Algorithm Comparison", heading_style)
        )

        comparison_data = [[
            "Algorithm",
            "Anomalies",
            "Normal Records",
            "Anomaly %"
        ]]

        for _, row in model_summary.iterrows():
            comparison_data.append([
                str(row["Algorithm"]),
                str(row["Anomalies Detected"]),
                str(row["Normal Records"]),
                str(row["Anomaly Percentage"])
            ])

        table = Table(
            comparison_data,
            colWidths=[55 * mm, 30 * mm, 35 * mm, 35 * mm],
            repeatRows=1
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ]))

        story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by DataSentinel | AI-Powered Data Quality Platform",
            subtitle_style
        )
    )

    document.build(story)

    return buffer.getvalue()
