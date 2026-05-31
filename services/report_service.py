# services/report_service.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import cv2
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from config import REPORT_OUTPUT_DIR


class ReportService:
    """
    PDF report generation service.

    Creates a research/demo medical-style report containing:
    - Diagnosis
    - Confidence
    - BI-RADS
    - Reliability
    - Grad-CAM heatmap
    - Notes and disclaimer
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir or REPORT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.styles = getSampleStyleSheet()

        self.title_style = ParagraphStyle(
            "TitleStyle",
            parent=self.styles["Title"],
            fontSize=18,
            spaceAfter=16,
        )

        self.section_style = ParagraphStyle(
            "SectionStyle",
            parent=self.styles["Heading2"],
            fontSize=13,
            spaceBefore=10,
            spaceAfter=8,
        )

        self.normal_style = self.styles["BodyText"]

    def generate(
        self,
        prediction: Dict,
        birads: Dict,
        confidence: Dict,
        heatmap: Optional[np.ndarray] = None,
        patient_id: str = "Demo Patient",
        save_name: Optional[str] = None,
    ) -> str:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_name is None:
            save_name = f"breast_cancer_report_{timestamp}.pdf"

        report_path = self.output_dir / save_name

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        story = []

        story.append(
            Paragraph(
                "AI-Assisted Breast Cancer Analysis Report",
                self.title_style,
            )
        )

        story.append(
            Paragraph(
                f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
                self.normal_style,
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                "Patient / Case Information",
                self.section_style,
            )
        )

        case_table = Table(
            [
                ["Patient ID", patient_id],
                ["System", "Explainable Multi-Task Breast Cancer AI"],
                ["Mode", "Research / Educational Use"],
            ],
            colWidths=[2.2 * inch, 3.8 * inch],
        )

        case_table.setStyle(self._table_style())
        story.append(case_table)
        story.append(Spacer(1, 14))

        story.append(
            Paragraph(
                "Prediction Summary",
                self.section_style,
            )
        )

        pred_table = Table(
            [
                ["Diagnosis", prediction.get("prediction", "N/A")],
                ["Cancer Probability", f"{prediction.get('confidence', 0):.2f}%"],
                ["BI-RADS", birads.get("birads", "N/A")],
                ["BI-RADS Confidence", f"{birads.get('confidence', 0):.2f}%"],
                ["Final Reliability", confidence.get("reliability", "N/A")],
                ["Uncertainty", f"{confidence.get('uncertainty', 0):.4f}"],
            ],
            colWidths=[2.2 * inch, 3.8 * inch],
        )

        pred_table.setStyle(self._table_style())
        story.append(pred_table)
        story.append(Spacer(1, 14))

        story.append(
            Paragraph(
                "Clinical Interpretation",
                self.section_style,
            )
        )

        diagnosis = prediction.get("prediction", "N/A")
        birads_label = birads.get("birads", "N/A")

        interpretation = self._generate_interpretation(
            diagnosis,
            birads_label,
            confidence,
        )

        story.append(
            Paragraph(
                interpretation,
                self.normal_style,
            )
        )

        story.append(Spacer(1, 14))

        if heatmap is not None:
            heatmap_path = self._save_temp_image(
                heatmap,
                f"report_heatmap_{timestamp}.png",
            )

            story.append(
                Paragraph(
                    "Explainability Heatmap",
                    self.section_style,
                )
            )

            story.append(
                ReportImage(
                    str(heatmap_path),
                    width=4.8 * inch,
                    height=4.8 * inch,
                )
            )

            story.append(Spacer(1, 14))

        story.append(
            Paragraph(
                "Important Disclaimer",
                self.section_style,
            )
        )

        story.append(
            Paragraph(
                "This report is generated by an AI research prototype. "
                "It is intended only for educational and research demonstration purposes. "
                "It must not be used as a standalone medical diagnosis. "
                "Final interpretation must be performed by qualified medical professionals.",
                self.normal_style,
            )
        )

        doc.build(story)

        return str(report_path)

    def _generate_interpretation(
        self,
        diagnosis: str,
        birads_label: str,
        confidence: Dict,
    ) -> str:

        reliability = confidence.get("reliability", "N/A")

        if diagnosis.lower() == "malignant":
            recommendation = (
                "The system detected imaging patterns associated with malignancy. "
                "Further expert radiological review is strongly recommended."
            )
        else:
            recommendation = (
                "The system did not detect strong malignant imaging patterns. "
                "However, clinical correlation and expert review are still recommended."
            )

        return (
            f"The model prediction is <b>{diagnosis}</b> with BI-RADS assessment "
            f"<b>{birads_label}</b>. The reliability level is <b>{reliability}</b>. "
            f"{recommendation}"
        )

    def _save_temp_image(
        self,
        image: np.ndarray,
        filename: str,
    ) -> Path:

        path = self.output_dir / filename

        img = image.copy()

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(path), img)

        return path

    def _table_style(self) -> TableStyle:

        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )


