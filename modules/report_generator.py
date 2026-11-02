"""
Report Generator Module
Converts AI analysis reports and quantitative financial summaries
into professional PDF and text documents using ReportLab.
"""

import io
import re
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds page numbers and running header/footer to PDF documents."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header
        self.drawString(54, 750, "Enriching Stock Market Data using OpenAI API | Equity Research Intelligence")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Footer
        self.line(54, 45, 558, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.drawString(54, 32, "Confidential & Academic Demonstration — Not Financial Advice")
        self.restoreState()


def generate_pdf_report(report_markdown: str, ticker: str, company_name: str) -> bytes:
    """
    Parses Markdown report text and renders a clean, styled PDF buffer.
    
    Args:
        report_markdown: Markdown formatted report text
        ticker: Stock symbol
        company_name: Company name
        
    Returns:
        bytes of the compiled PDF document
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=64,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        leftIndent=15,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=4,
    )

    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#718096"),
        spaceBefore=8,
    )

    story = []

    # Process markdown lines
    lines = report_markdown.split("\n")
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        # Clean bold and italic markdown tags to ReportLab HTML tags
        line_clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
        line_clean = re.sub(r"\*(.*?)\*", r"<i>\1</i>", line_clean)

        if line.startswith("# "):
            title_text = line_clean[2:].strip()
            story.append(Paragraph(title_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
        elif line.startswith("## "):
            h2_text = line_clean[3:].strip()
            story.append(Paragraph(h2_text, h2_style))
        elif line.startswith("### "):
            h3_text = line_clean[4:].strip()
            story.append(Paragraph(f"<b>{h3_text}</b>", body_style))
        elif line.startswith("- ") or line.startswith("• "):
            b_text = line_clean[2:].strip()
            story.append(Paragraph(f"&bull; {b_text}", bullet_style))
        elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. ") or line.startswith("4. "):
            story.append(Paragraph(line_clean, bullet_style))
        elif line.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceBefore=6, spaceAfter=6))
        elif "DISCLAIMER" in line.upper():
            story.append(Paragraph(line_clean, disclaimer_style))
        else:
            story.append(Paragraph(line_clean, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
