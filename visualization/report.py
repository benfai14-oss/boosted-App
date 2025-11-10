"""
Report generation utilities for the Climate Hedging Project
===========================================================

This module compiles results from:
- the global climate risk index,
- the ARIMAX price forecast,
- the hedging recommendation,

into a readable PDF report with visuals and narrative.

The function `generate_pdf_report_from_arimax()` is the high-level entry point.
It automatically reads JSON data files, generates charts, and produces
a finished PDF for business users.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt

# --- PDF support (ReportLab) ---
try:
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image as RLImage,
    )  # type: ignore
    from reportlab.lib.units import inch  # type: ignore
except ImportError:
    A4 = None  # type: ignore

# --- Plotting utilities ---
from visualization.charts import (
    plot_global_risk_index,
    plot_price_forecast,
)

# -------------------------------------------------------------------
# Helper to save figures
# -------------------------------------------------------------------
def _save_plot_to_file(fig: plt.Figure, path: str) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Main function: Generate PDF report directly from JSONs
# -------------------------------------------------------------------
def generate_pdf_report_from_arimax(
    forecast_path: str,
    risk_path: str,
    hedge_path: str,
    output_path: str,
    title: str = "Climate Risk and Hedging Report",
) -> Optional[str]:
    """
    Generate a PDF report summarising risk, ARIMAX forecasts,
    and the hedging recommendation.

    Parameters
    ----------
    forecast_path : str
        Path to ARIMAX forecast JSON file (e.g. soybean_forecast.json).
    risk_path : str
        Path to global risk index JSON file (e.g. soybean_global_index.json).
    hedge_path : str
        Path to hedge recommendation JSON file (e.g. soybean_hedge_rec.json).
    output_path : str
        Path where the final PDF will be written.
    title : str, optional
        Title displayed on the PDF.

    Returns
    -------
    str or None
        Absolute path of the generated report, or None if ReportLab not installed.
    """
    if A4 is None:
        print("⚠️ ReportLab not installed — cannot generate PDF.")
        return None

    # --- Load inputs ---
    forecast_df = pd.read_json(forecast_path)
    risk_df = pd.read_json(risk_path)
    with open(hedge_path) as f:
        hedge_rec = json.load(f)

    # --- Prepare output folders ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_dir = os.path.join(os.path.dirname(output_path), "tmp_charts")
    os.makedirs(temp_dir, exist_ok=True)

    # --- Generate plots ---
    print("[1/3] Generating charts...")
    fig1 = plot_global_risk_index(risk_df)
    fig2 = plot_price_forecast(forecast_df)
    risk_plot_path = os.path.join(temp_dir, "risk_index.png")
    forecast_plot_path = os.path.join(temp_dir, "price_forecast.png")
    _save_plot_to_file(fig1, risk_plot_path)
    _save_plot_to_file(fig2, forecast_plot_path)

    # --- Build PDF ---
    print("[2/3] Assembling report...")
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Italic"]))
    story.append(Spacer(1, 0.25 * inch))

    # --- Section 1: Risk Index ---
    story.append(Paragraph("Global Climate Risk Index", styles["Heading2"]))
    story.append(Paragraph(
        "The global climate risk index captures temperature, precipitation, "
        "and vegetation anomalies across key producing regions. Values above "
        "70 indicate elevated risk.", styles["BodyText"]))
    story.append(RLImage(risk_plot_path, width=5*inch, height=3*inch))
    story.append(Spacer(1, 0.2 * inch))

    # --- Section 2: Forecast ---
    story.append(Paragraph("Price Forecasts (ARIMAX Model)", styles["Heading2"]))
    story.append(Paragraph(
        "Forecasts are derived from an ARIMAX model using the climate risk index "
        "as an exogenous driver. The shaded area represents the 95% confidence interval.",
        styles["BodyText"]))
    story.append(RLImage(forecast_plot_path, width=5*inch, height=3*inch))
    story.append(Spacer(1, 0.2 * inch))

    # --- Section 3: Hedging Recommendation ---
    story.append(Paragraph("Hedging Recommendation", styles["Heading2"]))
    story.append(Paragraph(
        "Based on your ARIMAX forecast trend, climate risk level, and chosen profile, "
        "the following hedging action is suggested:", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(f"<b>Scenario:</b> {hedge_rec.get('scenario', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Risk Score:</b> {hedge_rec.get('risk_score', 0):.1f}", styles["Normal"]))
    story.append(Paragraph(f"<b>Forecast Price:</b> {hedge_rec.get('forecast_price', 0):.2f}", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"<b>Recommended Hedge Ratio:</b> {hedge_rec.get('hedge_ratio', 0)*100:.1f}%", styles["Normal"]))
    story.append(Paragraph(f"<b>Instrument:</b> {hedge_rec.get('instrument', '')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Hedge Notional:</b> {hedge_rec.get('hedge_notional', 0):,.2f}", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(hedge_rec.get("summary", ""), styles["BodyText"]))
    story.append(Spacer(1, 0.25 * inch))

    # --- Disclaimer ---
    story.append(Paragraph(
        "Disclaimer: This report is for informational purposes only and does not "
        "constitute financial advice. Please consult a professional before taking "
        "any trading or hedging decisions.", styles["Italic"]))

    doc.build(story)
    print(f"[3/3] ✅ PDF report generated: {output_path}")

    # --- Cleanup temp charts ---
    try:
        os.remove(risk_plot_path)
        os.remove(forecast_plot_path)
        os.rmdir(temp_dir)
    except OSError:
        pass

    return os.path.abspath(output_path)
