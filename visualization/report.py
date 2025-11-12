import os
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.units import inch

from visualization.charts import plot_price_forecast, plot_global_risk_index


# ============== Utility ==============
def _save_plot_to_file(fig: plt.Figure, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_market_prices(df: pd.DataFrame, commodity: str) -> plt.Figure:
    """Plot spot & futures prices for overview section (no embedded title)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["date"], df["price_spot"], label="Spot Price", color="#004b87", linewidth=1.8)
    if "price_front_fut" in df.columns:
        ax.plot(df["date"], df["price_front_fut"], label="Front Future", color="#999999", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    return fig


# ============== Header & Footer ==============
def _footer(canvas, doc):
    canvas.saveState()
    footer_text = "© 2025 Agrivise — Climate Risk & Hedging Intelligence"
    canvas.setFont("Helvetica", 8)
    page_width, _ = A4
    canvas.drawCentredString(page_width / 2.0, 0.5 * inch, footer_text)
    canvas.restoreState()


def _header(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.6 * inch, A4[1] - 0.5 * inch, f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    canvas.restoreState()


# ============== Report ==============
def generate_pdf_report_from_arimax(
    forecast_path: str,
    risk_path: str,
    hedge_path: str,
    output_path: str,
    title: Optional[str] = None,
) -> Optional[str]:
    if A4 is None:
        print("ReportLab not installed — cannot generate PDF.")
        return None

    # --- Load datasets
    forecast_df = pd.read_json(forecast_path)
    risk_df = pd.read_json(risk_path)
    with open(hedge_path) as f:
        hedge_rec = json.load(f)

    commodity = hedge_rec.get("commodity", "wheat").capitalize()
    title = title or f"{commodity} Climate Hedging Report"

    # --- Forecast horizon (dynamic)
    forecast_horizon = hedge_rec.get("forecast_horizon_weeks", 8)

    # --- Silver data
    silver_path = "data/silver/silver_data.csv"
    market_df = None
    if os.path.exists(silver_path):
        try:
            market_df = pd.read_csv(silver_path)
            market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")
            market_df = market_df.sort_values("date")
        except Exception:
            market_df = None

    # --- Temp
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(output_path), "tmp_charts")
    os.makedirs(tmp, exist_ok=True)

    # --- Charts
    print("[1/3] Generating charts...")
    fig_risk = plot_global_risk_index(risk_df)
    fig_forecast = plot_price_forecast(forecast_df)
    _save_plot_to_file(fig_risk, os.path.join(tmp, "risk.png"))
    _save_plot_to_file(fig_forecast, os.path.join(tmp, "forecast.png"))

    market_plot_path = None
    if market_df is not None and "price_spot" in market_df.columns:
        fig_market = plot_market_prices(market_df, commodity)
        market_plot_path = os.path.join(tmp, "market.png")
        _save_plot_to_file(fig_market, market_plot_path)

    # --- Build PDF
    print("[2/3] Assembling report...")
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Subtitle", fontSize=13, leading=15, spaceAfter=10, textColor=colors.HexColor("#004b87")))
    styles.add(ParagraphStyle(name="ChartTitle", fontSize=11, alignment=1, textColor=colors.HexColor("#004b87"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=11, textColor=colors.grey))
    styles.add(ParagraphStyle(name="BodyJustified", parent=styles["BodyText"], alignment=4, spaceAfter=10))

    story = []

    # --- Header / Title
    logo_path = os.path.join("visualization", "logo.png")
    story.append(Spacer(1, 0.25 * inch))
    if os.path.exists(logo_path):
        story.append(RLImage(logo_path, width=2.3 * inch, height=0.7 * inch))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.25 * inch))

    # --- Section 1: Hedging Input Summary
    story.append(Paragraph("1. Hedging Input Summary", styles["Subtitle"]))
    summary_data = [
        ["Profile", hedge_rec.get("profile", "balanced").capitalize()],
        ["Role", hedge_rec.get("role", "importer").capitalize()],
        ["Scenario", hedge_rec.get("scenario", "baseline").capitalize()],
        ["Risk Score", f"{hedge_rec.get('risk_score', 0):.1f} / 100"],
        ["Forecast Horizon", f"{forecast_horizon} weeks"],
        ["Exposure", f"{hedge_rec.get('exposure', 10000):,.0f} USD"],
        ["Suggested Hedge Ratio", f"{hedge_rec.get('hedge_ratio', 0)*100:.1f}%"],
        ["Hedge Notional", f"{hedge_rec.get('hedge_notional', 0):,.0f} units"],
        ["Instrument", hedge_rec.get("instrument", '—').capitalize()],
    ]
    table = Table(summary_data, colWidths=[2.2 * inch, 3.0 * inch])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))

    explanation = (
        f"Based on the <b>{hedge_rec.get('scenario', 'baseline')}</b> scenario and a global risk level of "
        f"<b>{hedge_rec.get('risk_score', 0):.1f}</b>, the suggested hedge ratio is "
        f"<b>{hedge_rec.get('hedge_ratio', 0)*100:.1f}%</b> of exposure "
        f"({hedge_rec.get('hedge_notional', 0):,.0f} units). "
        f"As a <b>{hedge_rec.get('role', 'importer')}</b> under a <b>{hedge_rec.get('profile', 'balanced')}</b> profile, "
        f"the recommended instrument is <b>{hedge_rec.get('instrument', 'long call options')}</b>. "
        f"This reflects Agrivise’s {forecast_horizon}-week ARIMAX forecast and climate risk assessment."
    )
    story.append(Paragraph(explanation, styles["BodyJustified"]))
    story.append(Spacer(1, 0.25 * inch))

    # --- Section 2: Market Prices
    story.append(Paragraph("2. Market Prices Overview", styles["Subtitle"]))
    story.append(Paragraph(
        "Observed spot and futures prices (15-year rolling window) used in model inputs.",
        styles["BodyJustified"]))
    story.append(Spacer(1, 0.15 * inch))
    if market_plot_path:
        story.append(Paragraph("Market Prices (Spot & Futures)", styles["ChartTitle"]))
        story.append(RLImage(market_plot_path, width=5.4 * inch, height=2.4 * inch))
    story.append(Spacer(1, 0.15 * inch))
    story.append(PageBreak())

    # --- Page 2: Risk + Forecast
    story.append(Paragraph("3. Global Climate Risk Index", styles["Subtitle"]))
    story.append(Paragraph(
        "Aggregated anomalies (temperature, precipitation, NDVI) across key producing regions.",
        styles["BodyJustified"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Global Climate Risk Index Over Time", styles["ChartTitle"]))
    story.append(RLImage(os.path.join(tmp, "risk.png"), width=5.6 * inch, height=3.0 * inch))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("4. ARIMAX Price Forecast", styles["Subtitle"]))
    story.append(Paragraph(
        f"The ARIMAX model combines historical prices with the global risk index to forecast the next {forecast_horizon} weeks.",
        styles["BodyJustified"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"ARIMAX Price Forecast ({forecast_horizon}-week Horizon)", styles["ChartTitle"]))
    story.append(RLImage(os.path.join(tmp, "forecast.png"), width=5.6 * inch, height=3.0 * inch))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph(
        "Disclaimer: This report is provided for informational purposes only and does not constitute financial advice. "
        "Agrivise accepts no responsibility for investment or hedging decisions based on this report.",
        styles["Small"],
    ))

    doc.build(story, onFirstPage=lambda c, d: (_header(c, d), _footer(c, d)), onLaterPages=_footer)
    print(f"[3/3] PDF report generated: {output_path}")

    try:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    except OSError:
        pass

    return os.path.abspath(output_path)

