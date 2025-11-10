"""
Core plotting utilities for the Climate Hedging Project.
Cleaned version: only includes plots actually used in the ARIMAX + Risk Index pipeline.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_global_risk_index(df: pd.DataFrame) -> plt.Figure:
    """Plot the global climate risk index over time."""
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["global_risk_0_100"], label="Global Risk Index")
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk (0–100)")
    ax.set_title("Global Climate Risk Index Over Time")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_price_forecast(forecast: pd.DataFrame) -> plt.Figure:
    """Plot ARIMAX price forecast with 95% confidence interval."""
    fig, ax = plt.subplots()
    x_axis = forecast["date"] if "date" in forecast.columns else range(len(forecast))
    ax.plot(x_axis, forecast["price_forecast"], label="Forecast")
    if {"lo95", "hi95"}.issubset(forecast.columns):
        ax.fill_between(x_axis, forecast["lo95"], forecast["hi95"], alpha=0.3, label="95% CI")
    ax.set_xlabel("Date" if "date" in forecast.columns else "Horizon (days)")
    ax.set_ylabel("Price")
    ax.set_title("ARIMAX Price Forecast")
    ax.legend()
    fig.autofmt_xdate()
    return fig
