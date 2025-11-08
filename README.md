# boosted-App
Boosted-App: Climate and Commodity Risk Pipeline

This project builds a complete data pipeline that connects climate signals, agricultural fundamentals, and commodity market dynamics.

The goal is to understand how weather and production factors can influence market prices, and to create a reproducible framework for monitoring climate-related risks.

Project structure

The repository is organized in layers, following a clear data engineering logic:

Layer A – Ingestion

Located in ingestion/ and controlled by scripts/pull_all.py.

This layer collects and standardizes external data from different sources:

Weather: Weekly temperature and precipitation anomalies per region (from the Open-Meteo API, with deterministic fallback if offline).

Market: Weekly commodity prices and realized volatility (from Yahoo Finance proxies).

Agriculture: Production and stock proxies by region (based on World Bank data or synthetic defaults).

All cleaned outputs are stored under:

data/silver/

Layer B – Transformation / Climate Index

Located in climate_index/.

This layer combines the silver datasets to build simple climate indicators:

regional climate stress scores

a global climate-commodity index

Layer C – Models & Hedging

Located in market_models/ and hedging/.

This layer tests relationships between climate variables and markets, and simulates how hedging or structured products could be affected.

Layer D – Interface / Visualization

Contains command-line tools and visualization stubs (under interface/ and visualization/).
The objective is to make the results accessible through dashboards or reports.
