Boosted-App: Climate-Driven Commodity Risk & Hedging Pipeline

1. Purpose & High-Level Architecture

Boosted-App is an end-to-end analytical pipeline that integrates climate risk, agricultural supply dynamics, and commodity price modeling to generate defensible and transparent hedging recommendations.

The project is built around four modular layers:
	1.	Layer A — Ingestion: Retrieve and standardize market, weather, and supply data
	2.	Layer B — Climate Index: Build a global climate stress index (0–100)
	3.	Layer C — Market Modeling & Hedging: ARIMAX + hedging logic
	4.	Layer D — Dashboard Interface: Interactive Streamlit exploration

This structure ensures transparency, reproducibility, and production-grade organization.

⸻

2. Repository Structure (Logical Overview)

📁 ingestion/

Layer A. Fetches historical weather anomalies, supply data, and commodity price time series.
Main script:
	•	scripts/pull_all.py → Produces standardized datasets for a selected commodity.

⸻

📁 data/silver/

Cleaned, standardized intermediate datasets:
	•	Weather anomalies
	•	Market prices
	•	Agricultural supply
	•	Merged silver_data.csv

These are aligned and ready for climate signal processing.

⸻

📁 climate_index/

Layer B. Builds the Global Climate Stress Index using:
	•	Weather anomalies
	•	Supply shifts
	•	Commodity-specific region weights
Output:
data/gold/<commodity>_global_index.json

📁 data/gold/

Final analytical outputs, consumed by the dashboard:
	•	Climate index
	•	ARIMAX forecasts (with confidence intervals)
	•	Hedging recommendation
	•	Raw merged datasets

⸻

📁 market_models/

Layer C. Implements the extended ARIMAX model:
	•	Log returns
	•	Lagged features
	•	Climate risk factor
	•	Seasonality
	•	Forecast intervals

Output:
data/gold/<commodity>_forecast.json


⸻

📁 hedging/

Layer C. Business-oriented hedging logic.
hedging/advanced.py computes:
	•	Hedge ratio
	•	Hedge notional
	•	Instrument (future/option)
	•	Recommended maturity (last Thursday ≤ last forecast date)
	•	Indicative strike
	•	Justification text

Output:
data/gold/<commodity>_hedge_rec.json


⸻

📁 interface/

Layer D. The unified CLI (interface/cli.py) and Streamlit dashboard.
CLI supports:
	•	Full pipeline run
	•	Independent execution of each layer
	•	Launching the Streamlit dashboard

⸻

📁 interface/streamlit_app.py

Interactive final dashboard including:
	•	Hedging recommendations
	•	Price forecasts + uncertainty bands
	•	Climate risk evolution
	•	Historical prices
	•	Raw data exploration

⸻

📁 visualization/

Plot helpers and legacy PDF generator (now replaced by Streamlit).

⸻

📁 configs/

Commodity-level metadata and region mappings.

⸻

📁 utils/

Logging helpers, documentation generators, safe file operations.

⸻

📁 scripts/

Developer utilities and test harnesses (test_layer_c.py, etc.).

⸻

3. Installation & Setup

Follow these steps to install dependencies and run the pipeline.

⸻

3.1 Create & Activate a Virtual Environment

Using Python venv:
python3 -m venv .venv
source .venv/bin/activate

Using Conda:
conda create -n boosted python=3.10 -y
conda activate boosted


⸻


3.2 Install Python Dependencies

Ensure you are in the repository root, then run:
pip install -r requirements.txt

If Streamlit is missing due to environment issues:
pip install streamlit

4. Running the Pipeline

4.1 Run the Entire Pipeline (Full End-to-End)

This executes:
	1.	Ingestion
	2.	Climate index computation
	3.	ARIMAX forecasting
	4.	Hedging recommendation
	5.	Streamlit dashboard

Run:
python -m interface.cli full-run \
  --commodity wheat \
  --profile balanced \
  --role importer \
  --exposure 10000

  A Streamlit dashboard will automatically launch.

⸻

5. Running Each Step Separately

5.1 Layer A — Ingestion

Creates the full 15-year rolling dataset:
python -m interface.cli ingest \
  --commodity wheat \
  --regions europe \
  --start 2023-01-01 \
  --end 2024-01-01


⸻

5.2 Layer B — Climate Index

Builds the climate stress indicator:
python -m interface.cli climate-index \
  --commodity wheat

⸻

5.3 Layer C — Market Model Forecast

Runs the ARIMAX model:
python -m interface.cli market-model \
  --commodity wheat

⸻

5.4 Layer C — Hedging Recommendation

Creates the hedging strategy:
python -m interface.cli hedge \
  --commodity wheat \
  --profile balanced \
  --role importer \
  --exposure 10000

⸻

5.5 Layer D — Streamlit Dashboard Only

Open the visualization interface without running the pipeline:
python -m interface.cli report \
  --commodity wheat

⸻

6. End-to-End Workflow Summary

The full system operates as follows:
	1.	Data ingestion: download & merge external sources
	2.	Climate index: compute a global climate-driven risk metric
	3.	Price modeling: ARIMAX with seasonality + climate → forecast
	4.	Hedging logic: instrument, notional, maturity, strike, rationale
	5.	Visualization: a clean, interactive dashboard for users

This makes the project suitable for:
	•	Corporates needing climate-aware hedging
	•	Analysts exploring market + climate interactions
	•	Developers integrating modular components
	•	Academia researching climate risk in commodities

⸻

7. Future Extensions

Potential enhancements include:
	•	Multi-maturity hedging ladder
	•	ML-based forecast benchmarks
	•	Scenario-based climate shocks
	•	Value-at-Risk climate-adjusted simulations
	•	Multi-commodity cross-analysis

⸻

8. License

MIT License — fully open-source for academic and industrial use.
