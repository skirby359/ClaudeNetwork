# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Email Metadata Analytics Platform — an end-to-end analytics solution for organizational email metadata (header-only: Date, Size, From, To). Infers structural coordination patterns, identifies exceptions, and presents findings in an interactive Streamlit dashboard.

## Project Structure

- `data/` — Source CSVs (e.g., `tikor-success-2010-done.csv`, ~100MB, 328K rows of Spokane County email metadata from 2010)
- `src/` — Python source code
  - `src/config.py` — AppConfig + DatasetConfig dataclasses
  - `src/cache_manager.py` — Parquet/pickle I/O with mtime invalidation
  - `src/state.py` — Streamlit session state, @st.cache_data loaders, date filters, comparison mode
  - `src/export.py` — CSV/Excel download button helpers
  - `src/drilldown.py` — Click-to-drill-down dialog infrastructure (person, community, week, dyad)
  - `src/ingest/` — CSV parsing, email parsing, size parsing, normalization
  - `src/transform/` — fact tables, weekly aggregation, timing, broadcast metrics
  - `src/analytics/` — Analytics modules:
    - `volume.py` — Trends, rolling averages, Gini coefficient, concentration
    - `network.py` — Graph build, centrality, communities, dyads
    - `timing_analytics.py` — Heatmaps, after-hours trends, burstiness, ping-pong
    - `broadcast_analytics.py` — Blast impact, high-blast senders, attention metrics
    - `anomaly.py` — Statistical outlier detection (z-score based)
    - `response_time.py` — Reply detection and response speed metrics
    - `hierarchy.py` — Nonhuman detection, hierarchy scoring, reciprocal team inference
    - `silos.py` — Community interaction matrix, silent pairs, bridges, removal simulation
    - `temporal_network.py` — Monthly snapshots, centrality trends, NMI stability
    - `size_forensics.py` — Size classification, templates, anomalies
    - `data_quality.py` — Zero-size messages, missing names, completeness
    - `narrative.py` — Auto-generated executive narrative
    - `comparison.py` — Period-over-period KPI deltas
- `pages/` — 20 Streamlit dashboard pages (01–20)
- `cache/` — Generated parquet/pickle files (gitignored)
- `tests/` — Unit tests for parsers and analytics modules
- `app.py` — Streamlit entry point

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/ -v

# Run ingestion pipeline only
python -c "from src.ingest.pipeline import run_ingestion; run_ingestion()"

# Launch dashboard
streamlit run app.py
```

## Technology Stack

- **Data engine**: DuckDB (SQL aggregation) + Polars (DataFrame ops)
- **Graph**: NetworkX + python-louvain
- **Dashboard**: Streamlit + Plotly + PyVis
- **Cache**: Parquet files + pickle for graph objects

## Key Design Decisions

- CSV parser is custom (quote-aware, not Python's csv module) because the To field contains comma-separated recipients that spill across CSV columns
- IMCEAEX Exchange addresses are resolved to normal email format
- All pipeline stages use mtime-based cache invalidation (if cache is newer than source, skip)
- Network graph uses betweenness centrality with k=500 sampling for graphs >5000 nodes

## Environment

- Python execution is pre-authorized in `.claude/settings.local.json`
