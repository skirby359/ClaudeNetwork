# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Email Metadata Analytics Platform — an end-to-end analytics solution for organizational email metadata (header-only: Date, Size, From, To). Infers structural coordination patterns, identifies exceptions, and presents findings in an interactive Streamlit dashboard.

## Project Structure

- `data/` — Source CSVs (e.g., `tikor-success-2010-done.csv`, ~100MB, 328K rows of Spokane County email metadata from 2010)
- `src/` — Python source code
  - `src/config.py` — AppConfig + DatasetConfig dataclasses
  - `src/cache_manager.py` — Parquet/pickle I/O with mtime invalidation
  - `src/state.py` — Streamlit session state and @st.cache_data loaders
  - `src/ingest/` — CSV parsing, email parsing, size parsing, normalization
  - `src/transform/` — fact tables, weekly aggregation, timing, broadcast metrics
  - `src/analytics/` — volume, network, timing, broadcast, anomaly analytics
- `pages/` — 10 Streamlit dashboard pages
- `cache/` — Generated parquet/pickle files (gitignored)
- `tests/` — Unit tests for parsers
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
