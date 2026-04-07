# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Email Metadata Analytics Platform — an end-to-end analytics solution for organizational email metadata (header-only: Date, Size, From, To). Infers structural coordination patterns, identifies exceptions, and presents findings in an interactive Streamlit dashboard. Designed for consulting engagements with government, legal, and HR clients.

## Project Structure

- `data/` — Source CSVs (e.g., Spokane County email metadata)
- `profiles/` — Saved engagement profiles (per-client JSON configs, gitignored)
- `src/` — Python source code
  - `src/config.py` — AppConfig + DatasetConfig dataclasses
  - `src/cache_manager.py` — Parquet/pickle I/O with mtime invalidation
  - `src/state.py` — Streamlit session state, @st.cache_data loaders, date filters, comparison mode, department enrichment
  - `src/export.py` — CSV/Excel/GraphML/JSON download button helpers
  - `src/export_pptx.py` — PowerPoint full report generation
  - `src/export_html.py` — HTML self-contained report with inline charts
  - `src/export_memo.py` — Executive memo (short consulting deliverable with auto-recommendations)
  - `src/engagement.py` — Engagement profiles (save/load per-client settings), alert rules engine
  - `src/drilldown.py` — Click-to-drill-down dialog infrastructure (person, community, week, dyad)
  - `src/anonymize.py` — Display-layer email anonymization
  - `src/ingest/` — Data intake
    - `csv_parser.py` — Custom quote-aware CSV parser
    - `email_parser.py` — Email address parsing, IMCEAEX resolution
    - `size_parser.py` — Size string parsing (10.8K → bytes)
    - `normalizer.py` — Email/name normalization, distribution list detection
    - `profiler.py` — Auto-detect encoding, delimiter, date format, column roles
    - `mailbox_import.py` — PST/MBOX file import
    - `msgraph.py` — Microsoft Graph API connector
    - `pipeline.py` — Full ingestion orchestration with per-file chunk caching
  - `src/transform/` — Fact tables, weekly aggregation, timing, broadcast metrics
  - `src/analytics/` — Analytics modules:
    - `volume.py` — Trends, rolling averages, Gini coefficient, concentration
    - `network.py` — Graph build, centrality, Louvain communities, dyads
    - `community_leiden.py` — Leiden multi-resolution communities, type classification, hierarchy nesting
    - `timing_analytics.py` — Heatmaps, after-hours trends, burstiness, ping-pong
    - `broadcast_analytics.py` — Blast impact, high-blast senders, attention metrics
    - `anomaly.py` — Statistical outlier detection (z-score based)
    - `response_time.py` — Reply detection (asof-join) and response speed metrics
    - `hierarchy.py` — Nonhuman detection + type classification, hierarchy scoring, reciprocal teams
    - `silos.py` — Community interaction matrix, silent pairs, bridges, removal simulation
    - `temporal_network.py` — Monthly snapshots, centrality trends, NMI stability
    - `structural_change.py` — Community shift classification (split/merge/reorg), node switches
    - `compliance.py` — Blackout windows, external spikes, key date gaps, after-hours clusters
    - `cascade.py` — Information forwarding chains, cascade metrics, amplifier identification
    - `bus_factor.py` — Articulation points, team bus factor, succession readiness
    - `size_forensics.py` — Size classification, templates, anomalies
    - `data_quality.py` — Zero-size messages, missing names, completeness
    - `health_score.py` — Composite 0-100 health score with 6 sub-dimensions, monthly trend
    - `narrative.py` — Auto-generated executive narrative
    - `comparison.py` — Period-over-period KPI deltas
- `pages/` — 30 Streamlit dashboard pages (00–29)
- `cache/` — Generated parquet/pickle files (gitignored)
- `tests/` — 181 tests (unit + integration)
- `app.py` — Streamlit entry point

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Launch dashboard
streamlit run app.py
```

## Technology Stack

- **Data engine**: Polars (DataFrame ops)
- **Graph**: NetworkX + python-louvain + leidenalg (optional)
- **Dashboard**: Streamlit + Plotly + PyVis
- **Exports**: python-pptx (PowerPoint), kaleido (chart images for HTML)
- **Cache**: Parquet files + pickle for graph objects
- **Auth**: MSAL (Microsoft Graph)

## Key Design Decisions

- CSV parser is custom (quote-aware, not Python's csv module) because the To field contains comma-separated recipients that spill across CSV columns
- IMCEAEX Exchange addresses are resolved to normal email format
- All pipeline stages use mtime-based cache invalidation (if cache is newer than source, skip)
- Network graph uses betweenness centrality with k=500 sampling for graphs >5000 nodes
- Community detection: Louvain (resolution 0.5, merge small, auto-label) with optional Leiden multi-resolution
- message_fact has list columns (to_emails, to_names) — must drop these before CSV export
- Department enrichment merges into person_dim at load time from session state
- Engagement profiles saved as JSON in profiles/ directory

## Dashboard Pages (30)

**Settings:** 00 — Data upload, profiles, department mapping, data profiler, MS365 connector

**Executive View:** 01 Executive Summary, 10 Risk Register, 29 Alert Dashboard, 19 Narrative

**Organizational Health:** 03 Time Norms, 13 Response Time, 18 Data Quality, 22 Health Score

**People & Structure:** 07 Bottlenecks, 14 Hierarchy, 15 Silos & Bridges, 27 Bus Factor

**Communication Patterns:** 02 Volume, 04 Broadcast, 05 Artifact vs Ping, 08 Dyads, 11 External, 12 Search, 28 Person Comparison

**Deep Dives:** 06 Network Map, 09 Coordination, 16 Temporal Evolution, 17 Size Forensics, 20 Comparison, 21 Automated Systems

**Advanced Analytics:** 23 Community v2 (Leiden), 24 Structural Change, 25 Compliance, 26 Cascades

## Environment

- Python execution is pre-authorized in `.claude/settings.local.json`
