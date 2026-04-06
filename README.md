# Email Metadata Analytics Platform

An end-to-end analytics solution for organizational email metadata. Analyzes **header-only** data (Date, Size, From, To) to infer structural coordination patterns, identify anomalies, and present findings in an interactive Streamlit dashboard.

## Features

- **20 interactive dashboard pages** covering volume trends, timing norms, network analysis, hierarchy inference, community detection, and more
- **Automatic ingestion** of CSV email metadata with custom quote-aware parsing
- **IMCEAEX Exchange address resolution** to standard email format
- **Network graph analysis** with PageRank, betweenness centrality, community detection (Louvain)
- **Temporal evolution** tracking with monthly snapshots and rising/fading detection
- **Response time analysis** using asof-join based reply detection
- **Nonhuman address detection** (copiers, bots, system accounts) via pattern + ratio analysis
- **Period comparison** with side-by-side KPI deltas
- **Data export** (CSV) on every analysis page
- **Click-to-drill-down** dialogs on charts and tables

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place CSV files in data/
# Expected columns: Date, Size, From, To

# Launch the dashboard
streamlit run app.py
```

## CSV Format

The platform expects CSV files with these columns:

| Column | Example |
|--------|---------|
| Date | `03/15/2010 14:30` |
| Size | `24 KB` or `1048576` |
| From | `"Smith, John" <jsmith@example.org>` |
| To | `"Doe, Jane" <jdoe@example.org>, user@external.com` |

Place CSV files in the `data/` directory. Multiple files are automatically merged. Subdirectories of `data/` create named datasets selectable from the sidebar.

## Dashboard Pages

1. **Executive Summary** -- Key findings at a glance with comparison mode
2. **Volume & Seasonality** -- Message flow trends with rolling averages
3. **Time Norms** -- Hour/day heatmap, after-hours trends, burstiness
4. **Broadcast & Attention** -- Mass-send patterns and inbox load
5. **Artifact vs Ping** -- Message size and purpose classification
6. **Network Map** -- Interactive PyVis network visualization
7. **Bottlenecks & Routing** -- Critical connectors by betweenness centrality
8. **Dyads & Asymmetry** -- Bidirectional relationship analysis
9. **Coordination & Churn** -- Community structure and activity patterns
10. **Risk Register** -- Statistical anomalies and flags
11. **External Contacts** -- Top external addresses by volume
12. **Search** -- Look up any email address with full profile
13. **Response Time** -- Reply detection and speed metrics
14. **Hierarchy Inference** -- Leadership pattern detection with reciprocal teams
15. **Silos & Bridges** -- Inter-community analysis with removal simulation
16. **Temporal Evolution** -- Monthly centrality trends and community stability
17. **Size Forensics** -- Size classification, templates, and anomalies
18. **Data Quality** -- Completeness gaps, zero-size messages, ingestion stats
19. **Narrative Insights** -- Auto-generated executive narrative
20. **Period Comparison** -- Side-by-side period analysis with deltas

## Technology Stack

- **Data engine**: Polars (DataFrame operations)
- **Graph analysis**: NetworkX + python-louvain (community detection)
- **Dashboard**: Streamlit + Plotly + PyVis
- **Cache**: Parquet files + pickle for graph objects (mtime-based invalidation)
- **Testing**: pytest

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
app.py                  # Streamlit entry point
data/                   # Source CSV files (gitignored)
cache/                  # Generated cache files (gitignored)
pages/                  # 20 dashboard pages
src/
  config.py             # AppConfig + DatasetConfig
  cache_manager.py      # Parquet/pickle I/O
  state.py              # Session state, loaders, filters
  export.py             # CSV/Excel download helpers
  drilldown.py          # Click-to-drill-down dialogs
  ingest/               # CSV parsing, email parsing, size parsing
  transform/            # Fact tables, weekly aggregation
  analytics/            # 13 analytics modules
tests/                  # Unit tests (60 tests)
```
