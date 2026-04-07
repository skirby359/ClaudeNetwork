# SPEC — Email Metadata Analytics Platform

## Product Vision

Platform-agnostic email communication pattern analysis from metadata only (Date, Size, From, To). No email bodies accessed. Runs on-premise or analyst-operated. Primary market: government oversight, legal e-discovery, HR consulting.

## Data Model

### Input Sources
- **CSV files** with columns: Date, Size, From, To (configurable via column mapping)
- **Microsoft Graph API** via Mail.ReadBasic.All scope (metadata only, body access blocked at API level)
- **MBOX files** via Python's built-in mailbox module
- **PST files** via optional pypff/libpff-python

### Core Tables (Polars DataFrames, cached as Parquet)

**message_fact** — one row per email message
| Column | Type | Description |
|--------|------|-------------|
| msg_id | Int64 | Sequential ID |
| timestamp | Datetime | Send time |
| size_bytes | Int64 | Message size |
| from_email | Utf8 | Sender (lowercase) |
| from_name | Utf8 | Sender display name |
| to_emails | List[Utf8] | Recipient list |
| to_names | List[Utf8] | Recipient names |
| n_recipients | Int64 | Count of recipients |
| week_id | Utf8 | ISO week (e.g., "2017-W03") |
| hour | Int32 | Hour of day (0-23) |
| day_of_week | Int32 | 0=Mon, 6=Sun |
| is_after_hours | Bool | Before 7AM or after 6PM |
| is_weekend | Bool | Saturday or Sunday |

**edge_fact** — one row per sender→recipient pair per message (exploded from message_fact)

**person_dim** — one row per unique email address
| Column | Type | Description |
|--------|------|-------------|
| email | Utf8 | Unique identifier |
| display_name | Utf8 | Best known name |
| domain | Utf8 | Email domain |
| is_internal | Bool | Matches configured internal domains |
| is_distribution_list | Bool | Heuristic DL detection |
| total_sent | Int64 | Messages sent |
| total_received | Int64 | Messages received |
| department | Utf8 | From CSV upload or domain fallback |

**graph_metrics** — one row per person with network metrics
| Column | Type | Description |
|--------|------|-------------|
| email | Utf8 | Person |
| in_degree | Int64 | Weighted incoming edges |
| out_degree | Int64 | Weighted outgoing edges |
| betweenness_centrality | Float64 | Bridge importance |
| community_id | Int64 | Community assignment |
| community_label | Utf8 | Auto-generated label |
| pagerank | Float64 | Network importance score |

## Analytics Modules (src/analytics/)

| Module | Key Functions | Input Tables |
|--------|--------------|--------------|
| volume.py | Gini, concentration, rolling averages | weekly_agg, edge_fact |
| network.py | Graph build, PageRank, betweenness, Louvain communities, dyads | edge_fact |
| community_leiden.py | Leiden multi-resolution, type classification, hierarchy nesting | edge_fact |
| timing_analytics.py | Heatmap, after-hours trends, burstiness, ping-pong | message_fact, edge_fact |
| broadcast_analytics.py | Blast tiers, high-blast senders | message_fact |
| anomaly.py | Z-score volume/sender anomalies | weekly_agg, edge_fact |
| response_time.py | Reply detection (asof-join), per-person/dept stats | edge_fact |
| hierarchy.py | Nonhuman detection + type classification, hierarchy score, reciprocal teams | edge_fact, person_dim |
| silos.py | Community interaction matrix, bridges, removal simulation | edge_fact, graph_metrics |
| temporal_network.py | Monthly snapshots, centrality trends, NMI stability | edge_fact |
| structural_change.py | Split/merge/reorg classification, node switches, NMI alerts | edge_fact (monthly) |
| compliance.py | Blackout windows, external spikes, key date gaps, after-hours clusters | message_fact, edge_fact |
| cascade.py | Forwarding chain detection, cascade metrics, amplifier identification | edge_fact |
| bus_factor.py | Articulation points, team bus factor, succession readiness, risk matrix | edge_fact, graph_metrics |
| size_forensics.py | Size classes, templates, per-sender profiles, anomalies | message_fact |
| data_quality.py | Zero-size, missing names, completeness | message_fact |
| health_score.py | Composite 0-100 score (6 dimensions), monthly trend | message_fact, edge_fact, graph_metrics |
| narrative.py | Auto-generated executive markdown | message_fact, weekly_agg, edge_fact, person_dim |
| comparison.py | Period-over-period KPI deltas | message_fact, edge_fact |

## Export Formats

| Format | Module | Description |
|--------|--------|-------------|
| CSV | export.py | Per-table download with anonymization support |
| Excel | export.py | XLSX via openpyxl |
| GraphML | export.py | Network graph with node attributes for Gephi/yEd |
| JSON | export.py | Network graph as nodes + edges for web visualization |
| PowerPoint | export_pptx.py | Full branded report: KPIs, tables, communities, bridges |
| HTML | export_html.py | Self-contained report with inline base64 chart images |
| Executive Memo | export_memo.py | Short consulting deliverable: cover, KPIs, health, risks, recommendations |

## Engagement Profile System

Per-client settings saved as JSON in `profiles/` directory:
- Internal domains, date format, column mapping
- Department mapping (companion CSV)
- Nonhuman toggle, key dates
- Alert rules with configurable thresholds
- Organization name

## Alert Rules Engine

6 default rules, each configurable per-engagement:
- After-hours rate > threshold per person
- Bus factor <= threshold per team
- Communication blackout > N hours
- External contact spike > N standard deviations
- Concentration (Gini) > threshold
- Health score < threshold

## Privacy Model

- **Metadata only**: No email body, subject, or attachment content accessed or stored
- **Microsoft Graph**: Uses Mail.ReadBasic.All — body access blocked at API level (returns 403)
- **Anonymization toggle**: Replaces emails with deterministic user_XXXXXX@domain aliases
- **Credentials**: Stored in .env.local (gitignored), loaded via environment variables
- **Cache**: Parquet files on local disk
- **Export**: CSV/Excel/HTML downloads respect anonymization toggle

## Data Intake

- **CSV**: Custom quote-aware parser handles malformed To fields with spilled recipients
- **Data profiler**: Auto-detects encoding, delimiter, date format, column roles
- **MBOX**: Native Python mailbox module, extracts Date/From/To/Cc/Size headers
- **PST**: Optional pypff dependency, walks folder tree extracting transport headers
- **Microsoft Graph**: App-only auth via MSAL, Mail.ReadBasic.All permission

## Community Detection

- **Louvain** (default): Resolution 0.5, pre-filter nonhuman, merge tiny (<3 members), auto-label
- **Leiden** (optional): Multi-resolution (coarse/medium/fine), type classification (pair/team/department/cross-functional), hierarchical nesting treemap

## Target Buyers

1. **Government oversight** (Inspector General, oversight boards) — $15-50K/engagement
2. **Legal e-discovery** (law firms, compliance) — $10-50K/engagement
3. **HR/ONA consulting** (org design, M&A due diligence) — $5-15K/project

## Tech Stack

- **Data**: Polars, NetworkX, python-louvain, leidenalg, igraph, scipy
- **Dashboard**: Streamlit, Plotly, PyVis
- **Exports**: python-pptx, kaleido (chart rendering), openpyxl
- **Cache**: Parquet (mtime invalidation), pickle (graph objects)
- **Auth**: MSAL (Microsoft Graph)
- **Deployment**: Docker, docker-compose
- **Testing**: pytest (181 tests — 89 unit + 59 integration + 33 coverage gap)
