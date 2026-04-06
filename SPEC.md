# SPEC — Email Metadata Analytics Platform

## Product Vision

Platform-agnostic email communication pattern analysis from metadata only (Date, Size, From, To). No email bodies accessed. Runs on-premise or analyst-operated. Primary market: government oversight, legal e-discovery, HR consulting.

## Data Model

### Input
- **CSV files** with columns: Date, Size, From, To (configurable via column mapping)
- **Microsoft Graph API** via Mail.ReadBasic.All scope (metadata only, body access blocked at API level)
- **Future**: PST/MBOX import, Gmail API

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

**graph_metrics** — one row per person with network metrics
| Column | Type | Description |
|--------|------|-------------|
| email | Utf8 | Person |
| in_degree | Int64 | Weighted incoming edges |
| out_degree | Int64 | Weighted outgoing edges |
| betweenness_centrality | Float64 | Bridge importance |
| community_id | Int64 | Louvain community assignment |
| pagerank | Float64 | Network importance score |

## Analytics Modules (src/analytics/)

| Module | Key Functions | Input Tables |
|--------|--------------|--------------|
| volume.py | Gini, concentration, rolling averages | weekly_agg, edge_fact |
| network.py | Graph build, PageRank, betweenness, communities, dyads | edge_fact |
| timing_analytics.py | Heatmap, after-hours trends, burstiness, ping-pong | message_fact, edge_fact |
| broadcast_analytics.py | Blast tiers, high-blast senders | message_fact |
| anomaly.py | Z-score volume/sender anomalies | weekly_agg, edge_fact |
| response_time.py | Reply detection (asof-join), per-person/dept stats | edge_fact |
| hierarchy.py | Nonhuman detection, hierarchy score, reciprocal teams | edge_fact, person_dim |
| silos.py | Community interaction matrix, bridges, removal simulation | edge_fact, graph_metrics |
| temporal_network.py | Monthly snapshots, centrality trends, NMI stability | edge_fact |
| size_forensics.py | Size classes, templates, per-sender profiles, anomalies | message_fact |
| data_quality.py | Zero-size, missing names, completeness | message_fact |
| narrative.py | Auto-generated executive markdown | message_fact, weekly_agg, edge_fact, person_dim |
| comparison.py | Period-over-period KPI deltas | message_fact, edge_fact |

## Privacy Model

- **Metadata only**: No email body, subject, or attachment content accessed or stored
- **Microsoft Graph**: Uses Mail.ReadBasic.All — body access blocked at API level (returns 403)
- **Anonymization toggle**: Replaces emails with deterministic user_XXXXXX@domain aliases
- **Credentials**: Stored in .env.local (gitignored), loaded via environment variables
- **Cache**: Parquet files on local disk (not encrypted — future improvement)
- **Export**: CSV downloads respect anonymization toggle

## Community Detection (Current → Planned)

**Current**: Louvain with default resolution on undirected graph. Over-fragments (628 communities on Spokane data).

**Planned improvements**:
1. Resolution parameter tuning (0.5 instead of 1.0)
2. Pre-filter nonhuman addresses before detection
3. Merge tiny communities (<3 members)
4. Auto-label by central person + dominant domain
5. Future: Leiden algorithm with multi-resolution hierarchical view

## Target Buyers

1. **Government oversight** (Inspector General, oversight boards) — $15-50K/engagement
2. **Legal e-discovery** (law firms, compliance) — $10-50K/engagement
3. **HR/ONA consulting** (org design, M&A due diligence) — $5-15K/project

## Tech Stack

- **Data**: Polars, NetworkX, python-louvain, scipy
- **Dashboard**: Streamlit, Plotly, PyVis
- **Cache**: Parquet (mtime invalidation), pickle (graph objects)
- **Auth connector**: MSAL (Microsoft), future: google-auth
- **Deployment**: Docker, docker-compose
- **Testing**: pytest (84 tests)
