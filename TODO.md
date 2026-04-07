# TODO — Email Metadata Analytics Platform

## This Week: Demo-Ready Improvements

### Community Detection Fix
- [x] Add `resolution=0.5` to Louvain calls
- [x] Add `merge_small_communities()` function
- [x] Pre-filter nonhuman addresses before community detection
- [x] Auto-label communities by central person + dominant domain

### Executive Summary Overhaul (Page 01)
- [x] Add Human vs Machine split as first KPI row (donut chart)
- [x] Add Critical Personnel Risk section (top 5 bridges)
- [x] Add response time health metric
- [x] Add external dependencies summary
- [x] Replace jargon with plain English

### Global UX
- [x] Move nonhuman filter to global sidebar toggle
- [x] Replace jargon labels on Page 07
- [x] Create Automated Systems Dashboard (Page 21)
- [x] Reorganize sidebar navigation into 5 tiers

## Next Week: New Pages + Navigation

### New Page: Automated Systems Dashboard (Page 21)
- [x] Machine vs human volume breakdown
- [x] Top automated senders with type classification
- [x] Size template fingerprinting
- [x] Hour-of-day heatmap for each automated system

### Navigation Reorganization
- [x] Group pages into 5 tiers in sidebar
- [x] Add glossary/help expander to sidebar

### Org Health Score
- [x] Composite 0-100 score from 6 dimensions
- [x] Radar chart visualization
- [x] Trend over time (compute per-month and show line chart)

## Later: New Analytics (Post First Sale)

### Compliance Pattern Detection
- [x] Communication blackout window detection
- [x] External contact spike alerts
- [x] User-supplied key dates with +/- window gap analysis
- [x] After-hours clustering (groups working late together)

### Structural Change Detection
- [x] Classify monthly community shifts as split/merge/reorg
- [x] Track individual node community switches
- [x] NMI drop alert threshold

### Information Cascade Detection
- [x] Chain A→B→C within configurable time window
- [x] Cascade depth, breadth, velocity metrics
- [x] "Amplifier" node identification

### Key-Person Dependency / Bus Factor
- [x] Articulation point detection (NetworkX built-in)
- [x] Per-team bus factor simulation
- [x] Succession readiness (overlap of contact networks)

### Community Detection v2 (Leiden)
- [x] Switch to Leiden algorithm (`leidenalg` package)
- [x] Multi-resolution detection (coarse/medium/fine)
- [x] Community type classification (department/team/pair/external)
- [x] Hierarchical nesting view

### Additional Features
- [x] PST/MBOX file import
- [ ] Google Workspace connector (Gmail API)
- [x] Person comparison page (side-by-side behavioral metrics)
- [x] Department enrichment (optional CSV upload of email→department mapping)
- [x] PowerPoint export
- [x] GraphML/JSON network export for legal exhibits
