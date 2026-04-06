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
- [ ] Machine vs human volume breakdown
- [ ] Top automated senders with type classification
- [ ] Size template fingerprinting
- [ ] Hour-of-day heatmap for each automated system

### Navigation Reorganization
- [x] Group pages into 5 tiers in sidebar
- [x] Add glossary/help expander to sidebar

### Org Health Score
- [x] Composite 0-100 score from 6 dimensions
- [x] Radar chart visualization
- [ ] Trend over time (future: compute per-month and show line chart)

## Later: New Analytics (Post First Sale)

### Compliance Pattern Detection
- [ ] Communication blackout window detection
- [ ] External contact spike alerts
- [ ] User-supplied key dates with +/- window gap analysis
- [ ] After-hours clustering (groups working late together)

### Structural Change Detection
- [ ] Classify monthly community shifts as split/merge/reorg
- [ ] Track individual node community switches
- [ ] NMI drop alert threshold

### Information Cascade Detection
- [ ] Chain A→B→C within configurable time window
- [ ] Cascade depth, breadth, velocity metrics
- [ ] "Amplifier" node identification

### Key-Person Dependency / Bus Factor
- [ ] Articulation point detection (NetworkX built-in)
- [ ] Per-team bus factor simulation
- [ ] Succession readiness (overlap of contact networks)

### Community Detection v2 (Leiden)
- [ ] Switch to Leiden algorithm (`leidenalg` package)
- [ ] Multi-resolution detection (coarse/medium/fine)
- [ ] Community type classification (department/team/pair/external)
- [ ] Hierarchical nesting view

### Additional Features
- [ ] PST/MBOX file import
- [ ] Google Workspace connector (Gmail API)
- [ ] Person comparison page (side-by-side behavioral metrics)
- [ ] Department enrichment (optional CSV upload of email→department mapping)
- [ ] PowerPoint export
- [ ] GraphML/JSON network export for legal exhibits
