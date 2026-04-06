# TODO — Email Metadata Analytics Platform

## This Week: Demo-Ready Improvements

### Community Detection Fix
- [ ] Add `resolution=0.5` to Louvain calls in `src/analytics/network.py:46` and `src/analytics/temporal_network.py:47`
- [ ] Add `merge_small_communities()` function — merge <3-member communities into nearest neighbor
- [ ] Pre-filter nonhuman addresses before community detection in `compute_node_metrics()`
- [ ] Auto-label communities by central person + dominant domain (replace "C0, C1" labels)

### Executive Summary Overhaul (Page 01)
- [ ] Add Human vs Machine split as first KPI row (donut chart + metrics)
- [ ] Add Critical Personnel Risk section (top 5 bridges with removal impact)
- [ ] Add response time health metric
- [ ] Add external dependencies summary
- [ ] Replace jargon with plain English throughout

### Global UX
- [ ] Move nonhuman filter to global sidebar (app.py) instead of per-page checkboxes
- [ ] Replace jargon labels on Page 07 ("betweenness" → "cross-group connector")

## Next Week: New Pages + Navigation

### New Page: Automated Systems Dashboard (Page 21)
- [ ] Machine vs human volume breakdown
- [ ] Top automated senders with type classification
- [ ] Size template fingerprinting
- [ ] Hour-of-day heatmap for each automated system

### Navigation Reorganization
- [ ] Group 21+ pages into 5 tiers in sidebar (Executive, Health, People, Details, Deep Dives)
- [ ] Add glossary/help expander to sidebar

## Later: New Analytics (Post First Sale)

### Org Health Score
- [ ] Composite 0-100 score from: Gini, reciprocity rate, after-hours burden, response velocity, silo permeability, single-point-of-failure risk
- [ ] Radar chart visualization
- [ ] Trend over time

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
