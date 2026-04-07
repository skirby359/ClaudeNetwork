"""Page-level smoke tests: exercise every analytics path each page calls.

Uses an extended 500-message, 15-person, 90-day synthetic dataset
to cover temporal pages (16, 24) that need 2+ months of data.
"""

import datetime as dt
import random

import pytest
import polars as pl
import networkx as nx


# ---------------------------------------------------------------------------
# Extended fixtures — 500 msgs, 15 people, 90 days, 5 external
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def people():
    internal = [f"person{i}@example.com" for i in range(10)]
    external = [f"ext{i}@partner.org" for i in range(5)]
    return internal + external


@pytest.fixture(scope="module")
def internal_emails():
    return {f"person{i}@example.com" for i in range(10)}


@pytest.fixture(scope="module")
def message_fact(people):
    random.seed(42)
    # 500 messages over 90 days (Jan 1 - Mar 31 2024)
    timestamps = [dt.datetime(2024, 1, 1) + dt.timedelta(hours=i * 4.3) for i in range(500)]
    records = []
    for i, ts in enumerate(timestamps):
        sender = random.choice(people)
        n_recip = random.randint(1, 4)
        recipients = random.sample([p for p in people if p != sender], min(n_recip, 14))
        records.append({
            "msg_id": i,
            "timestamp": ts,
            "size_bytes": random.choice([500, 1000, 5000, 10000, 50000, 100000]),
            "from_email": sender,
            "from_name": sender.split("@")[0],
            "to_emails": recipients,
            "to_names": [r.split("@")[0] for r in recipients],
            "n_recipients": len(recipients),
            "week_id": ts.strftime("%G-W%V"),
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "is_after_hours": ts.hour >= 18 or ts.hour < 7,
            "is_weekend": ts.weekday() >= 5,
        })
    return pl.DataFrame(records)


@pytest.fixture(scope="module")
def edge_fact(message_fact):
    return message_fact.select([
        "msg_id", "timestamp", "size_bytes", "from_email",
        "to_emails", "n_recipients", "week_id", "hour",
        "day_of_week", "is_after_hours", "is_weekend",
    ]).explode("to_emails").rename({"to_emails": "to_email"})


@pytest.fixture(scope="module")
def person_dim(people):
    n = len(people)
    return pl.DataFrame({
        "email": people,
        "display_name": [p.split("@")[0] for p in people],
        "total_sent": [30] * n,
        "total_received": [30] * n,
        "domain": [p.split("@")[1] for p in people],
        "is_internal": [i < 10 for i in range(n)],
        "is_distribution_list": [False] * n,
        "department": (["Engineering"] * 4 + ["Marketing"] * 3 + ["Sales"] * 3
                       + ["Partner"] * 5),
    })


@pytest.fixture(scope="module")
def graph(edge_fact):
    from src.analytics.network import build_graph
    return build_graph(edge_fact)


@pytest.fixture(scope="module")
def graph_metrics(graph):
    from src.analytics.network import compute_node_metrics
    return compute_node_metrics(graph)


@pytest.fixture(scope="module")
def weekly_agg(message_fact, edge_fact):
    from src.transform.weekly_agg import compute_weekly_stats
    return compute_weekly_stats(message_fact, edge_fact)


@pytest.fixture(scope="module")
def nonhuman_emails(person_dim, edge_fact):
    from src.analytics.hierarchy import detect_nonhuman_addresses
    flagged = detect_nonhuman_addresses(person_dim, edge_fact)
    return frozenset(flagged.filter(pl.col("is_nonhuman"))["email"].to_list())


# ---------------------------------------------------------------------------
# Page 01: Executive Summary
# ---------------------------------------------------------------------------
class TestPage01:
    def test_period_summary(self, message_fact, edge_fact):
        from src.analytics.comparison import compute_period_summary
        result = compute_period_summary(message_fact, edge_fact)
        assert isinstance(result, dict)
        assert "total_edges" in result

    def test_sender_concentration(self, edge_fact):
        from src.analytics.volume import compute_sender_concentration
        result = compute_sender_concentration(edge_fact)
        assert "gini" in result

    def test_narrative(self, message_fact, weekly_agg, edge_fact, person_dim):
        from src.analytics.narrative import generate_executive_narrative
        text = generate_executive_narrative(message_fact, weekly_agg, edge_fact, person_dim)
        assert len(text) > 20

    def test_bridges(self, edge_fact, graph_metrics):
        from src.analytics.silos import identify_bridges
        from src.analytics.network import build_graph
        G = build_graph(edge_fact)
        lookup = dict(zip(graph_metrics["email"].to_list(), graph_metrics["community_id"].to_list()))
        bridges = identify_bridges(G, lookup)
        assert isinstance(bridges, pl.DataFrame)

    def test_reply_times(self, edge_fact):
        from src.analytics.response_time import compute_reply_times
        rt = compute_reply_times(edge_fact)
        assert isinstance(rt, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 02: Volume & Seasonality
# ---------------------------------------------------------------------------
class TestPage02:
    def test_weekly_stats(self, weekly_agg):
        assert len(weekly_agg) > 0
        assert "msg_count" in weekly_agg.columns


# ---------------------------------------------------------------------------
# Page 03: Time Norms
# ---------------------------------------------------------------------------
class TestPage03:
    def test_heatmap(self, message_fact):
        from src.analytics.timing_analytics import compute_hour_day_heatmap
        result = compute_hour_day_heatmap(message_fact)
        assert len(result) > 0

    def test_after_hours_trend(self, message_fact):
        from src.analytics.timing_analytics import compute_after_hours_by_week
        result = compute_after_hours_by_week(message_fact)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Page 04: Broadcast & Attention
# ---------------------------------------------------------------------------
class TestPage04:
    def test_broadcast_stats(self, message_fact):
        from src.transform.broadcast import compute_broadcast_stats
        result = compute_broadcast_stats(message_fact)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 05: Artifact vs Ping
# ---------------------------------------------------------------------------
class TestPage05:
    def test_size_classification(self, message_fact):
        from src.analytics.size_forensics import classify_by_size
        result = classify_by_size(message_fact)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Page 06: Network Map
# ---------------------------------------------------------------------------
class TestPage06:
    def test_graph_construction(self, graph, graph_metrics):
        assert graph.number_of_nodes() > 0
        assert len(graph_metrics) > 0


# ---------------------------------------------------------------------------
# Page 07: Bottlenecks & Routing
# ---------------------------------------------------------------------------
class TestPage07:
    def test_betweenness_ranking(self, graph_metrics):
        top = graph_metrics.sort("betweenness_centrality", descending=True).head(10)
        assert len(top) > 0


# ---------------------------------------------------------------------------
# Page 08: Dyads & Asymmetry
# ---------------------------------------------------------------------------
class TestPage08:
    def test_dyads(self, edge_fact):
        from src.analytics.network import compute_dyads
        dyads = compute_dyads(edge_fact)
        assert len(dyads) > 0


# ---------------------------------------------------------------------------
# Page 09: Coordination & Churn
# ---------------------------------------------------------------------------
class TestPage09:
    def test_community_sizes(self, graph_metrics):
        comm_counts = graph_metrics.group_by("community_id").agg(pl.len().alias("n"))
        assert len(comm_counts) > 0


# ---------------------------------------------------------------------------
# Page 10: Risk Register
# ---------------------------------------------------------------------------
class TestPage10:
    def test_volume_anomalies(self, weekly_agg):
        from src.analytics.anomaly import detect_volume_anomalies
        result = detect_volume_anomalies(weekly_agg)
        assert isinstance(result, pl.DataFrame)

    def test_sender_anomalies(self, edge_fact, person_dim):
        from src.analytics.anomaly import detect_sender_anomalies
        result = detect_sender_anomalies(edge_fact, person_dim)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 11: External Contacts
# ---------------------------------------------------------------------------
class TestPage11:
    def test_external_filtering(self, edge_fact, internal_emails):
        external_edges = edge_fact.filter(
            ~pl.col("to_email").is_in(list(internal_emails))
        )
        assert len(external_edges) > 0


# ---------------------------------------------------------------------------
# Page 13: Response Time
# ---------------------------------------------------------------------------
class TestPage13:
    def test_person_response_stats(self, edge_fact):
        from src.analytics.response_time import compute_reply_times, compute_person_response_stats
        rt = compute_reply_times(edge_fact)
        stats = compute_person_response_stats(rt)
        assert isinstance(stats, pl.DataFrame)

    def test_dept_response_stats(self, edge_fact, person_dim):
        from src.analytics.response_time import compute_reply_times, compute_department_response_stats
        rt = compute_reply_times(edge_fact)
        stats = compute_department_response_stats(rt, person_dim)
        assert isinstance(stats, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 14: Hierarchy Inference
# ---------------------------------------------------------------------------
class TestPage14:
    def test_hierarchy_score(self, edge_fact, person_dim):
        from src.analytics.hierarchy import compute_hierarchy_score
        scores = compute_hierarchy_score(edge_fact, person_dim)
        assert len(scores) > 0

    def test_reciprocal_teams(self, edge_fact, person_dim):
        from src.analytics.hierarchy import infer_reciprocal_teams
        teams = infer_reciprocal_teams(edge_fact, person_dim, min_msgs_per_direction=1, min_team_size=2)
        assert isinstance(teams, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 15: Silos & Bridges
# ---------------------------------------------------------------------------
class TestPage15:
    def test_community_interaction_matrix(self, edge_fact, graph_metrics):
        from src.analytics.silos import compute_community_interaction_matrix
        lookup = dict(zip(graph_metrics["email"].to_list(), graph_metrics["community_id"].to_list()))
        matrix = compute_community_interaction_matrix(edge_fact, lookup)
        assert isinstance(matrix, pl.DataFrame)

    def test_simulate_removal(self, graph, people):
        from src.analytics.silos import simulate_removal
        result = simulate_removal(graph, people[0])
        assert "component_increase" in result


# ---------------------------------------------------------------------------
# Page 16: Temporal Evolution (needs 2+ months)
# ---------------------------------------------------------------------------
class TestPage16:
    def test_monthly_snapshots(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        snapshots = build_monthly_snapshots(edge_fact)
        assert len(snapshots) >= 2  # 90 days = ~3 months

    def test_centrality_trends(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots, compute_centrality_trends
        snapshots = build_monthly_snapshots(edge_fact)
        trends = compute_centrality_trends(snapshots)
        assert len(trends) > 0

    def test_community_stability(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots, compute_community_stability
        snapshots = build_monthly_snapshots(edge_fact)
        stability = compute_community_stability(snapshots)
        assert len(stability) >= 1  # At least one month pair

    def test_rising_fading(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots, compute_centrality_trends, detect_rising_fading
        snapshots = build_monthly_snapshots(edge_fact)
        trends = compute_centrality_trends(snapshots)
        rising, fading = detect_rising_fading(trends, min_months=2)
        assert isinstance(rising, pl.DataFrame)
        assert isinstance(fading, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 17: Size Forensics
# ---------------------------------------------------------------------------
class TestPage17:
    def test_size_templates(self, message_fact):
        from src.analytics.size_forensics import detect_size_templates
        result = detect_size_templates(message_fact)
        assert isinstance(result, pl.DataFrame)

    def test_sender_size_profile(self, message_fact):
        from src.analytics.size_forensics import compute_sender_size_profile
        result = compute_sender_size_profile(message_fact)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 18: Data Quality
# ---------------------------------------------------------------------------
class TestPage18:
    def test_quality_metrics(self, message_fact):
        from src.analytics.data_quality import compute_quality_metrics
        q = compute_quality_metrics(message_fact)
        assert q["total_messages"] == 500


# ---------------------------------------------------------------------------
# Page 19: Narrative
# ---------------------------------------------------------------------------
class TestPage19:
    def test_narrative(self, message_fact, weekly_agg, edge_fact, person_dim):
        from src.analytics.narrative import generate_executive_narrative
        text = generate_executive_narrative(message_fact, weekly_agg, edge_fact, person_dim)
        assert "Volume trend" in text or "Work patterns" in text


# ---------------------------------------------------------------------------
# Page 20: Period Comparison
# ---------------------------------------------------------------------------
class TestPage20:
    def test_period_comparison(self, message_fact, edge_fact):
        from src.analytics.comparison import compute_period_summary, compute_delta
        current = compute_period_summary(message_fact, edge_fact)
        previous = compute_period_summary(message_fact, edge_fact)
        delta = compute_delta(current, previous)
        assert isinstance(delta, dict)


# ---------------------------------------------------------------------------
# Page 21: Automated Systems
# ---------------------------------------------------------------------------
class TestPage21:
    def test_nonhuman_detection(self, person_dim, edge_fact):
        from src.analytics.hierarchy import detect_nonhuman_addresses, classify_nonhuman_type
        result = detect_nonhuman_addresses(person_dim, edge_fact)
        assert "is_nonhuman" in result.columns
        # Type classification
        assert classify_nonhuman_type("copier@example.com") == "Copier/Scanner"


# ---------------------------------------------------------------------------
# Page 22: Health Score
# ---------------------------------------------------------------------------
class TestPage22:
    def test_health_score(self, message_fact, edge_fact, graph_metrics):
        from src.analytics.health_score import compute_health_score
        result = compute_health_score(message_fact, edge_fact, graph_metrics)
        assert 0 <= result["composite"] <= 100

    def test_health_trend(self, message_fact, edge_fact, graph_metrics):
        from src.analytics.health_score import compute_health_trend
        trend = compute_health_trend(message_fact, edge_fact, graph_metrics)
        assert len(trend) >= 2  # 90 days = 3 months


# ---------------------------------------------------------------------------
# Page 23: Community Detection v2 (Leiden)
# ---------------------------------------------------------------------------
class TestPage23:
    def test_leiden_communities(self, graph):
        from src.analytics.community_leiden import detect_leiden_communities
        result = detect_leiden_communities(graph)
        assert len(result) > 0
        for col in ["community_coarse", "community_medium", "community_fine", "community_type"]:
            assert col in result.columns

    def test_hierarchy_nesting(self, graph):
        from src.analytics.community_leiden import detect_leiden_communities, build_hierarchy_nesting
        leiden = detect_leiden_communities(graph)
        nesting = build_hierarchy_nesting(leiden)
        assert isinstance(nesting, pl.DataFrame)

    def test_algorithm_comparison(self, graph):
        from src.analytics.community_leiden import compare_louvain_leiden
        result = compare_louvain_leiden(graph)
        assert "louvain_communities" in result


# ---------------------------------------------------------------------------
# Page 24: Structural Change
# ---------------------------------------------------------------------------
class TestPage24:
    def test_classify_shifts(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        from src.analytics.structural_change import classify_community_shifts
        snapshots = build_monthly_snapshots(edge_fact)
        shifts = classify_community_shifts(snapshots)
        assert len(shifts) >= 1

    def test_node_switches(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        from src.analytics.structural_change import track_node_switches, compute_switch_rates
        snapshots = build_monthly_snapshots(edge_fact)
        switches = track_node_switches(snapshots)
        rates = compute_switch_rates(switches)
        assert isinstance(rates, pl.DataFrame)

    def test_community_flow(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        from src.analytics.structural_change import build_community_flow
        snapshots = build_monthly_snapshots(edge_fact)
        flow = build_community_flow(snapshots)
        assert isinstance(flow, pl.DataFrame)

    def test_nmi_alerts(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots, compute_community_stability
        from src.analytics.structural_change import nmi_drop_alerts
        snapshots = build_monthly_snapshots(edge_fact)
        stability = compute_community_stability(snapshots)
        alerts = nmi_drop_alerts(stability)
        assert isinstance(alerts, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 25: Compliance Patterns
# ---------------------------------------------------------------------------
class TestPage25:
    def test_blackout_windows(self, message_fact):
        from src.analytics.compliance import detect_blackout_windows
        result = detect_blackout_windows(message_fact, min_gap_hours=24, min_historical_volume=1)
        assert isinstance(result, pl.DataFrame)

    def test_external_spikes(self, edge_fact, person_dim):
        from src.analytics.compliance import detect_external_spikes
        result = detect_external_spikes(edge_fact, person_dim)
        assert isinstance(result, pl.DataFrame)

    def test_key_date_analysis(self, message_fact):
        from src.analytics.compliance import key_date_gap_analysis
        key_dates = [
            {"label": "Budget Review", "date": dt.date(2024, 2, 1)},
            {"label": "Reorg", "date": dt.date(2024, 3, 1)},
        ]
        result = key_date_gap_analysis(message_fact, key_dates)
        assert len(result) == 2
        assert "volume_change_pct" in result.columns

    def test_after_hours_clusters(self, edge_fact):
        from src.analytics.compliance import detect_after_hours_clusters
        result = detect_after_hours_clusters(edge_fact, min_after_hours_msgs=1, min_cluster_size=2)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 26: Information Cascades
# ---------------------------------------------------------------------------
class TestPage26:
    def test_cascades(self, edge_fact):
        from src.analytics.cascade import detect_cascades, compute_cascade_metrics, identify_amplifiers
        cascades = detect_cascades(edge_fact, max_delay_minutes=120, min_chain_length=2)
        metrics = compute_cascade_metrics(cascades)
        amplifiers = identify_amplifiers(cascades)
        assert isinstance(metrics, pl.DataFrame)
        assert isinstance(amplifiers, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 27: Bus Factor
# ---------------------------------------------------------------------------
class TestPage27:
    def test_articulation_points(self, graph):
        from src.analytics.bus_factor import find_articulation_points
        ap = find_articulation_points(graph)
        assert isinstance(ap, list)

    def test_succession_readiness(self, graph):
        from src.analytics.bus_factor import find_articulation_points, compute_succession_readiness
        ap = find_articulation_points(graph)
        result = compute_succession_readiness(graph, ap)
        assert isinstance(result, pl.DataFrame)

    def test_risk_matrix(self, graph, graph_metrics):
        from src.analytics.bus_factor import find_articulation_points, compute_dependency_risk_matrix
        ap = find_articulation_points(graph)
        result = compute_dependency_risk_matrix(graph, graph_metrics, ap)
        assert "risk_score" in result.columns

    def test_team_bus_factor(self, edge_fact, graph, person_dim):
        from src.analytics.hierarchy import infer_reciprocal_teams
        from src.analytics.bus_factor import compute_team_bus_factor
        teams = infer_reciprocal_teams(edge_fact, person_dim, min_msgs_per_direction=1, min_team_size=2)
        result = compute_team_bus_factor(teams, graph)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 28: Person Comparison
# ---------------------------------------------------------------------------
class TestPage28:
    def test_person_metrics(self, edge_fact, message_fact, graph_metrics, person_dim, people):
        """Simulate the _person_metrics function from page 28."""
        email = people[0]
        sent = edge_fact.filter(pl.col("from_email") == email)
        received = edge_fact.filter(pl.col("to_email") == email)
        sent_msgs = message_fact.filter(pl.col("from_email") == email)

        n_sent = len(sent)
        n_received = len(received)
        assert n_sent >= 0
        assert n_received >= 0

        node = graph_metrics.filter(pl.col("email") == email)
        if len(node) > 0:
            assert "betweenness_centrality" in node.columns
            assert "pagerank" in node.columns

    def test_two_person_comparison(self, people, edge_fact, message_fact, graph_metrics):
        """Verify two different people can be compared without crash."""
        for email in people[:2]:
            sent = edge_fact.filter(pl.col("from_email") == email)
            node = graph_metrics.filter(pl.col("email") == email)
            # Just verify no crash
            assert isinstance(sent, pl.DataFrame)
            assert isinstance(node, pl.DataFrame)


# ---------------------------------------------------------------------------
# Page 29: Alert Dashboard
# ---------------------------------------------------------------------------
class TestPage29:
    def test_alert_evaluation(self, message_fact, edge_fact, graph_metrics):
        from src.engagement import evaluate_alerts, default_alert_rules
        from src.analytics.health_score import compute_health_score
        from src.analytics.compliance import detect_blackout_windows

        rules = default_alert_rules()
        health = compute_health_score(message_fact, edge_fact, graph_metrics)
        blackouts = detect_blackout_windows(message_fact, min_gap_hours=24, min_historical_volume=1)

        alerts = evaluate_alerts(
            rules, message_fact, edge_fact, graph_metrics,
            health_score=health, blackouts=blackouts,
        )
        assert isinstance(alerts, list)

    def test_alerts_with_bus_factor(self, message_fact, edge_fact, graph_metrics, graph, person_dim):
        from src.engagement import evaluate_alerts, default_alert_rules
        from src.analytics.hierarchy import infer_reciprocal_teams
        from src.analytics.bus_factor import compute_team_bus_factor

        teams = infer_reciprocal_teams(edge_fact, person_dim, min_msgs_per_direction=1, min_team_size=2)
        team_bf = compute_team_bus_factor(teams, graph)

        rules = default_alert_rules()
        alerts = evaluate_alerts(
            rules, message_fact, edge_fact, graph_metrics,
            team_bus_factor=team_bf,
        )
        assert isinstance(alerts, list)


# ---------------------------------------------------------------------------
# Export integration: all formats work with this dataset
# ---------------------------------------------------------------------------
class TestExportIntegration:
    def test_pptx_with_full_data(self, message_fact, edge_fact, person_dim, graph_metrics):
        from src.export_pptx import generate_pptx
        from src.analytics.health_score import compute_health_score
        health = compute_health_score(message_fact, edge_fact, graph_metrics)
        pptx_bytes = generate_pptx(
            message_fact=message_fact, edge_fact=edge_fact,
            person_dim=person_dim, graph_metrics=graph_metrics,
            health_score=health,
            start_date=dt.date(2024, 1, 1), end_date=dt.date(2024, 3, 31),
        )
        assert len(pptx_bytes) > 1000

    def test_html_with_full_data(self, message_fact, edge_fact, person_dim, graph_metrics, weekly_agg):
        from src.export_html import generate_html_report
        from src.analytics.health_score import compute_health_score
        from src.analytics.narrative import generate_executive_narrative
        health = compute_health_score(message_fact, edge_fact, graph_metrics)
        narrative = generate_executive_narrative(message_fact, weekly_agg, edge_fact, person_dim)
        html = generate_html_report(
            message_fact=message_fact, edge_fact=edge_fact,
            person_dim=person_dim, graph_metrics=graph_metrics,
            health_score=health, narrative=narrative,
            start_date=dt.date(2024, 1, 1), end_date=dt.date(2024, 3, 31),
        )
        assert "<html>" in html
        assert len(html) > 5000

    def test_memo_with_alerts(self, message_fact, edge_fact, person_dim, graph_metrics):
        from src.export_memo import generate_executive_memo
        from src.analytics.health_score import compute_health_score
        from src.engagement import evaluate_alerts, default_alert_rules
        health = compute_health_score(message_fact, edge_fact, graph_metrics)
        alerts = evaluate_alerts(default_alert_rules(), message_fact, edge_fact, graph_metrics, health_score=health)
        memo_bytes = generate_executive_memo(
            message_fact=message_fact, edge_fact=edge_fact,
            person_dim=person_dim, graph_metrics=graph_metrics,
            health_score=health, alerts=alerts,
            start_date=dt.date(2024, 1, 1), end_date=dt.date(2024, 3, 31),
            org_name="Test Corp",
        )
        assert len(memo_bytes) > 1000

    def test_csv_export_no_nested(self, message_fact, edge_fact, person_dim):
        """Regression: message_fact must drop list columns before CSV."""
        exportable = message_fact.drop(["to_emails", "to_names"])
        csv_bytes = exportable.write_csv().encode("utf-8")
        assert len(csv_bytes) > 0
        # edge_fact and person_dim should export cleanly
        edge_fact.write_csv()
        person_dim.write_csv()
