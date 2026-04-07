"""Integration smoke tests: verify all analytics modules run without crashing on synthetic data."""

import datetime as dt
import pytest
import polars as pl
import networkx as nx


# ---------------------------------------------------------------------------
# Shared synthetic test data
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, start: str = "2024-01-01") -> list[dt.datetime]:
    """Generate n timestamps spread over several months."""
    base = dt.datetime.strptime(start, "%Y-%m-%d")
    return [base + dt.timedelta(hours=i * 3) for i in range(n)]


def _make_emails(n_people: int = 10) -> list[str]:
    return [f"person{i}@example.com" for i in range(n_people)]


@pytest.fixture
def people():
    return _make_emails(10)


@pytest.fixture
def message_fact(people):
    """Synthetic message_fact with 200 messages across 10 people over ~25 days."""
    import random
    random.seed(42)
    timestamps = _make_timestamps(200)
    records = []
    for i, ts in enumerate(timestamps):
        sender = random.choice(people)
        n_recip = random.randint(1, 3)
        recipients = random.sample([p for p in people if p != sender], min(n_recip, 9))
        records.append({
            "msg_id": i,
            "timestamp": ts,
            "size_bytes": random.randint(500, 50000),
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


@pytest.fixture
def edge_fact(message_fact):
    """Exploded edge_fact from message_fact."""
    return message_fact.select([
        "msg_id", "timestamp", "size_bytes", "from_email",
        "to_emails", "n_recipients", "week_id", "hour",
        "day_of_week", "is_after_hours", "is_weekend",
    ]).explode("to_emails").rename({"to_emails": "to_email"})


@pytest.fixture
def person_dim(people):
    """Synthetic person_dim."""
    return pl.DataFrame({
        "email": people,
        "display_name": [p.split("@")[0] for p in people],
        "total_sent": [20] * len(people),
        "total_received": [20] * len(people),
        "domain": ["example.com"] * len(people),
        "is_internal": [True] * len(people),
        "is_distribution_list": [False] * len(people),
        "department": ["Engineering"] * 5 + ["Marketing"] * 3 + ["Sales"] * 2,
    })


@pytest.fixture
def graph(edge_fact):
    from src.analytics.network import build_graph
    return build_graph(edge_fact)


@pytest.fixture
def graph_metrics(graph):
    from src.analytics.network import compute_node_metrics
    return compute_node_metrics(graph)


# ---------------------------------------------------------------------------
# Analytics module smoke tests
# ---------------------------------------------------------------------------

class TestNetworkAnalytics:
    def test_build_graph(self, graph):
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_compute_node_metrics(self, graph_metrics):
        assert len(graph_metrics) > 0
        assert "email" in graph_metrics.columns
        assert "community_id" in graph_metrics.columns
        assert "betweenness_centrality" in graph_metrics.columns
        assert "pagerank" in graph_metrics.columns

    def test_compute_dyads(self, edge_fact):
        from src.analytics.network import compute_dyads
        dyads = compute_dyads(edge_fact)
        assert len(dyads) > 0
        assert "total_pair_msgs" in dyads.columns


class TestVolumeAnalytics:
    def test_gini_coefficient(self):
        from src.analytics.volume import gini_coefficient
        import numpy as np
        assert gini_coefficient(np.array([1, 1, 1, 1])) == pytest.approx(0.0, abs=0.01)
        assert gini_coefficient(np.array([0, 0, 0, 100])) > 0.5


class TestTimingAnalytics:
    def test_heatmap_data(self, message_fact):
        from src.analytics.timing_analytics import compute_hour_day_heatmap
        heatmap = compute_hour_day_heatmap(message_fact)
        assert len(heatmap) > 0


class TestBroadcastAnalytics:
    def test_broadcast_stats(self, message_fact):
        from src.transform.broadcast import compute_broadcast_stats
        stats = compute_broadcast_stats(message_fact)
        assert len(stats) > 0


class TestResponseTime:
    def test_reply_times(self, edge_fact):
        from src.analytics.response_time import compute_reply_times
        rt = compute_reply_times(edge_fact)
        # May or may not find replies depending on data, but should not crash
        assert isinstance(rt, pl.DataFrame)

    def test_person_response_stats(self, edge_fact):
        from src.analytics.response_time import compute_reply_times, compute_person_response_stats
        rt = compute_reply_times(edge_fact)
        stats = compute_person_response_stats(rt)
        assert isinstance(stats, pl.DataFrame)


class TestHierarchy:
    def test_detect_nonhuman(self, person_dim, edge_fact):
        from src.analytics.hierarchy import detect_nonhuman_addresses
        result = detect_nonhuman_addresses(person_dim, edge_fact)
        assert "is_nonhuman" in result.columns

    def test_hierarchy_score(self, edge_fact, person_dim):
        from src.analytics.hierarchy import compute_hierarchy_score
        scores = compute_hierarchy_score(edge_fact, person_dim)
        assert len(scores) > 0

    def test_reciprocal_teams(self, edge_fact, person_dim):
        from src.analytics.hierarchy import infer_reciprocal_teams
        teams = infer_reciprocal_teams(edge_fact, person_dim, min_msgs_per_direction=1, min_team_size=2)
        assert isinstance(teams, pl.DataFrame)

    def test_classify_nonhuman_type(self):
        from src.analytics.hierarchy import classify_nonhuman_type
        assert classify_nonhuman_type("copier@example.com") == "Copier/Scanner"
        assert classify_nonhuman_type("noreply@example.com") == "Notification"
        assert classify_nonhuman_type("random@example.com") == "Other Automated"


class TestSilos:
    def test_community_interaction(self, edge_fact, graph_metrics):
        from src.analytics.silos import compute_community_interaction_matrix
        lookup = dict(zip(graph_metrics["email"].to_list(), graph_metrics["community_id"].to_list()))
        matrix = compute_community_interaction_matrix(edge_fact, lookup)
        assert isinstance(matrix, pl.DataFrame)

    def test_bridges(self, graph, graph_metrics):
        from src.analytics.silos import identify_bridges
        lookup = dict(zip(graph_metrics["email"].to_list(), graph_metrics["community_id"].to_list()))
        bridges = identify_bridges(graph, lookup)
        assert isinstance(bridges, pl.DataFrame)

    def test_simulate_removal(self, graph, people):
        from src.analytics.silos import simulate_removal
        result = simulate_removal(graph, people[0])
        assert "before_components" in result
        assert "after_components" in result


class TestTemporalNetwork:
    def test_monthly_snapshots(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        snapshots = build_monthly_snapshots(edge_fact)
        assert isinstance(snapshots, dict)
        assert len(snapshots) > 0

    def test_community_stability(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots, compute_community_stability
        snapshots = build_monthly_snapshots(edge_fact)
        stability = compute_community_stability(snapshots)
        assert isinstance(stability, pl.DataFrame)

    def test_centrality_trends(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots, compute_centrality_trends
        snapshots = build_monthly_snapshots(edge_fact)
        trends = compute_centrality_trends(snapshots)
        assert isinstance(trends, pl.DataFrame)


class TestAnomaly:
    def test_anomaly_detection(self, message_fact, edge_fact, person_dim):
        from src.analytics.anomaly import detect_volume_anomalies, detect_sender_anomalies
        from src.transform.weekly_agg import compute_weekly_stats
        wa = compute_weekly_stats(message_fact, edge_fact)
        vol_anom = detect_volume_anomalies(wa)
        assert isinstance(vol_anom, pl.DataFrame)
        sender_anom = detect_sender_anomalies(edge_fact, person_dim)
        assert isinstance(sender_anom, pl.DataFrame)


class TestNarrative:
    def test_generate_narrative(self, message_fact, edge_fact, person_dim):
        from src.analytics.narrative import generate_executive_narrative
        from src.transform.weekly_agg import compute_weekly_stats
        wa = compute_weekly_stats(message_fact, edge_fact)
        text = generate_executive_narrative(message_fact, wa, edge_fact, person_dim)
        assert isinstance(text, str)
        assert len(text) > 50


class TestHealthScore:
    def test_compute_health(self, message_fact, edge_fact, graph_metrics):
        from src.analytics.health_score import compute_health_score
        result = compute_health_score(message_fact, edge_fact, graph_metrics)
        assert "composite" in result
        assert 0 <= result["composite"] <= 100
        assert len(result["sub_scores"]) == 6

    def test_health_trend(self, message_fact, edge_fact, graph_metrics):
        from src.analytics.health_score import compute_health_trend
        trend = compute_health_trend(message_fact, edge_fact, graph_metrics)
        assert isinstance(trend, pl.DataFrame)


class TestComparison:
    def test_compute_delta(self):
        from src.analytics.comparison import compute_delta
        current = {"msg_count": 100}
        previous = {"msg_count": 80}
        result = compute_delta(current, previous)
        assert result["msg_count"]["delta"] == 20
        assert result["msg_count"]["pct"] == pytest.approx(25.0)


class TestDataQuality:
    def test_quality_metrics(self, message_fact):
        from src.analytics.data_quality import compute_quality_metrics
        q = compute_quality_metrics(message_fact)
        assert "total_messages" in q
        assert q["total_messages"] == len(message_fact)


class TestSizeForensics:
    def test_classify_by_size(self, message_fact):
        from src.analytics.size_forensics import classify_by_size
        result = classify_by_size(message_fact)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# New analytics modules (from recent features)
# ---------------------------------------------------------------------------

class TestCommunityLeiden:
    def test_detect_communities(self, graph):
        from src.analytics.community_leiden import detect_leiden_communities
        result = detect_leiden_communities(graph)
        assert len(result) > 0
        assert "community_coarse" in result.columns
        assert "community_medium" in result.columns
        assert "community_fine" in result.columns
        assert "community_type" in result.columns

    def test_compare_louvain_leiden(self, graph):
        from src.analytics.community_leiden import compare_louvain_leiden
        result = compare_louvain_leiden(graph)
        assert "louvain_communities" in result
        assert "louvain_modularity" in result

    def test_hierarchy_nesting(self, graph):
        from src.analytics.community_leiden import detect_leiden_communities, build_hierarchy_nesting
        leiden_df = detect_leiden_communities(graph)
        nesting = build_hierarchy_nesting(leiden_df)
        assert isinstance(nesting, pl.DataFrame)


class TestStructuralChange:
    def test_classify_shifts(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        from src.analytics.structural_change import classify_community_shifts
        snapshots = build_monthly_snapshots(edge_fact)
        shifts = classify_community_shifts(snapshots)
        assert isinstance(shifts, pl.DataFrame)

    def test_track_node_switches(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        from src.analytics.structural_change import track_node_switches, compute_switch_rates
        snapshots = build_monthly_snapshots(edge_fact)
        switches = track_node_switches(snapshots)
        assert isinstance(switches, pl.DataFrame)
        rates = compute_switch_rates(switches)
        assert isinstance(rates, pl.DataFrame)

    def test_community_flow(self, edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        from src.analytics.structural_change import build_community_flow
        snapshots = build_monthly_snapshots(edge_fact)
        flow = build_community_flow(snapshots)
        assert isinstance(flow, pl.DataFrame)


class TestCompliance:
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
        key_dates = [{"label": "Test Event", "date": dt.date(2024, 1, 10)}]
        result = key_date_gap_analysis(message_fact, key_dates)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_after_hours_clusters(self, edge_fact):
        from src.analytics.compliance import detect_after_hours_clusters
        result = detect_after_hours_clusters(edge_fact, min_after_hours_msgs=1, min_cluster_size=2)
        assert isinstance(result, pl.DataFrame)


class TestCascade:
    def test_detect_cascades(self, edge_fact):
        from src.analytics.cascade import detect_cascades
        result = detect_cascades(edge_fact, max_delay_minutes=120, min_chain_length=2)
        assert isinstance(result, pl.DataFrame)

    def test_cascade_metrics(self, edge_fact):
        from src.analytics.cascade import detect_cascades, compute_cascade_metrics
        cascades = detect_cascades(edge_fact, max_delay_minutes=120, min_chain_length=2)
        metrics = compute_cascade_metrics(cascades)
        assert isinstance(metrics, pl.DataFrame)

    def test_amplifiers(self, edge_fact):
        from src.analytics.cascade import detect_cascades, identify_amplifiers
        cascades = detect_cascades(edge_fact, max_delay_minutes=120, min_chain_length=2)
        amplifiers = identify_amplifiers(cascades)
        assert isinstance(amplifiers, pl.DataFrame)


class TestBusFactor:
    def test_articulation_points(self, graph):
        from src.analytics.bus_factor import find_articulation_points
        ap = find_articulation_points(graph)
        assert isinstance(ap, list)

    def test_succession_readiness(self, graph):
        from src.analytics.bus_factor import find_articulation_points, compute_succession_readiness
        ap = find_articulation_points(graph)
        result = compute_succession_readiness(graph, ap)
        assert isinstance(result, pl.DataFrame)

    def test_dependency_risk_matrix(self, graph, graph_metrics):
        from src.analytics.bus_factor import find_articulation_points, compute_dependency_risk_matrix
        ap = find_articulation_points(graph)
        result = compute_dependency_risk_matrix(graph, graph_metrics, ap)
        assert isinstance(result, pl.DataFrame)
        assert "risk_score" in result.columns


# ---------------------------------------------------------------------------
# Export module smoke tests
# ---------------------------------------------------------------------------

class TestExports:
    def test_graphml_export(self, graph, graph_metrics):
        """Verify GraphML serialization doesn't crash."""
        import io
        G_export = graph.copy()
        buf = io.BytesIO()
        nx.write_graphml(G_export, buf)
        assert buf.tell() > 0

    def test_json_export(self, graph, graph_metrics):
        """Verify JSON export produces valid output."""
        import json
        nodes = [{"id": n} for n in graph.nodes()]
        edges = [{"source": u, "target": v} for u, v in graph.edges()]
        data = {"nodes": nodes, "edges": edges}
        result = json.dumps(data)
        assert len(result) > 0

    def test_pptx_export(self, message_fact, edge_fact, person_dim, graph_metrics):
        """Verify PowerPoint generation doesn't crash."""
        from src.export_pptx import generate_pptx
        pptx_bytes = generate_pptx(
            message_fact=message_fact,
            edge_fact=edge_fact,
            person_dim=person_dim,
            graph_metrics=graph_metrics,
            start_date=dt.date(2024, 1, 1),
            end_date=dt.date(2024, 1, 25),
        )
        assert len(pptx_bytes) > 1000  # Should be a valid PPTX

    def test_html_export(self, message_fact, edge_fact, person_dim, graph_metrics):
        """Verify HTML report generation doesn't crash."""
        from src.export_html import generate_html_report
        html = generate_html_report(
            message_fact=message_fact,
            edge_fact=edge_fact,
            person_dim=person_dim,
            graph_metrics=graph_metrics,
            start_date=dt.date(2024, 1, 1),
            end_date=dt.date(2024, 1, 25),
        )
        assert "<html>" in html
        assert "Total Messages" in html


# ---------------------------------------------------------------------------
# Data profiler smoke tests
# ---------------------------------------------------------------------------

class TestProfiler:
    def test_detect_date_format(self):
        from src.ingest.profiler import detect_date_format
        samples = ["01/15/2024 14:30", "02/20/2024 09:00", "03/01/2024 16:45"]
        result = detect_date_format(samples)
        assert result is not None
        fmt, label = result
        assert fmt == "%m/%d/%Y %H:%M"

    def test_detect_date_format_iso(self):
        from src.ingest.profiler import detect_date_format
        samples = ["2024-01-15 14:30:00", "2024-02-20 09:00:00"]
        result = detect_date_format(samples)
        assert result is not None
        fmt, label = result
        assert "Y" in fmt

    def test_detect_encoding(self, tmp_path):
        from src.ingest.profiler import detect_encoding
        f = tmp_path / "test.csv"
        f.write_text("Date,Size,From,To\n01/01/2024,1K,a@b.com,c@d.com\n", encoding="utf-8")
        assert detect_encoding(f) == "utf-8"

    def test_detect_column_roles(self):
        from src.ingest.profiler import detect_column_roles
        header = ["Date", "Size", "From", "To"]
        sample = [["01/01/2024", "1K", "alice@example.com", "bob@example.com"]]
        roles = detect_column_roles(header, sample)
        assert roles["date_col"] == 0
        assert roles["size_col"] == 1
        assert roles["from_col"] == 2
        assert roles["to_col"] == 3

    def test_profile_csv(self, tmp_path):
        from src.ingest.profiler import profile_csv
        f = tmp_path / "test.csv"
        lines = [
            "Date,Size,From,To",
            "01/15/2024 14:30,10K,alice@example.com,bob@example.com",
            "01/16/2024 09:00,5K,bob@example.com,alice@example.com",
            "01/17/2024 11:30,8K,carol@example.com,alice@example.com",
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        profile = profile_csv(f)
        assert "error" not in profile
        assert profile["encoding"] == "utf-8"
        assert profile["n_data_lines"] == 3
        assert profile["date_format"] == "%m/%d/%Y %H:%M"


# ---------------------------------------------------------------------------
# Mailbox import smoke tests
# ---------------------------------------------------------------------------

class TestMailboxImport:
    def test_mbox_import(self, tmp_path):
        """Create a minimal MBOX and verify import."""
        from src.ingest.mailbox_import import import_mbox
        mbox_path = tmp_path / "test.mbox"
        # Write a minimal MBOX with one message
        mbox_content = (
            "From sender@example.com Mon Jan 15 14:30:00 2024\n"
            "Date: Mon, 15 Jan 2024 14:30:00 +0000\n"
            "From: Alice <alice@example.com>\n"
            "To: Bob <bob@example.com>\n"
            "Subject: Test\n"
            "Content-Length: 5\n"
            "\n"
            "Hello\n"
            "\n"
        )
        mbox_path.write_text(mbox_content)
        df, next_id, errors = import_mbox(mbox_path)
        assert len(df) == 1
        assert df["from_email"][0] == "alice@example.com"

    def test_detect_file_type(self, tmp_path):
        from src.ingest.mailbox_import import detect_file_type
        mbox = tmp_path / "test.mbox"
        mbox.write_text("From sender@example.com Mon Jan 1 00:00:00 2024\n")
        assert detect_file_type(mbox) == "mbox"

        pst = tmp_path / "test.pst"
        pst.write_text("")
        assert detect_file_type(pst) == "pst"


# ---------------------------------------------------------------------------
# Engagement profile tests
# ---------------------------------------------------------------------------

class TestEngagement:
    def test_default_alert_rules(self):
        from src.engagement import default_alert_rules
        rules = default_alert_rules()
        assert len(rules) == 6
        assert all("name" in r and "metric" in r and "threshold" in r for r in rules)

    def test_save_load_profile(self, tmp_path):
        from src.engagement import save_profile, load_profile, PROFILES_DIR
        import src.engagement as eng
        # Temporarily redirect profiles dir
        orig_dir = eng.PROFILES_DIR
        eng.PROFILES_DIR = tmp_path

        settings = {
            "internal_domains": ["example.com"],
            "date_format": "%Y-%m-%d %H:%M",
            "column_mapping": {"date": "Date", "size": "Size", "from": "From", "to": "To"},
            "exclude_nonhuman": True,
            "key_dates": [],
            "alert_rules": [],
            "org_name": "Test Org",
        }
        save_profile("test_client", settings)
        loaded, dept_df = load_profile("test_client")
        assert loaded["org_name"] == "Test Org"
        assert loaded["internal_domains"] == ["example.com"]
        assert dept_df is None

        eng.PROFILES_DIR = orig_dir

    def test_save_load_with_departments(self, tmp_path):
        from src.engagement import save_profile, load_profile
        import src.engagement as eng
        orig_dir = eng.PROFILES_DIR
        eng.PROFILES_DIR = tmp_path

        dept_df = pl.DataFrame({"email": ["a@b.com"], "department": ["Engineering"]})
        save_profile("test_dept", {"org_name": "Test"}, department_df=dept_df)
        loaded, loaded_dept = load_profile("test_dept")
        assert loaded_dept is not None
        assert len(loaded_dept) == 1

        eng.PROFILES_DIR = orig_dir

    def test_evaluate_alerts(self, message_fact, edge_fact, graph_metrics):
        from src.engagement import evaluate_alerts, default_alert_rules
        rules = default_alert_rules()
        alerts = evaluate_alerts(rules, message_fact, edge_fact, graph_metrics)
        assert isinstance(alerts, list)

    def test_evaluate_alerts_with_health(self, message_fact, edge_fact, graph_metrics):
        from src.engagement import evaluate_alerts, default_alert_rules
        from src.analytics.health_score import compute_health_score
        rules = default_alert_rules()
        health = compute_health_score(message_fact, edge_fact, graph_metrics)
        alerts = evaluate_alerts(
            rules, message_fact, edge_fact, graph_metrics, health_score=health,
        )
        assert isinstance(alerts, list)


class TestExecutiveMemo:
    def test_generate_memo(self, message_fact, edge_fact, person_dim, graph_metrics):
        from src.export_memo import generate_executive_memo
        memo_bytes = generate_executive_memo(
            message_fact=message_fact,
            edge_fact=edge_fact,
            person_dim=person_dim,
            graph_metrics=graph_metrics,
            start_date=dt.date(2024, 1, 1),
            end_date=dt.date(2024, 1, 25),
            org_name="Test Corp",
        )
        assert len(memo_bytes) > 1000

    def test_memo_with_alerts(self, message_fact, edge_fact, person_dim, graph_metrics):
        from src.export_memo import generate_executive_memo
        alerts = [
            {"name": "Test Alert", "severity": "critical",
             "entity": "alice@example.com", "detail": "Test detail"},
        ]
        memo_bytes = generate_executive_memo(
            message_fact=message_fact,
            edge_fact=edge_fact,
            person_dim=person_dim,
            graph_metrics=graph_metrics,
            alerts=alerts,
            org_name="Test Corp",
        )
        assert len(memo_bytes) > 1000
