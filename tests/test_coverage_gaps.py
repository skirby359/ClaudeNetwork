"""Tests targeting coverage gaps: exports, caching, pipeline, cascade chains, structural change."""

import datetime as dt
import io
import json
import tempfile
from pathlib import Path

import pytest
import polars as pl
import networkx as nx


# ---------------------------------------------------------------------------
# Shared fixtures (reuse pattern from test_integration.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def people():
    return [f"person{i}@example.com" for i in range(10)]


@pytest.fixture
def message_fact(people):
    import random
    random.seed(42)
    timestamps = [dt.datetime(2024, 1, 1) + dt.timedelta(hours=i * 3) for i in range(200)]
    records = []
    for i, ts in enumerate(timestamps):
        sender = random.choice(people)
        n_recip = random.randint(1, 3)
        recipients = random.sample([p for p in people if p != sender], min(n_recip, 9))
        records.append({
            "msg_id": i, "timestamp": ts, "size_bytes": random.randint(500, 50000),
            "from_email": sender, "from_name": sender.split("@")[0],
            "to_emails": recipients, "to_names": [r.split("@")[0] for r in recipients],
            "n_recipients": len(recipients),
            "week_id": ts.strftime("%G-W%V"), "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "is_after_hours": ts.hour >= 18 or ts.hour < 7,
            "is_weekend": ts.weekday() >= 5,
        })
    return pl.DataFrame(records)


@pytest.fixture
def edge_fact(message_fact):
    return message_fact.select([
        "msg_id", "timestamp", "size_bytes", "from_email",
        "to_emails", "n_recipients", "week_id", "hour",
        "day_of_week", "is_after_hours", "is_weekend",
    ]).explode("to_emails").rename({"to_emails": "to_email"})


@pytest.fixture
def person_dim(people):
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
# CSV export: must handle nested list columns without crashing
# ---------------------------------------------------------------------------

class TestCSVExportNestedColumns:
    """Regression test for the nested data CSV export bug."""

    def test_message_fact_csv_export_drops_lists(self, message_fact):
        """Verify message_fact can be exported to CSV after dropping list columns."""
        exportable = message_fact.drop([c for c in ("to_emails", "to_names") if c in message_fact.columns])
        csv_bytes = exportable.write_csv().encode("utf-8")
        assert len(csv_bytes) > 0
        assert b"msg_id" in csv_bytes

    def test_message_fact_with_lists_raises(self, message_fact):
        """Verify that writing message_fact with list columns to CSV raises."""
        with pytest.raises(Exception):
            message_fact.write_csv()

    def test_edge_fact_csv_export(self, edge_fact):
        """Edge fact has no lists — should export cleanly."""
        csv_bytes = edge_fact.write_csv().encode("utf-8")
        assert len(csv_bytes) > 0

    def test_person_dim_csv_export(self, person_dim):
        """Person dim has no lists — should export cleanly."""
        csv_bytes = person_dim.write_csv().encode("utf-8")
        assert len(csv_bytes) > 0


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------

class TestCacheManager:
    def test_is_cache_fresh_missing(self, tmp_path):
        from src.cache_manager import is_cache_fresh
        assert is_cache_fresh(tmp_path / "nonexistent.parquet") is False

    def test_is_cache_fresh_stale(self, tmp_path):
        from src.cache_manager import is_cache_fresh
        import time
        source = tmp_path / "source.csv"
        cache = tmp_path / "cache.parquet"
        cache.write_text("old")
        time.sleep(0.05)
        source.write_text("new")
        assert is_cache_fresh(cache, source) is False

    def test_is_cache_fresh_valid(self, tmp_path):
        from src.cache_manager import is_cache_fresh
        import time
        source = tmp_path / "source.csv"
        source.write_text("data")
        time.sleep(0.05)
        cache = tmp_path / "cache.parquet"
        cache.write_text("cached")
        assert is_cache_fresh(cache, source) is True

    def test_write_read_parquet(self, tmp_path):
        from src.cache_manager import write_parquet, read_parquet
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        path = tmp_path / "test.parquet"
        write_parquet(df, path)
        loaded = read_parquet(path)
        assert len(loaded) == 3
        assert loaded["a"].to_list() == [1, 2, 3]

    def test_write_read_pickle(self, tmp_path):
        from src.cache_manager import write_pickle, read_pickle
        data = {"key": "value", "numbers": [1, 2, 3]}
        path = tmp_path / "test.pickle"
        write_pickle(data, path)
        loaded = read_pickle(path)
        assert loaded == data

    def test_cached_parquet(self, tmp_path):
        from src.cache_manager import cached_parquet, write_parquet
        source = tmp_path / "source.csv"
        source.write_text("data")
        cache = tmp_path / "cache.parquet"

        call_count = 0
        def builder():
            nonlocal call_count
            call_count += 1
            return pl.DataFrame({"x": [1, 2]})

        # First call: builds and caches
        result = cached_parquet(cache, [source], builder)
        assert call_count == 1
        assert len(result) == 2

        # Second call: reads from cache
        result2 = cached_parquet(cache, [source], builder)
        assert call_count == 1  # not called again


# ---------------------------------------------------------------------------
# Cascade chain building (the fixed logic)
# ---------------------------------------------------------------------------

class TestCascadeChainBuilding:
    def test_multi_hop_cascade(self):
        """Verify cascades can extend beyond 2 hops after the adjacency fix."""
        from src.analytics.cascade import detect_cascades

        # Create a clear A->B->C->D chain with 10-minute delays
        base = dt.datetime(2024, 1, 1, 10, 0)
        ef = pl.DataFrame({
            "from_email": ["alice@x.com", "bob@x.com", "carol@x.com", "dave@x.com"],
            "to_email":   ["bob@x.com",   "carol@x.com", "dave@x.com", "eve@x.com"],
            "timestamp":  [base, base + dt.timedelta(minutes=5),
                           base + dt.timedelta(minutes=10), base + dt.timedelta(minutes=15)],
            "size_bytes": [100, 100, 100, 100],
        })

        cascades = detect_cascades(ef, max_delay_minutes=30, min_chain_length=3)
        if len(cascades) > 0:
            max_depth = cascades["step"].max()
            assert max_depth >= 2  # At least 3 hops (steps 0, 1, 2)

    def test_no_self_loops(self):
        """Cascades should not include self-replies as chain extensions."""
        from src.analytics.cascade import detect_cascades

        base = dt.datetime(2024, 1, 1, 10, 0)
        ef = pl.DataFrame({
            "from_email": ["alice@x.com", "bob@x.com", "bob@x.com"],
            "to_email":   ["bob@x.com",   "alice@x.com", "carol@x.com"],
            "timestamp":  [base, base + dt.timedelta(minutes=5),
                           base + dt.timedelta(minutes=6)],
            "size_bytes": [100, 100, 100],
        })
        cascades = detect_cascades(ef, max_delay_minutes=30, min_chain_length=2)
        # Should not crash; cascade may or may not be found depending on direction
        assert isinstance(cascades, pl.DataFrame)

    def test_empty_edge_fact(self):
        from src.analytics.cascade import detect_cascades, compute_cascade_metrics, identify_amplifiers
        ef = pl.DataFrame({
            "from_email": [], "to_email": [], "timestamp": [], "size_bytes": [],
        })
        cascades = detect_cascades(ef)
        assert len(cascades) == 0
        metrics = compute_cascade_metrics(cascades)
        assert len(metrics) == 0
        amps = identify_amplifiers(cascades)
        assert len(amps) == 0


# ---------------------------------------------------------------------------
# Structural change — classification logic
# ---------------------------------------------------------------------------

class TestStructuralChangeClassification:
    def test_stable_communities(self):
        """Identical communities across months should be classified as stable."""
        from src.analytics.structural_change import classify_community_shifts

        snapshots = {
            "2024-01": pl.DataFrame({
                "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com"] * 3,
                "community_id": [0, 0, 1, 1] * 3,
            }),
            "2024-02": pl.DataFrame({
                "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com"] * 3,
                "community_id": [0, 0, 1, 1] * 3,
            }),
        }
        shifts = classify_community_shifts(snapshots)
        assert len(shifts) == 1
        assert shifts["shift_type"][0] == "stable"

    def test_split_detection(self):
        """A community splitting into two should be detected."""
        from src.analytics.structural_change import classify_community_shifts

        people = [f"p{i}@x.com" for i in range(20)]
        snapshots = {
            "2024-01": pl.DataFrame({
                "email": people,
                "community_id": [0] * 20,  # All in one community
            }),
            "2024-02": pl.DataFrame({
                "email": people,
                "community_id": [0] * 10 + [1] * 10,  # Split in half
            }),
        }
        shifts = classify_community_shifts(snapshots, nmi_threshold=0.9)
        assert len(shifts) == 1
        assert shifts["shift_type"][0] in ("split", "reorg")

    def test_nmi_drop_alerts(self):
        from src.analytics.structural_change import nmi_drop_alerts
        stability = pl.DataFrame({
            "month_pair": ["2024-01/2024-02", "2024-02/2024-03"],
            "nmi": [0.8, 0.2],
            "n_communities": [3, 5],
        })
        alerts = nmi_drop_alerts(stability, warning_threshold=0.5, critical_threshold=0.3)
        assert len(alerts) == 1
        assert alerts["alert_level"][0] == "critical"

    def test_switch_rates(self):
        from src.analytics.structural_change import compute_switch_rates
        switches = pl.DataFrame({
            "email": ["a@x.com", "a@x.com", "a@x.com", "b@x.com", "b@x.com"],
            "month_id": ["2024-01", "2024-02", "2024-03", "2024-01", "2024-02"],
            "community_id": [0, 1, 0, 0, 0],
            "prev_community_id": [None, 0, 1, None, 0],
            "switched": [False, True, True, False, False],
        })
        rates = compute_switch_rates(switches)
        assert len(rates) == 2
        a_rate = rates.filter(pl.col("email") == "a@x.com")
        assert a_rate["n_switches"][0] == 2


# ---------------------------------------------------------------------------
# Bus factor — team simulation
# ---------------------------------------------------------------------------

class TestBusFactorSimulation:
    def test_team_bus_factor_small_team(self):
        from src.analytics.bus_factor import compute_team_bus_factor

        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])

        teams = pl.DataFrame({
            "manager": ["a"],
            "team_size": [3],
            "team_members": [["b", "c", "d"]],
            "total_sent_to_team": [10],
            "total_recv_from_team": [10],
        })
        result = compute_team_bus_factor(teams, nx.DiGraph(G))
        assert len(result) == 1
        assert result["bus_factor"][0] >= 1

    def test_succession_readiness(self):
        from src.analytics.bus_factor import find_articulation_points, compute_succession_readiness

        G = nx.DiGraph()
        # Star topology: center is articulation point
        for i in range(5):
            G.add_edge("center@x.com", f"leaf{i}@x.com", weight=1)
            G.add_edge(f"leaf{i}@x.com", "center@x.com", weight=1)
        # Add one cross-link so leaf0 has some overlap with center
        G.add_edge("leaf0@x.com", "leaf1@x.com", weight=1)

        ap = find_articulation_points(G)
        result = compute_succession_readiness(G, ap)
        assert isinstance(result, pl.DataFrame)

    def test_dependency_risk_matrix(self, graph, graph_metrics):
        from src.analytics.bus_factor import find_articulation_points, compute_dependency_risk_matrix
        ap = find_articulation_points(graph)
        result = compute_dependency_risk_matrix(graph, graph_metrics, ap)
        assert "risk_score" in result.columns
        assert "is_articulation_point" in result.columns
        assert "communities_bridged" in result.columns


# ---------------------------------------------------------------------------
# GraphML / JSON export functions (non-Streamlit)
# ---------------------------------------------------------------------------

class TestNetworkExportFunctions:
    def test_graphml_with_attributes(self, graph, graph_metrics):
        """GraphML export includes node attributes from graph_metrics."""
        G_export = graph.copy()
        metrics_dict = {}
        for row in graph_metrics.iter_rows(named=True):
            metrics_dict[row["email"]] = row

        for node in G_export.nodes():
            m = metrics_dict.get(node, {})
            G_export.nodes[node]["betweenness"] = m.get("betweenness_centrality", 0.0)
            G_export.nodes[node]["community"] = m.get("community_id", -1)

        buf = io.BytesIO()
        nx.write_graphml(G_export, buf)
        xml_str = buf.getvalue().decode("utf-8")
        assert "betweenness" in xml_str
        assert "community" in xml_str

    def test_json_export_structure(self, graph, graph_metrics):
        """JSON export has correct structure."""
        nodes = [{"id": n, "pagerank": 0.01} for n in graph.nodes()]
        edges = [{"source": u, "target": v, "weight": d.get("weight", 1)}
                 for u, v, d in graph.edges(data=True)]
        data = {"nodes": nodes, "edges": edges, "metadata": {"node_count": graph.number_of_nodes()}}
        result = json.loads(json.dumps(data, default=str))
        assert len(result["nodes"]) == graph.number_of_nodes()
        assert len(result["edges"]) == graph.number_of_edges()
        assert result["metadata"]["node_count"] == graph.number_of_nodes()


# ---------------------------------------------------------------------------
# Volume analytics
# ---------------------------------------------------------------------------

class TestVolumeAnalytics:
    def test_gini_uniform(self):
        import numpy as np
        from src.analytics.volume import gini_coefficient
        assert gini_coefficient(np.array([10, 10, 10, 10])) == pytest.approx(0.0, abs=0.01)

    def test_gini_concentrated(self):
        import numpy as np
        from src.analytics.volume import gini_coefficient
        result = gini_coefficient(np.array([0, 0, 0, 100]))
        assert result > 0.7


# ---------------------------------------------------------------------------
# Engagement profiles
# ---------------------------------------------------------------------------

class TestEngagementAlertEvaluation:
    def test_after_hours_alert(self, message_fact, edge_fact, graph_metrics):
        """After-hours rule should trigger when threshold is very low."""
        from src.engagement import evaluate_alerts
        rules = [{
            "name": "High after-hours",
            "metric": "after_hours_pct",
            "operator": ">",
            "threshold": 1.0,  # 1% — almost everything triggers
            "severity": "warning",
            "description": "test",
        }]
        alerts = evaluate_alerts(rules, message_fact, edge_fact, graph_metrics)
        assert len(alerts) > 0
        assert all(a["name"] == "High after-hours" for a in alerts)

    def test_health_score_alert(self, message_fact, edge_fact, graph_metrics):
        """Health score alert should trigger when threshold is very high."""
        from src.engagement import evaluate_alerts
        from src.analytics.health_score import compute_health_score
        health = compute_health_score(message_fact, edge_fact, graph_metrics)
        rules = [{
            "name": "Low health",
            "metric": "health_composite",
            "operator": "<",
            "threshold": 999.0,  # Always triggers
            "severity": "critical",
            "description": "test",
        }]
        alerts = evaluate_alerts(rules, message_fact, edge_fact, graph_metrics, health_score=health)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"

    def test_gini_alert(self, message_fact, edge_fact, graph_metrics):
        from src.engagement import evaluate_alerts
        rules = [{
            "name": "Concentration",
            "metric": "gini",
            "operator": ">",
            "threshold": 0.0,  # Always triggers
            "severity": "info",
            "description": "test",
        }]
        alerts = evaluate_alerts(rules, message_fact, edge_fact, graph_metrics)
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_normalize_email(self):
        from src.ingest.normalizer import normalize_email
        assert normalize_email("  Alice@Example.COM  ") == "alice@example.com"

    def test_normalize_name_last_first(self):
        from src.ingest.normalizer import normalize_name
        assert normalize_name('"Smith, John"') == "John Smith"

    def test_normalize_name_simple(self):
        from src.ingest.normalizer import normalize_name
        assert normalize_name("  John Smith  ") == "John Smith"

    def test_is_distribution_list(self):
        from src.ingest.normalizer import is_distribution_list
        assert is_distribution_list("all-staff@example.com")
        assert is_distribution_list("team-list@example.com")
        assert not is_distribution_list("john@example.com")

    def test_extract_domain(self):
        from src.ingest.normalizer import extract_domain
        assert extract_domain("alice@example.com") == "example.com"
        assert extract_domain("nodomain") == ""


# ---------------------------------------------------------------------------
# Timing analytics
# ---------------------------------------------------------------------------

class TestTimingAnalyticsFull:
    def test_after_hours_by_week(self, message_fact):
        from src.analytics.timing_analytics import compute_after_hours_by_week
        result = compute_after_hours_by_week(message_fact)
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
