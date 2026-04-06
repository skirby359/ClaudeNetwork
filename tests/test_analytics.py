"""Tests for analytics modules: data_quality, comparison, narrative,
response_time, hierarchy, silos, temporal_network, size_forensics."""

import datetime as dt

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Fixtures: minimal DataFrames that mirror the schema used across analytics
# ---------------------------------------------------------------------------

@pytest.fixture
def message_fact():
    """Minimal message_fact table."""
    return pl.DataFrame({
        "msg_id": [f"m{i}" for i in range(10)],
        "timestamp": [
            dt.datetime(2010, 3, 1, 9, 0) + dt.timedelta(hours=i * 2)
            for i in range(10)
        ],
        "from_email": [
            "alice@spokanecounty.org", "bob@spokanecounty.org",
            "alice@spokanecounty.org", "carol@external.com",
            "alice@spokanecounty.org", "bob@spokanecounty.org",
            "carol@external.com", "alice@spokanecounty.org",
            "bob@spokanecounty.org", "alice@spokanecounty.org",
        ],
        "from_name": [
            "Alice", "Bob", "Alice", "", "Alice",
            "Bob", "Carol", "Alice", "Bob", "Alice",
        ],
        "to_emails": [
            ["bob@spokanecounty.org"],
            ["alice@spokanecounty.org"],
            ["bob@spokanecounty.org", "carol@external.com"],
            ["alice@spokanecounty.org"],
            ["bob@spokanecounty.org"],
            ["alice@spokanecounty.org", "carol@external.com"],
            ["alice@spokanecounty.org"],
            ["bob@spokanecounty.org"],
            ["alice@spokanecounty.org"],
            ["bob@spokanecounty.org"],
        ],
        "n_recipients": [1, 1, 2, 1, 1, 2, 1, 1, 1, 1],
        "size_bytes": [1024, 2048, 0, 512, 5000, 100, 5000, 3000, 2000, 1500],
        "is_after_hours": [False, False, True, False, False, True, False, False, False, True],
        "is_weekend": [False, False, False, False, False, False, True, False, False, False],
    })


@pytest.fixture
def edge_fact():
    """Minimal edge_fact table (one row per from->to pair per message)."""
    rows = []
    ts_base = dt.datetime(2010, 3, 1, 9, 0)
    pairs = [
        ("alice@spokanecounty.org", "bob@spokanecounty.org"),
        ("bob@spokanecounty.org", "alice@spokanecounty.org"),
        ("alice@spokanecounty.org", "bob@spokanecounty.org"),
        ("alice@spokanecounty.org", "carol@external.com"),
        ("carol@external.com", "alice@spokanecounty.org"),
        ("alice@spokanecounty.org", "bob@spokanecounty.org"),
        ("bob@spokanecounty.org", "alice@spokanecounty.org"),
        ("bob@spokanecounty.org", "carol@external.com"),
        ("carol@external.com", "alice@spokanecounty.org"),
        ("alice@spokanecounty.org", "bob@spokanecounty.org"),
        ("bob@spokanecounty.org", "alice@spokanecounty.org"),
        ("alice@spokanecounty.org", "bob@spokanecounty.org"),
    ]
    for i, (f, t) in enumerate(pairs):
        rows.append({
            "from_email": f, "to_email": t,
            "timestamp": ts_base + dt.timedelta(hours=i),
            "size_bytes": 1024 + i * 100,
            "is_after_hours": i % 3 == 0,
            "is_weekend": i == 6,
        })
    return pl.DataFrame(rows)


@pytest.fixture
def person_dim():
    """Minimal person_dim table."""
    return pl.DataFrame({
        "email": [
            "alice@spokanecounty.org",
            "bob@spokanecounty.org",
            "carol@external.com",
        ],
        "display_name": ["Alice", "Bob", "Carol"],
        "domain": ["spokanecounty.org", "spokanecounty.org", "external.com"],
        "is_internal": [True, True, False],
        "total_sent": [5, 3, 2],
        "total_received": [4, 5, 1],
    })


# ===================================================================
# data_quality
# ===================================================================

class TestDataQuality:
    def test_compute_quality_metrics(self, message_fact):
        from src.analytics.data_quality import compute_quality_metrics
        q = compute_quality_metrics(message_fact)
        assert q["total_messages"] == 10
        assert q["zero_size_count"] == 1  # one message has size_bytes=0
        assert q["missing_name_count"] == 1  # carol has empty name
        assert q["zero_size_pct"] == pytest.approx(0.1)

    def test_compute_quality_metrics_empty(self):
        from src.analytics.data_quality import compute_quality_metrics
        empty = pl.DataFrame({
            "msg_id": [], "size_bytes": [], "from_name": [], "n_recipients": [],
        })
        q = compute_quality_metrics(empty)
        assert q["total_messages"] == 0

    def test_compute_per_file_stats_empty(self):
        from src.analytics.data_quality import compute_per_file_stats
        df = compute_per_file_stats([])
        assert len(df) == 0

    def test_compute_per_file_stats(self):
        from src.analytics.data_quality import compute_per_file_stats
        stats = [{"file": "a.csv", "rows": 100, "errors": 2, "cached": False}]
        df = compute_per_file_stats(stats)
        assert len(df) == 1
        assert df["rows"][0] == 100


# ===================================================================
# comparison
# ===================================================================

class TestComparison:
    def test_compute_period_summary(self, message_fact, edge_fact):
        from src.analytics.comparison import compute_period_summary
        summary = compute_period_summary(message_fact, edge_fact)
        assert summary["total_messages"] == 10
        assert summary["unique_senders"] > 0
        assert 0 <= summary["after_hours_rate"] <= 1

    def test_compute_delta(self):
        from src.analytics.comparison import compute_delta
        current = {"total_messages": 100, "rate": 0.5}
        previous = {"total_messages": 80, "rate": 0.4}
        delta = compute_delta(current, previous)
        assert delta["total_messages"]["delta"] == 20
        assert delta["total_messages"]["pct"] == pytest.approx(25.0)
        assert delta["rate"]["delta"] == pytest.approx(0.1)

    def test_compute_delta_zero_previous(self):
        from src.analytics.comparison import compute_delta
        current = {"val": 10}
        previous = {"val": 0}
        delta = compute_delta(current, previous)
        assert delta["val"]["delta"] == 10
        assert delta["val"]["pct"] == 0.0  # division by zero guard


# ===================================================================
# narrative
# ===================================================================

class TestNarrative:
    def test_generate_executive_narrative(self, message_fact, edge_fact, person_dim):
        from src.analytics.narrative import generate_executive_narrative
        # Need a weekly_agg-like table
        weekly_agg = pl.DataFrame({
            "week_start": [dt.date(2010, 3, 1), dt.date(2010, 3, 8)],
            "msg_count": [5, 5],
            "recipient_impressions": [7, 5],
            "total_bytes": [10000, 8000],
        })
        text = generate_executive_narrative(message_fact, weekly_agg, edge_fact, person_dim)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Work patterns" in text

    def test_empty_message_fact(self, person_dim):
        from src.analytics.narrative import generate_executive_narrative
        empty_mf = pl.DataFrame({
            "msg_id": [], "timestamp": [], "from_email": [],
            "is_after_hours": [], "is_weekend": [],
        })
        empty_wa = pl.DataFrame({"week_start": [], "msg_count": []})
        empty_ef = pl.DataFrame({"from_email": [], "to_email": []})
        text = generate_executive_narrative(empty_mf, empty_wa, empty_ef, person_dim)
        assert "No messages" in text


# ===================================================================
# response_time
# ===================================================================

class TestResponseTime:
    def test_compute_reply_times(self, edge_fact):
        from src.analytics.response_time import compute_reply_times
        result = compute_reply_times(edge_fact)
        assert "person_a" in result.columns
        assert "median_reply_seconds" in result.columns
        # We have back-and-forth between alice and bob, so replies should be found
        assert len(result) > 0

    def test_compute_reply_times_empty(self):
        from src.analytics.response_time import compute_reply_times
        empty = pl.DataFrame({
            "from_email": [], "to_email": [], "timestamp": [],
        })
        result = compute_reply_times(empty)
        assert len(result) == 0

    def test_compute_person_response_stats(self, edge_fact):
        from src.analytics.response_time import compute_reply_times, compute_person_response_stats
        reply_times = compute_reply_times(edge_fact)
        if len(reply_times) > 0:
            stats = compute_person_response_stats(reply_times)
            assert "email" in stats.columns
            assert "median_response_sec" in stats.columns

    def test_compute_department_response_stats(self, edge_fact, person_dim):
        from src.analytics.response_time import (
            compute_reply_times, compute_department_response_stats,
        )
        reply_times = compute_reply_times(edge_fact)
        if len(reply_times) > 0:
            dept = compute_department_response_stats(reply_times, person_dim)
            assert "domain" in dept.columns


# ===================================================================
# hierarchy
# ===================================================================

class TestHierarchy:
    def test_is_likely_nonhuman(self):
        from src.analytics.hierarchy import is_likely_nonhuman
        assert is_likely_nonhuman("noreply@example.com")
        assert is_likely_nonhuman("donotreply@example.com")
        assert is_likely_nonhuman("copier123@example.com")
        assert not is_likely_nonhuman("alice@example.com")

    def test_detect_nonhuman_addresses(self, person_dim, edge_fact):
        from src.analytics.hierarchy import detect_nonhuman_addresses
        result = detect_nonhuman_addresses(person_dim, edge_fact)
        assert "is_nonhuman" in result.columns
        assert len(result) == len(person_dim)

    def test_compute_hierarchy_score(self, edge_fact, person_dim):
        from src.analytics.hierarchy import compute_hierarchy_score
        scores = compute_hierarchy_score(edge_fact, person_dim)
        assert "hierarchy_score" in scores.columns
        assert len(scores) > 0

    def test_infer_reciprocal_teams(self, edge_fact, person_dim):
        from src.analytics.hierarchy import infer_reciprocal_teams
        # With small data and low thresholds
        teams = infer_reciprocal_teams(
            edge_fact, person_dim,
            min_msgs_per_direction=1,
            min_team_size=1,
        )
        assert "manager" in teams.columns

    def test_build_reporting_pairs_empty(self):
        from src.analytics.hierarchy import build_reporting_pairs_from_teams
        empty = pl.DataFrame({
            "manager": [], "team_size": [],
            "total_sent_to_team": [], "total_recv_from_team": [],
            "team_members": [], "display_name": [],
        })
        result = build_reporting_pairs_from_teams(empty)
        assert len(result) == 0


# ===================================================================
# silos
# ===================================================================

class TestSilos:
    def test_compute_community_interaction_matrix(self, edge_fact):
        from src.analytics.silos import compute_community_interaction_matrix
        lookup = {
            "alice@spokanecounty.org": 0,
            "bob@spokanecounty.org": 0,
            "carol@external.com": 1,
        }
        matrix = compute_community_interaction_matrix(edge_fact, lookup)
        assert "comm_from" in matrix.columns
        assert "msg_count" in matrix.columns
        assert len(matrix) > 0

    def test_find_silent_community_pairs(self):
        from src.analytics.silos import find_silent_community_pairs
        matrix = pl.DataFrame({
            "comm_from": [0, 0, 1],
            "comm_to": [0, 1, 1],
            "msg_count": [10, 5, 8],
        })
        # Communities 0, 1, 2 exist. 0-2 and 1-2 should be silent.
        silent = find_silent_community_pairs(matrix, [0, 1, 2])
        assert (0, 2) in silent
        assert (1, 2) in silent
        assert (0, 1) not in silent  # they communicate

    def test_identify_bridges(self, edge_fact):
        import networkx as nx
        from src.analytics.silos import identify_bridges
        from src.analytics.network import build_graph

        G = build_graph(edge_fact)
        lookup = {
            "alice@spokanecounty.org": 0,
            "bob@spokanecounty.org": 0,
            "carol@external.com": 1,
        }
        bridges = identify_bridges(G, lookup)
        assert "email" in bridges.columns
        # Alice and bob connect to carol in community 1
        assert len(bridges) > 0

    def test_simulate_removal(self, edge_fact):
        from src.analytics.silos import simulate_removal
        from src.analytics.network import build_graph

        G = build_graph(edge_fact)
        result = simulate_removal(G, "alice@spokanecounty.org")
        assert "before_components" in result
        assert "after_components" in result
        assert result["before_components"] >= 1


# ===================================================================
# temporal_network
# ===================================================================

class TestTemporalNetwork:
    @pytest.fixture
    def multi_month_edge_fact(self):
        """Edge fact spanning 3 months for temporal analysis."""
        rows = []
        pairs = [
            ("alice@spokanecounty.org", "bob@spokanecounty.org"),
            ("bob@spokanecounty.org", "alice@spokanecounty.org"),
            ("carol@external.com", "alice@spokanecounty.org"),
        ]
        for month_offset in range(3):
            for day in range(1, 15):
                for f, t in pairs:
                    rows.append({
                        "from_email": f, "to_email": t,
                        "timestamp": dt.datetime(2010, 1 + month_offset, day, 10, 0),
                        "size_bytes": 1024,
                        "is_after_hours": False,
                        "is_weekend": False,
                    })
        return pl.DataFrame(rows)

    def test_build_monthly_snapshots(self, multi_month_edge_fact):
        from src.analytics.temporal_network import build_monthly_snapshots
        snapshots = build_monthly_snapshots(multi_month_edge_fact)
        assert len(snapshots) == 3
        for month_id, df in snapshots.items():
            assert "email" in df.columns
            assert "pagerank" in df.columns

    def test_compute_centrality_trends(self, multi_month_edge_fact):
        from src.analytics.temporal_network import (
            build_monthly_snapshots, compute_centrality_trends,
        )
        snapshots = build_monthly_snapshots(multi_month_edge_fact)
        trends = compute_centrality_trends(snapshots)
        assert "email" in trends.columns
        assert "month_id" in trends.columns
        assert len(trends) > 0

    def test_detect_rising_fading(self, multi_month_edge_fact):
        from src.analytics.temporal_network import (
            build_monthly_snapshots, compute_centrality_trends,
            detect_rising_fading,
        )
        snapshots = build_monthly_snapshots(multi_month_edge_fact)
        trends = compute_centrality_trends(snapshots)
        rising, fading = detect_rising_fading(trends, min_months=2)
        assert "email" in rising.columns
        assert "change" in rising.columns

    def test_compute_community_stability(self, multi_month_edge_fact):
        from src.analytics.temporal_network import (
            build_monthly_snapshots, compute_community_stability,
        )
        snapshots = build_monthly_snapshots(multi_month_edge_fact)
        stability = compute_community_stability(snapshots)
        assert "month_pair" in stability.columns
        assert "nmi" in stability.columns

    def test_compute_nmi(self):
        from src.analytics.temporal_network import _compute_nmi
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([0, 0, 1, 1, 2, 2])
        # Identical partitions should have NMI = 1
        nmi = _compute_nmi(labels1, labels2)
        assert nmi == pytest.approx(1.0, abs=0.01)

    def test_compute_nmi_empty(self):
        from src.analytics.temporal_network import _compute_nmi
        assert _compute_nmi(np.array([]), np.array([])) == 0.0


# ===================================================================
# size_forensics
# ===================================================================

class TestSizeForensics:
    def test_classify_by_size(self, message_fact):
        from src.analytics.size_forensics import classify_by_size
        result = classify_by_size(message_fact)
        assert "size_class" in result.columns
        classes = result["size_class"].unique().to_list()
        # With our test data (0 to 5000 bytes), expect tiny and small
        assert any(c in classes for c in ["tiny", "small"])

    def test_detect_size_templates(self, message_fact):
        from src.analytics.size_forensics import detect_size_templates
        # Low min_occurrences for test data
        result = detect_size_templates(message_fact, min_occurrences=2)
        assert "size_bytes" in result.columns
        assert "occurrence_count" in result.columns

    def test_compute_sender_size_profile(self, message_fact):
        from src.analytics.size_forensics import compute_sender_size_profile
        result = compute_sender_size_profile(message_fact)
        assert "from_email" in result.columns
        assert "avg_size" in result.columns
        assert "dominant_class" in result.columns

    def test_detect_size_anomalies(self, message_fact):
        from src.analytics.size_forensics import detect_size_anomalies
        result = detect_size_anomalies(message_fact, z_threshold=1.0)
        assert "size_zscore" in result.columns

    def test_classify_by_size_custom_thresholds(self, message_fact):
        from src.analytics.size_forensics import classify_by_size
        custom = {"tiny": 500, "small": 2000, "medium": 4000, "large": 8000}
        result = classify_by_size(message_fact, thresholds=custom)
        assert "size_class" in result.columns
