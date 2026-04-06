"""Tests for organizational health score."""

import datetime as dt
import polars as pl
import pytest

from src.analytics.health_score import compute_health_score


@pytest.fixture
def sample_data():
    mf = pl.DataFrame({
        "msg_id": list(range(20)),
        "timestamp": [dt.datetime(2024, 1, 15, 10 + i % 12, 0) for i in range(20)],
        "from_email": ["alice@ex.com"] * 5 + ["bob@ex.com"] * 5 + ["carol@ex.com"] * 5 + ["dave@ex.com"] * 5,
        "size_bytes": [1024] * 20,
        "is_after_hours": [False] * 16 + [True] * 4,
        "is_weekend": [False] * 20,
    })
    ef = pl.DataFrame({
        "from_email": ["alice@ex.com", "bob@ex.com", "alice@ex.com", "carol@ex.com", "dave@ex.com", "bob@ex.com"],
        "to_email": ["bob@ex.com", "alice@ex.com", "carol@ex.com", "alice@ex.com", "carol@ex.com", "dave@ex.com"],
        "timestamp": [dt.datetime(2024, 1, 15, 10 + i, 0) for i in range(6)],
        "size_bytes": [1024] * 6,
    })
    gm = pl.DataFrame({
        "email": ["alice@ex.com", "bob@ex.com", "carol@ex.com", "dave@ex.com"],
        "community_id": [0, 0, 1, 1],
        "betweenness_centrality": [0.05, 0.02, 0.01, 0.0],
        "pagerank": [0.3, 0.25, 0.25, 0.2],
    })
    return mf, ef, gm


class TestHealthScore:
    def test_returns_composite(self, sample_data):
        mf, ef, gm = sample_data
        result = compute_health_score(mf, ef, gm)
        assert "composite" in result
        assert 0 <= result["composite"] <= 100

    def test_has_six_subscores(self, sample_data):
        mf, ef, gm = sample_data
        result = compute_health_score(mf, ef, gm)
        assert len(result["sub_scores"]) == 6
        for key, score in result["sub_scores"].items():
            assert "value" in score
            assert "label" in score
            assert 0 <= score["value"] <= 100

    def test_with_reply_time(self, sample_data):
        mf, ef, gm = sample_data
        result = compute_health_score(mf, ef, gm, reply_median_seconds=900)  # 15 min
        assert result["sub_scores"]["responsiveness"]["value"] == 100

    def test_slow_reply(self, sample_data):
        mf, ef, gm = sample_data
        result = compute_health_score(mf, ef, gm, reply_median_seconds=86400)  # 24 hours
        assert result["sub_scores"]["responsiveness"]["value"] < 10

    def test_empty_data(self):
        mf = pl.DataFrame({"msg_id": [], "from_email": [], "is_after_hours": [], "is_weekend": []})
        ef = pl.DataFrame({"from_email": [], "to_email": []})
        gm = pl.DataFrame({"email": [], "community_id": [], "betweenness_centrality": []})
        result = compute_health_score(mf, ef, gm)
        assert "composite" in result
