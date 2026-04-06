"""Tests for the anonymization module (non-Streamlit parts)."""

import polars as pl

from src.anonymize import _build_alias


class TestBuildAlias:
    def test_deterministic(self):
        """Same email + seed always produces same alias."""
        a1 = _build_alias("alice@example.com", "seed123")
        a2 = _build_alias("alice@example.com", "seed123")
        assert a1 == a2

    def test_different_emails_different_aliases(self):
        """Different emails produce different aliases."""
        a = _build_alias("alice@example.com", "seed")
        b = _build_alias("bob@example.com", "seed")
        assert a != b

    def test_different_seeds_different_aliases(self):
        """Different seeds produce different aliases for same email."""
        a = _build_alias("alice@example.com", "seed1")
        b = _build_alias("alice@example.com", "seed2")
        assert a != b

    def test_preserves_domain(self):
        """Alias preserves the original domain."""
        alias = _build_alias("alice@spokanecounty.org", "seed")
        assert alias.endswith("@spokanecounty.org")
        assert alias.startswith("user_")

    def test_format(self):
        """Alias follows user_XXXXXX@domain format."""
        alias = _build_alias("test@example.com", "seed")
        parts = alias.split("@")
        assert len(parts) == 2
        assert parts[0].startswith("user_")
        assert len(parts[0]) == 11  # user_ + 6 hex chars
