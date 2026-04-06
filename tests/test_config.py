"""Tests for config module."""

from src.config import AppConfig


class TestDetectInternalDomains:
    def test_basic_detection(self):
        emails = [
            "alice@example.com", "bob@example.com", "carol@example.com",
            "dave@external.org", "eve@other.net",
        ]
        result = AppConfig.detect_internal_domains(emails)
        assert "example.com" in result

    def test_multiple_internal(self):
        emails = [
            "a@dept-a.gov", "b@dept-a.gov", "c@dept-a.gov",
            "d@dept-b.gov", "e@dept-b.gov",
            "f@external.com",
        ]
        result = AppConfig.detect_internal_domains(emails, top_n=3)
        assert "dept-a.gov" in result

    def test_empty(self):
        assert AppConfig.detect_internal_domains([]) == []

    def test_no_at_sign(self):
        assert AppConfig.detect_internal_domains(["noatsign", "another"]) == []

    def test_case_insensitive(self):
        emails = ["a@Example.COM", "b@example.com"]
        result = AppConfig.detect_internal_domains(emails)
        assert "example.com" in result
