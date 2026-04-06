"""Tests for Microsoft Graph connector (offline — no API calls)."""

import datetime as dt

import polars as pl
import pytest

from src.ingest.msgraph import (
    graph_messages_to_dataframe,
    _extract_email,
    _extract_name,
)


class TestExtractEmail:
    def test_basic(self):
        recipient = {"emailAddress": {"address": "Alice@Example.COM", "name": "Alice"}}
        assert _extract_email(recipient) == "alice@example.com"

    def test_missing(self):
        assert _extract_email({}) == ""
        assert _extract_email({"emailAddress": {}}) == ""

    def test_none_address(self):
        assert _extract_email({"emailAddress": {"address": None}}) == ""


class TestExtractName:
    def test_basic(self):
        recipient = {"emailAddress": {"address": "a@b.com", "name": "Alice Smith"}}
        assert _extract_name(recipient) == "Alice Smith"

    def test_missing(self):
        assert _extract_name({}) == ""


class TestGraphMessagesToDataframe:
    def _make_message(self, sent, from_email, to_emails, size=1024):
        return {
            "sentDateTime": sent,
            "from": {"emailAddress": {"address": from_email, "name": ""}},
            "toRecipients": [
                {"emailAddress": {"address": e, "name": ""}} for e in to_emails
            ],
            "ccRecipients": [],
            "size": size,
            "internetMessageId": f"<test-{sent}>",
        }

    def test_basic_conversion(self):
        msgs = [
            self._make_message("2024-01-15T10:30:00Z", "alice@example.com", ["bob@example.com"]),
            self._make_message("2024-01-15T11:00:00Z", "bob@example.com", ["alice@example.com", "carol@example.com"]),
        ]
        df = graph_messages_to_dataframe(msgs)
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "from_email" in df.columns
        assert "to_emails" in df.columns
        assert "size_bytes" in df.columns
        assert "is_after_hours" in df.columns
        assert "week_id" in df.columns
        assert df["n_recipients"][1] == 2

    def test_empty_input(self):
        df = graph_messages_to_dataframe([])
        assert len(df) == 0

    def test_skips_missing_sent(self):
        msgs = [{"from": {"emailAddress": {"address": "a@b.com"}}, "toRecipients": []}]
        df = graph_messages_to_dataframe(msgs)
        assert len(df) == 0

    def test_skips_missing_from(self):
        msgs = [{"sentDateTime": "2024-01-15T10:00:00Z", "from": {}, "toRecipients": [
            {"emailAddress": {"address": "b@c.com"}}
        ]}]
        df = graph_messages_to_dataframe(msgs)
        assert len(df) == 0

    def test_skips_no_recipients(self):
        msgs = [self._make_message("2024-01-15T10:00:00Z", "a@b.com", [])]
        df = graph_messages_to_dataframe(msgs)
        assert len(df) == 0

    def test_includes_cc(self):
        msg = {
            "sentDateTime": "2024-01-15T10:00:00Z",
            "from": {"emailAddress": {"address": "alice@example.com", "name": ""}},
            "toRecipients": [{"emailAddress": {"address": "bob@example.com", "name": ""}}],
            "ccRecipients": [{"emailAddress": {"address": "carol@example.com", "name": ""}}],
            "size": 500,
        }
        df = graph_messages_to_dataframe([msg])
        assert len(df) == 1
        assert df["n_recipients"][0] == 2

    def test_after_hours_detection(self):
        # 22:00 should be after hours
        msgs = [self._make_message("2024-01-15T22:00:00Z", "a@b.com", ["c@d.com"])]
        df = graph_messages_to_dataframe(msgs)
        assert df["is_after_hours"][0] == True

    def test_weekend_detection(self):
        # 2024-01-13 is a Saturday
        msgs = [self._make_message("2024-01-13T10:00:00Z", "a@b.com", ["c@d.com"])]
        df = graph_messages_to_dataframe(msgs)
        assert df["is_weekend"][0] == True

    def test_msg_id_sequence(self):
        msgs = [
            self._make_message("2024-01-15T10:00:00Z", "a@b.com", ["c@d.com"]),
            self._make_message("2024-01-15T11:00:00Z", "a@b.com", ["c@d.com"]),
        ]
        df = graph_messages_to_dataframe(msgs, start_msg_id=100)
        assert df["msg_id"][0] == 100
        assert df["msg_id"][1] == 101
