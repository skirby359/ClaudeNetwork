"""Tests for email_parser module."""

from src.ingest.email_parser import (
    parse_email_address,
    split_recipients,
    parse_recipients,
    resolve_imceaex,
)


class TestResolveImceaex:
    def test_basic(self):
        addr = "IMCEAEX-_O=SPOKANE+20COUNTY_OU=GALACTIC_CN=RECIPIENTS_CN=BHOPP@spokanecounty.org"
        assert resolve_imceaex(addr) == "bhopp@spokanecounty.org"

    def test_non_imceaex_passthrough(self):
        assert resolve_imceaex("user@example.com") == "user@example.com"


class TestParseEmailAddress:
    def test_quoted_name_with_brackets(self):
        raw = '"Hopp, Bryan" <BHopp@spokanecounty.org>'
        name, email = parse_email_address(raw)
        assert email == "bhopp@spokanecounty.org"
        assert "Bryan" in name or "Hopp" in name

    def test_imceaex_address(self):
        raw = '"Hopp, Bryan" <IMCEAEX-_O=SPOKANE+20COUNTY_OU=GALACTIC_CN=RECIPIENTS_CN=BHOPP@spokanecounty.org>'
        name, email = parse_email_address(raw)
        assert email == "bhopp@spokanecounty.org"

    def test_bare_email(self):
        name, email = parse_email_address("user@example.com")
        assert email == "user@example.com"
        assert name == ""

    def test_name_with_angle_brackets(self):
        raw = "Ted Warne <tedw@pro-msi.com>"
        name, email = parse_email_address(raw)
        assert email == "tedw@pro-msi.com"
        assert "Ted" in name

    def test_empty(self):
        name, email = parse_email_address("")
        assert email == ""

    def test_quoted_email_in_name(self):
        raw = """\"'Karl_Susz@lincolnelectric.com'\" <Karl_Susz@lincolnelectric.com>"""
        name, email = parse_email_address(raw)
        assert email == "karl_susz@lincolnelectric.com"


class TestSplitRecipients:
    def test_single_recipient(self):
        blob = '"Hopp, Bryan" <BHopp@spokanecounty.org>'
        tokens = split_recipients(blob)
        assert len(tokens) == 1

    def test_multiple_recipients(self):
        blob = 'Ted Warne <tedw@pro-msi.com>, "Hopp, Bryan" <BHopp@spokanecounty.org>'
        tokens = split_recipients(blob)
        assert len(tokens) == 2

    def test_trailing_commas(self):
        blob = '"Hopp, Bryan" <BHopp@spokanecounty.org>,,,,,,'
        tokens = split_recipients(blob)
        assert len(tokens) == 1

    def test_empty(self):
        assert split_recipients("") == []
        assert split_recipients("   ") == []


class TestParseRecipients:
    def test_two_recipients(self):
        blob = 'Ted Warne <tedw@pro-msi.com>, "Hopp, Bryan" <BHopp@spokanecounty.org>'
        results = parse_recipients(blob)
        assert len(results) == 2
        emails = [r[1] for r in results]
        assert "tedw@pro-msi.com" in emails
        assert "bhopp@spokanecounty.org" in emails
