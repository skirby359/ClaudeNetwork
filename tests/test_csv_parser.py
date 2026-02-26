"""Tests for csv_parser module."""

from src.ingest.csv_parser import _parse_csv_fields


def test_simple_line():
    line = '11/30/2010 13:27,10.8K,"Hopp, Bryan" <BHopp@spokanecounty.org>,"Smith, John" <JSmith@spokanecounty.org>,,,,,'
    fields = _parse_csv_fields(line)
    assert fields[0] == "11/30/2010 13:27"
    assert fields[1] == "10.8K"
    assert "Hopp" in fields[2]
    assert "Smith" in fields[3]


def test_multi_recipient_line():
    line = '12/17/2010 9:16,13.5K,"Lancaster, Bret" <BLancaster@spokanecounty.org>,Ted Warne <tedw@pro-msi.com>, "Hopp, Bryan" <BHopp@spokanecounty.org>,,,,,'
    fields = _parse_csv_fields(line)
    assert fields[0] == "12/17/2010 9:16"
    assert fields[1] == "13.5K"
    assert "Lancaster" in fields[2]
    # To blob should contain both recipients
    assert "Ted Warne" in fields[3]
    assert "Hopp" in fields[3]


def test_imceaex_from():
    line = '11/30/2010 13:27,10.8K,"Hopp, Bryan" <IMCEAEX-_O=SPOKANE+20COUNTY_OU=GALACTIC_CN=RECIPIENTS_CN=BHOPP@spokanecounty.org>,"Smith, John" <JSmith@spokanecounty.org>,,,,,'
    fields = _parse_csv_fields(line)
    assert "IMCEAEX" in fields[2]
    assert "Smith" in fields[3]


def test_trailing_commas_stripped():
    line = '1/1/2010 8:00,5K,user@test.com,recipient@test.com,,,,,,,,,,,'
    fields = _parse_csv_fields(line)
    assert fields[3] == "recipient@test.com"
    assert not fields[3].endswith(",")
