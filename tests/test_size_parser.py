"""Tests for size_parser module."""

from src.ingest.size_parser import parse_size


def test_kilobytes():
    assert parse_size("10.8K") == int(10.8 * 1024)


def test_megabytes():
    assert parse_size("1.2M") == int(1.2 * 1024 * 1024)


def test_plain_number():
    assert parse_size("1024") == 1024


def test_bytes_suffix():
    assert parse_size("512B") == 512


def test_gigabytes():
    assert parse_size("2.5G") == int(2.5 * 1024 ** 3)


def test_whitespace():
    assert parse_size("  5.6K  ") == int(5.6 * 1024)


def test_empty():
    assert parse_size("") is None


def test_none_input():
    assert parse_size(None) is None


def test_invalid():
    assert parse_size("abc") is None


def test_case_insensitive():
    assert parse_size("10k") == int(10 * 1024)
    assert parse_size("10m") == int(10 * 1024 ** 2)
