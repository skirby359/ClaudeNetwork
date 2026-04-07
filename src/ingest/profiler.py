"""Data profiler: auto-detect CSV format, encoding, date format, and column mapping."""

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Common date format patterns to try, ordered by likelihood
DATE_FORMATS = [
    ("%m/%d/%Y %H:%M", "MM/DD/YYYY HH:MM"),
    ("%m/%d/%Y %H:%M:%S", "MM/DD/YYYY HH:MM:SS"),
    ("%Y-%m-%d %H:%M:%S", "YYYY-MM-DD HH:MM:SS"),
    ("%Y-%m-%d %H:%M", "YYYY-MM-DD HH:MM"),
    ("%d/%m/%Y %H:%M", "DD/MM/YYYY HH:MM"),
    ("%d/%m/%Y %H:%M:%S", "DD/MM/YYYY HH:MM:SS"),
    ("%m-%d-%Y %H:%M", "MM-DD-YYYY HH:MM"),
    ("%Y/%m/%d %H:%M:%S", "YYYY/MM/DD HH:MM:SS"),
    ("%d-%b-%Y %H:%M", "DD-Mon-YYYY HH:MM"),
    ("%d-%b-%Y %H:%M:%S", "DD-Mon-YYYY HH:MM:SS"),
    ("%b %d, %Y %H:%M", "Mon DD, YYYY HH:MM"),
    ("%Y-%m-%dT%H:%M:%S", "ISO 8601"),
    ("%Y-%m-%dT%H:%M:%SZ", "ISO 8601 UTC"),
]

# Common encodings to try
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

# Regex for detecting email-like strings
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Regex for detecting size-like strings
_SIZE_RE = re.compile(r"^\d+\.?\d*\s*[KMGkmg]?[Bb]?$")

# Regex for date-like patterns
_DATE_RE = re.compile(r"\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}")


def detect_encoding(file_path: Path) -> str:
    """Try multiple encodings and return the first that works cleanly."""
    sample_bytes = file_path.read_bytes()[:8192]

    for enc in ENCODINGS:
        try:
            text = sample_bytes.decode(enc)
            # Check for replacement characters (sign of wrong encoding)
            if "\ufffd" not in text and "\x00" not in text:
                return enc
        except (UnicodeDecodeError, UnicodeError):
            continue

    return "utf-8"  # fallback


def detect_delimiter(sample_lines: list[str]) -> str:
    """Detect CSV delimiter from sample lines."""
    candidates = {",": 0, "\t": 0, ";": 0, "|": 0}
    for line in sample_lines[1:6]:  # skip header, check 5 data lines
        for delim in candidates:
            candidates[delim] += line.count(delim)

    if not any(candidates.values()):
        return ","

    return max(candidates, key=candidates.get)


def detect_date_format(date_samples: list[str]) -> tuple[str, str] | None:
    """Try each format against sample date strings, return first that works.

    Returns (strftime_format, human_readable_label) or None.
    """
    for fmt, label in DATE_FORMATS:
        matches = 0
        for sample in date_samples[:20]:
            sample = sample.strip().strip('"').strip("'")
            if not sample:
                continue
            try:
                datetime.strptime(sample, fmt)
                matches += 1
            except ValueError:
                continue

        # If >50% of samples parse, we found it
        if matches > len(date_samples) * 0.5:
            return fmt, label

    return None


def detect_column_roles(header: list[str], sample_rows: list[list[str]]) -> dict:
    """Detect which columns contain date, size, from, and to fields.

    Returns dict with keys: date_col, size_col, from_col, to_col (0-indexed positions).
    """
    n_cols = len(header)
    roles = {"date_col": None, "size_col": None, "from_col": None, "to_col": None}

    # Score each column
    for col_idx in range(n_cols):
        col_name = header[col_idx].lower().strip().strip('"')
        values = [row[col_idx].strip() if col_idx < len(row) else "" for row in sample_rows]

        # Check by header name first
        if any(k in col_name for k in ("date", "time", "sent", "received")):
            if roles["date_col"] is None:
                roles["date_col"] = col_idx
                continue

        if any(k in col_name for k in ("size", "bytes", "length")):
            if roles["size_col"] is None:
                roles["size_col"] = col_idx
                continue

        if any(k in col_name for k in ("from", "sender", "author")):
            if roles["from_col"] is None:
                roles["from_col"] = col_idx
                continue

        if any(k in col_name for k in ("to", "recipient", "receiver")):
            if roles["to_col"] is None:
                roles["to_col"] = col_idx
                continue

        # Check by content patterns
        non_empty = [v for v in values if v]
        if not non_empty:
            continue

        date_score = sum(1 for v in non_empty if _DATE_RE.search(v)) / len(non_empty)
        email_score = sum(1 for v in non_empty if _EMAIL_RE.search(v)) / len(non_empty)
        size_score = sum(1 for v in non_empty if _SIZE_RE.match(v)) / len(non_empty)

        if date_score > 0.5 and roles["date_col"] is None:
            roles["date_col"] = col_idx
        elif size_score > 0.5 and roles["size_col"] is None:
            roles["size_col"] = col_idx
        elif email_score > 0.5:
            if roles["from_col"] is None:
                roles["from_col"] = col_idx
            elif roles["to_col"] is None:
                roles["to_col"] = col_idx

    return roles


def profile_csv(file_path: Path) -> dict:
    """Profile a CSV file and return detected settings.

    Returns dict with:
        encoding, delimiter, n_lines, n_columns,
        header, sample_rows, date_format, date_format_label,
        column_roles, date_samples, warnings
    """
    warnings = []

    # Detect encoding
    encoding = detect_encoding(file_path)

    # Read sample lines
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        raw_lines = []
        for i, line in enumerate(f):
            raw_lines.append(line.rstrip("\n").rstrip("\r"))
            if i >= 50:
                break

    if len(raw_lines) < 2:
        return {"error": "File has fewer than 2 lines", "warnings": ["File too short"]}

    # Count total lines (fast)
    n_lines = 0
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        for _ in f:
            n_lines += 1

    # Detect delimiter
    delimiter = detect_delimiter(raw_lines)

    # Parse header and sample rows
    header_line = raw_lines[0]
    header = [f.strip().strip('"') for f in header_line.split(delimiter)]
    n_columns = len(header)

    sample_rows = []
    for line in raw_lines[1:21]:
        fields = [f.strip().strip('"') for f in line.split(delimiter)]
        sample_rows.append(fields)

    # Detect column roles
    column_roles = detect_column_roles(header, sample_rows)

    # Detect date format from date column
    date_format = None
    date_format_label = None
    date_samples = []
    if column_roles["date_col"] is not None:
        date_col = column_roles["date_col"]
        date_samples = [row[date_col] for row in sample_rows if date_col < len(row)]
        result = detect_date_format(date_samples)
        if result:
            date_format, date_format_label = result
        else:
            warnings.append(f"Could not auto-detect date format from samples: {date_samples[:3]}")
    else:
        warnings.append("Could not identify a date column")

    # Check for common issues
    if column_roles["from_col"] is None:
        warnings.append("Could not identify a sender/from column")
    if column_roles["to_col"] is None:
        warnings.append("Could not identify a recipient/to column")

    # Check for encoding issues
    sample_text = "\n".join(raw_lines[:10])
    if "?" in sample_text and encoding != "utf-8":
        warnings.append(f"Possible encoding issues detected (using {encoding})")

    # Detect if this is the custom Spokane-style format (no proper CSV quoting)
    is_custom_format = False
    if n_columns >= 4:
        # If the "to" column values contain commas, it might be the custom format
        if column_roles["to_col"] is not None:
            to_col = column_roles["to_col"]
            for row in sample_rows:
                if to_col < len(row):
                    remaining = delimiter.join(row[to_col:])
                    if _EMAIL_RE.findall(remaining) and len(remaining) > 100:
                        is_custom_format = True
                        break

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "delimiter_label": {"," : "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}.get(delimiter, delimiter),
        "n_lines": n_lines,
        "n_data_lines": n_lines - 1,
        "n_columns": n_columns,
        "header": header,
        "sample_rows": sample_rows[:10],
        "date_format": date_format,
        "date_format_label": date_format_label,
        "date_samples": date_samples[:5],
        "column_roles": column_roles,
        "is_custom_format": is_custom_format,
        "warnings": warnings,
    }
