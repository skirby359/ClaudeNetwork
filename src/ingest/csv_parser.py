"""Quote-aware CSV line parser for malformed email metadata CSVs.

The CSV is malformed: the To field contains comma-separated recipients that spill
across CSV columns. We cannot use Python's csv module for correct parsing.

Strategy:
1. Read raw lines, detect multi-line continuation
2. Quote-aware character-by-character parsing to extract Date, Size, From
3. Everything after the From field is the raw To blob
"""

import re
from pathlib import Path
from typing import Iterator

# A data line starts with a date pattern like "11/30/2010 13:27"
_DATE_START_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}")


def _parse_csv_fields(line: str) -> list[str]:
    """Quote-aware parser that extracts the first 3 CSV fields from a line.

    Returns [date, size, from, to_blob] where to_blob is everything remaining.
    """
    fields = []
    current = []
    in_quotes = False
    i = 0
    n = len(line)

    while i < n:
        ch = line[i]

        if ch == '"':
            if in_quotes:
                # Check for escaped quote (doubled)
                if i + 1 < n and line[i + 1] == '"':
                    current.append('"')
                    i += 2
                    continue
                else:
                    in_quotes = False
            else:
                in_quotes = True
            i += 1
            continue

        if ch == ',' and not in_quotes:
            fields.append(''.join(current).strip())
            current = []
            # Once we have 3 fields (Date, Size, From), everything else is To blob
            if len(fields) == 3:
                to_blob = line[i + 1:].strip()
                # Strip trailing commas (empty CSV columns)
                to_blob = to_blob.rstrip(',').strip()
                fields.append(to_blob)
                return fields
            i += 1
            continue

        current.append(ch)
        i += 1

    # Flush remaining
    fields.append(''.join(current).strip())

    # Pad to 4 fields if needed
    while len(fields) < 4:
        fields.append("")

    return fields[:4]


def iter_raw_lines(csv_path: Path) -> Iterator[str]:
    """Iterate over logical lines from the CSV, joining continuation lines.

    A continuation line is one that does not start with a date pattern
    and is not the header line.
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()  # Skip header
        if not header:
            return

        current_line = None
        for raw_line in f:
            raw_line = raw_line.rstrip("\n").rstrip("\r")
            if not raw_line.strip():
                continue

            if _DATE_START_RE.match(raw_line):
                if current_line is not None:
                    yield current_line
                current_line = raw_line
            else:
                # Continuation line — append to current
                if current_line is not None:
                    current_line += " " + raw_line.strip()
                # else: orphan line before first data line, skip

        if current_line is not None:
            yield current_line


def parse_csv(csv_path: Path) -> Iterator[dict]:
    """Parse the CSV file and yield dicts with keys: date, size, from_raw, to_raw.

    Yields one dict per logical message line.
    """
    for line in iter_raw_lines(csv_path):
        fields = _parse_csv_fields(line)
        if len(fields) >= 4:
            yield {
                "date": fields[0],
                "size": fields[1],
                "from_raw": fields[2],
                "to_raw": fields[3],
            }
