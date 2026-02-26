"""Parse human-readable size strings like '10.8K' or '1.2M' into bytes."""

import re

_MULTIPLIERS = {
    "": 1,
    "B": 1,
    "K": 1024,
    "M": 1024 ** 2,
    "G": 1024 ** 3,
}

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([BKMG]?)\s*$", re.IGNORECASE)


def parse_size(s: str) -> int | None:
    """Parse a size string like '10.8K' into integer bytes.

    Returns None if the string cannot be parsed.
    """
    if not s:
        return None
    m = _SIZE_RE.match(s.strip())
    if not m:
        return None
    value = float(m.group(1))
    suffix = m.group(2).upper()
    return int(value * _MULTIPLIERS.get(suffix, 1))
