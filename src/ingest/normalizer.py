"""Identity normalization and distribution list detection."""

import re

# Common distribution list patterns
_DL_PATTERNS = [
    re.compile(r"^(all[-_.]?|everyone|staff|team|group|dept|department)", re.IGNORECASE),
    re.compile(r"([-_.]list|[-_.]all|[-_.]group|[-_.]team|[-_.]dept)@", re.IGNORECASE),
    re.compile(r"^dl[-_.]", re.IGNORECASE),
    re.compile(r"undisclosed", re.IGNORECASE),
]


def normalize_email(email: str) -> str:
    """Case-fold and strip whitespace from an email address."""
    return email.strip().lower()


def normalize_name(name: str) -> str:
    """Clean up a display name: remove extra quotes, whitespace, reorder Last, First."""
    name = name.strip().strip('"').strip("'").strip()
    name = re.sub(r"\s+", " ", name)

    # If name is "Last, First" format, reorder to "First Last"
    if "," in name:
        parts = name.split(",", 1)
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            if first and last and not "@" in name:
                name = f"{first} {last}"

    return name


def is_distribution_list(email: str, name: str = "") -> bool:
    """Heuristic check if an address is a distribution list."""
    for pat in _DL_PATTERNS:
        if pat.search(email) or (name and pat.search(name)):
            return True
    return False


def is_internal(email: str, internal_domains: list[str]) -> bool:
    """Check if an email belongs to an internal domain."""
    email_lower = email.lower()
    for domain in internal_domains:
        if email_lower.endswith("@" + domain.lower()):
            return True
    return False


def extract_domain(email: str) -> str:
    """Extract the domain part of an email address."""
    if "@" in email:
        return email.split("@", 1)[1].lower()
    return ""
