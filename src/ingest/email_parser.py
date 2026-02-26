"""Parse email addresses, display names, and resolve IMCEAEX addresses."""

import re

# Match "Display Name" <email@domain.com> or bare email
_ADDR_RE = re.compile(
    r"""
    (?:                         # Optional display name part
        "?                      # Optional opening quote
        '?                      # Optional inner quote
        ([^"<]*?)               # Display name (group 1)
        '?                      # Optional inner quote
        "?                      # Optional closing quote
        \s*                     # Optional whitespace
    )?
    <([^>]+)>                   # Email in angle brackets (group 2)
    """,
    re.VERBOSE,
)

_BARE_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

# IMCEAEX pattern: IMCEAEX-_O=ORG_OU=UNIT_CN=RECIPIENTS_CN=USER@domain
_IMCEAEX_RE = re.compile(
    r"IMCEAEX-.*_CN=RECIPIENTS_CN=(\w+)@([\w.]+)",
    re.IGNORECASE,
)


def resolve_imceaex(email: str) -> str:
    """Convert IMCEAEX address to a normal-looking email.

    IMCEAEX-_O=SPOKANE+20COUNTY_OU=GALACTIC_CN=RECIPIENTS_CN=BHOPP@spokanecounty.org
    → bhopp@spokanecounty.org
    """
    m = _IMCEAEX_RE.search(email)
    if m:
        user = m.group(1).lower()
        domain = m.group(2).lower()
        return f"{user}@{domain}"
    return email.lower()


def parse_email_address(raw: str) -> tuple[str, str]:
    """Parse a single email token into (display_name, email).

    Returns (display_name, normalized_email).
    """
    raw = raw.strip().strip('"').strip("'").strip()
    if not raw:
        return ("", "")

    m = _ADDR_RE.search(raw)
    if m:
        name = m.group(1).strip().strip('"').strip("'").strip() if m.group(1) else ""
        email = m.group(2).strip()
        email = resolve_imceaex(email)
        # Clean up display name: remove extra quotes and whitespace
        name = re.sub(r'\s+', ' ', name).strip(', ')
        return (name, email.lower())

    # Try bare email
    m = _BARE_EMAIL_RE.search(raw)
    if m:
        email = resolve_imceaex(m.group(0))
        return ("", email.lower())

    return ("", "")


def split_recipients(to_blob: str) -> list[str]:
    """Split a To blob into individual recipient tokens.

    Strategy: split on '>, ' boundary (end of angle-bracket address followed by comma+space).
    Also handles bare comma-separated emails without angle brackets.
    """
    if not to_blob or not to_blob.strip():
        return []

    to_blob = to_blob.strip().rstrip(",").strip()
    if not to_blob:
        return []

    # If there are angle brackets, split on '>, ' and re-add the '>'
    if ">" in to_blob:
        parts = to_blob.split(">, ")
        tokens = []
        for i, part in enumerate(parts):
            part = part.strip().rstrip(",").strip()
            if not part:
                continue
            # Re-add closing bracket if it was split off (except for last part which may already have it)
            if i < len(parts) - 1 and not part.endswith(">"):
                part += ">"
            tokens.append(part)
        return tokens

    # No angle brackets — try comma split for bare emails
    return [t.strip() for t in to_blob.split(",") if t.strip()]


def parse_recipients(to_blob: str) -> list[tuple[str, str]]:
    """Parse a To blob into a list of (display_name, email) tuples."""
    tokens = split_recipients(to_blob)
    results = []
    for token in tokens:
        name, email = parse_email_address(token)
        if email:
            results.append((name, email))
    return results
