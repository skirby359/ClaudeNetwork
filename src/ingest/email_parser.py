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

# Undisclosed / suppressed / hidden recipient patterns
_UNDISCLOSED_RE = re.compile(
    r"(?:undisclosed|suppressed|not\s+shown|hidden|no\s+revelados|destinatarios)",
    re.IGNORECASE,
)

# System/bounce senders that lack an @ sign
_SYSTEM_SENDERS = {
    "mailer-daemon", "postmaster", "system administrator",
}

# Short internal hostnames to expand with a default domain
_SHORT_HOST_RE = re.compile(r"^([\w.+-]+)@(\w+)$")  # user@hostname (no dot in domain)

# Bare user@shorthost (no dot in domain) — not caught by _BARE_EMAIL_RE
_SHORT_EMAIL_RE = re.compile(r"[\w.+-]+@\w+$")


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


def parse_email_address(raw: str, default_domain: str = "spokanecounty.org") -> tuple[str, str]:
    """Parse a single email token into (display_name, email).

    Returns (display_name, normalized_email).
    Handles system senders (MAILER-DAEMON), bounce (<>), and short hostnames (user@galactic).
    """
    raw = raw.strip().strip('"').strip("'").strip()
    if not raw or raw == "<>":
        return ("", "")

    m = _ADDR_RE.search(raw)
    if m:
        name = m.group(1).strip().strip('"').strip("'").strip() if m.group(1) else ""
        email = m.group(2).strip()
        if not email or email == "<>":
            return ("", "")
        email = resolve_imceaex(email)
        # Clean up display name: remove extra quotes and whitespace
        name = re.sub(r'\s+', ' ', name).strip(', ')
        # Expand short hostnames (user@galactic -> user@galactic.spokanecounty.org)
        email = _expand_short_host(email, default_domain)
        return (name, email.lower())

    # Try bare email (full domain with dot)
    m = _BARE_EMAIL_RE.search(raw)
    if m:
        email = resolve_imceaex(m.group(0))
        email = _expand_short_host(email, default_domain)
        return ("", email.lower())

    # Try short-host email (user@hostname, no dot in domain)
    m = _SHORT_EMAIL_RE.search(raw)
    if m:
        email = _expand_short_host(m.group(0), default_domain)
        return ("", email.lower())

    # System senders without @ (MAILER-DAEMON, System Administrator, etc.)
    if raw.lower().replace(" ", "-") in _SYSTEM_SENDERS or raw.lower() in _SYSTEM_SENDERS:
        synthetic = raw.lower().replace(" ", "-") + "@" + default_domain
        return (raw, synthetic)

    return ("", "")


def _expand_short_host(email: str, default_domain: str) -> str:
    """Expand user@hostname to user@hostname.domain when hostname has no dots."""
    m = _SHORT_HOST_RE.match(email)
    if m:
        return f"{m.group(1)}@{m.group(2)}.{default_domain}"
    return email


def is_undisclosed(to_blob: str) -> bool:
    """Check if a To blob represents undisclosed/hidden recipients."""
    cleaned = to_blob.strip().rstrip(":;,").strip()
    return bool(_UNDISCLOSED_RE.search(cleaned))


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


def parse_recipients(to_blob: str, default_domain: str = "spokanecounty.org") -> list[tuple[str, str]]:
    """Parse a To blob into a list of (display_name, email) tuples.

    Handles undisclosed recipients by returning a sentinel address.
    Empty To fields are treated as undisclosed.
    """
    if not to_blob or not to_blob.strip():
        return [("Undisclosed Recipients", f"undisclosed-recipients@{default_domain}")]

    cleaned = to_blob.strip().rstrip(":;,").strip()
    if not cleaned or cleaned == ";":
        return [("Undisclosed Recipients", f"undisclosed-recipients@{default_domain}")]

    # Handle undisclosed/hidden recipients
    if is_undisclosed(to_blob):
        return [("Undisclosed Recipients", f"undisclosed-recipients@{default_domain}")]

    tokens = split_recipients(to_blob)
    results = []
    for token in tokens:
        name, email = parse_email_address(token, default_domain)
        if email:
            results.append((name, email))
    return results
