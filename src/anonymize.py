"""Email anonymization: display-layer masking that preserves analytics accuracy.

Anonymization is purely cosmetic — raw data stays intact for correct
computation. Only the display strings change. The mapping is consistent
within a session so the same real email always maps to the same alias.
"""

import hashlib

import streamlit as st
import polars as pl


def _build_alias(email: str, seed: str) -> str:
    """Generate a deterministic alias for an email address."""
    h = hashlib.sha256(f"{seed}{email}".encode()).hexdigest()[:6]
    domain = email.split("@")[1] if "@" in email else "unknown"
    return f"user_{h}@{domain}"


def is_anonymized() -> bool:
    """Return True if anonymization mode is active."""
    return st.session_state.get("_anonymize_enabled", False)


def render_anonymize_toggle() -> bool:
    """Render the anonymize toggle in the sidebar. Returns current state."""
    with st.sidebar:
        enabled = st.toggle(
            "Anonymize emails",
            value=st.session_state.get("_anonymize_enabled", False),
            key="_anon_toggle",
            help="Replace real email addresses with anonymized aliases for demos and privacy",
        )
        st.session_state._anonymize_enabled = enabled
    return enabled


def _get_seed() -> str:
    """Get or create a per-session anonymization seed."""
    if "_anon_seed" not in st.session_state:
        import secrets
        st.session_state._anon_seed = secrets.token_hex(8)
    return st.session_state._anon_seed


def _get_mapping() -> dict[str, str]:
    """Get or create the email -> alias mapping dict."""
    if "_anon_map" not in st.session_state:
        st.session_state._anon_map = {}
    return st.session_state._anon_map


def anon(email: str) -> str:
    """Anonymize a single email address. Pass-through if mode is off."""
    if not is_anonymized() or not email:
        return email
    mapping = _get_mapping()
    if email not in mapping:
        mapping[email] = _build_alias(email, _get_seed())
    return mapping[email]


def anon_name(name: str, email: str) -> str:
    """Anonymize a display name. Returns alias-based name if mode is on."""
    if not is_anonymized() or not name:
        return name
    alias = anon(email)
    # Return the user_XXXX part as the display name
    return alias.split("@")[0]


def anon_df(df: pl.DataFrame, email_cols: list[str] | None = None) -> pl.DataFrame:
    """Anonymize email columns in a Polars DataFrame for display.

    Does NOT modify the original DataFrame — returns a new one.
    Only call this right before st.dataframe() or st.plotly_chart().
    """
    if not is_anonymized():
        return df

    if email_cols is None:
        # Auto-detect email columns
        email_cols = [c for c in df.columns if "email" in c.lower()]

    result = df
    for col in email_cols:
        if col in result.columns:
            original = result[col].to_list()
            anonymized = [anon(e) if isinstance(e, str) else e for e in original]
            result = result.with_columns(pl.Series(col, anonymized))

    # Also anonymize display_name if present
    if "display_name" in result.columns and any(c in result.columns for c in email_cols):
        ref_col = next(c for c in email_cols if c in result.columns)
        names = result["display_name"].to_list()
        emails = result[ref_col].to_list()
        anon_names = [
            anon_name(n, e) if isinstance(n, str) and isinstance(e, str) else n
            for n, e in zip(names, emails)
        ]
        result = result.with_columns(pl.Series("display_name", anon_names))

    return result


def anon_dict(d: dict, email_keys: list[str] | None = None) -> dict:
    """Anonymize email values in a dictionary for display."""
    if not is_anonymized():
        return d

    if email_keys is None:
        email_keys = [k for k in d if "email" in k.lower()]

    result = dict(d)
    for key in email_keys:
        if key in result and isinstance(result[key], str):
            result[key] = anon(result[key])
    return result
