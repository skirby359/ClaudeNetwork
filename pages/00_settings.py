"""Page 0: Settings — Data onboarding, internal domains, column mapping."""

import shutil
from pathlib import Path

import streamlit as st
import polars as pl

from src.state import get_config, load_message_fact, load_person_dim
from src.config import AppConfig


def _clear_all_caches(config: AppConfig):
    """Clear all in-memory and disk caches."""
    st.cache_resource.clear()
    st.cache_data.clear()
    st.session_state.pop("date_range", None)
    st.session_state.pop("_date_selection", None)
    st.session_state.pop("data_fingerprint", None)
    for f in config.cache_dir.iterdir():
        if f.suffix in (".parquet", ".pickle"):
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Settings", layout="wide")
st.title("Settings & Data Onboarding")

config = get_config()

# =========================================================================
# Section 1: File Upload
# =========================================================================
st.header("Upload Data")
st.caption(
    "Upload CSV files containing email metadata. "
    "Files are saved to the data directory and ingested automatically."
)

uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key="settings_upload",
)

if uploaded_files:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in uploaded_files:
        dest = config.data_dir / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        saved.append(uploaded.name)
    if saved:
        st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")
        st.info("Click **Reload Pipeline** below to ingest the new data.")

st.divider()

# =========================================================================
# Section 2: Internal Domains Configuration
# =========================================================================
st.header("Internal Domains")
st.caption(
    "Internal domains identify which email addresses belong to your organization. "
    "People with these domains are classified as 'internal' in all analytics."
)

# Load current setting from session state or try to auto-detect
current_domains = st.session_state.get("_internal_domains", None)

if current_domains is None:
    # Try to read from existing person_dim
    try:
        person_dim = load_person_dim()
        all_emails = person_dim["email"].to_list()
        detected = AppConfig.detect_internal_domains(all_emails)
        current_domains = detected
    except Exception:
        current_domains = []

# Show auto-detection info
col_left, col_right = st.columns([3, 1])
with col_left:
    domains_input = st.text_area(
        "Internal domains (one per line)",
        value="\n".join(current_domains),
        height=120,
        placeholder="e.g.\nspokanecounty.org\ncityofspokane.org",
        key="settings_domains_input",
    )

with col_right:
    st.markdown("**Auto-detect**")
    if st.button("Detect from data"):
        try:
            person_dim = load_person_dim()
            all_emails = person_dim["email"].to_list()
            detected = AppConfig.detect_internal_domains(all_emails, top_n=5)
            st.session_state._internal_domains = detected
            st.rerun()
        except Exception as e:
            st.error(f"Could not detect: {e}")

# Parse and save domains
new_domains = [d.strip().lower() for d in domains_input.strip().split("\n") if d.strip()]

if new_domains != current_domains:
    st.session_state._internal_domains = new_domains

# Show current state
if new_domains:
    st.write(f"**Active internal domains:** {', '.join(new_domains)}")
else:
    st.warning("No internal domains configured. All addresses will be classified as external.")

st.divider()

# =========================================================================
# Section 3: Column Mapping
# =========================================================================
st.header("CSV Column Mapping")
st.caption(
    "If your CSV files use different column names than the defaults, "
    "configure the mapping here. The custom CSV parser assumes column "
    "order: Date, Size, From, To (everything after From is the To field)."
)

col_a, col_b, col_c, col_d = st.columns(4)

# Get current mapping from session state
current_mapping = st.session_state.get("_column_mapping", {
    "date": "Date", "size": "Size", "from": "From", "to": "To",
})

with col_a:
    date_col = st.text_input("Date column", value=current_mapping["date"], key="map_date")
with col_b:
    size_col = st.text_input("Size column", value=current_mapping["size"], key="map_size")
with col_c:
    from_col = st.text_input("From column", value=current_mapping["from"], key="map_from")
with col_d:
    to_col = st.text_input("To column", value=current_mapping["to"], key="map_to")

new_mapping = {"date": date_col, "size": size_col, "from": from_col, "to": to_col}
st.session_state._column_mapping = new_mapping

# Date format
date_format = st.text_input(
    "Date format (Python strftime)",
    value=st.session_state.get("_date_format", "%m/%d/%Y %H:%M"),
    key="settings_date_format",
    help="Common formats: %m/%d/%Y %H:%M, %Y-%m-%d %H:%M:%S, %d/%m/%Y %H:%M",
)
st.session_state._date_format = date_format

# Preview a CSV file
st.subheader("CSV Preview")
csv_files = config.discover_csv_files()
if csv_files:
    preview_file = st.selectbox("Select file to preview", [f.name for f in csv_files])
    if preview_file:
        file_path = config.data_dir / preview_file
        try:
            # Read first 5 lines raw
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(6)]
            st.code("".join(lines), language="csv")
        except Exception as e:
            st.error(f"Could not read file: {e}")
else:
    st.info("No CSV files in data directory. Upload files above.")

st.divider()

# =========================================================================
# Section 4: Microsoft 365 Connector
# =========================================================================
st.header("Microsoft 365 Connector")
st.caption(
    "Pull email metadata directly from Microsoft 365 using the Graph API. "
    "Uses **Mail.ReadBasic.All** permission — can read headers (Date, From, To, Size) "
    "but **cannot access email bodies**. Enforced at the API level."
)

with st.expander("Setup Instructions", expanded=False):
    st.markdown("""
    ### How to set up Microsoft 365 access

    **You need:** Admin access to your Microsoft 365 / Azure AD tenant.

    **Step 1: Register an app in Azure AD**
    1. Go to [Azure Portal > App Registrations](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
    2. Click **New registration**
    3. Name: `Email Analytics` (or any name)
    4. Supported account types: **Single tenant**
    5. Redirect URI: leave blank (not needed for app-only auth)
    6. Click **Register**

    **Step 2: Add API permissions**
    1. In your new app, go to **API permissions**
    2. Click **Add a permission** > **Microsoft Graph** > **Application permissions**
    3. Search for `Mail.ReadBasic.All` and check it
    4. Click **Add permissions**
    5. Click **Grant admin consent for [your org]** (requires Global Admin)

    **Step 3: Create a client secret**
    1. Go to **Certificates & secrets**
    2. Click **New client secret**
    3. Set expiry (recommend 12 months)
    4. **Copy the secret value immediately** (you can't see it again)

    **Step 4: Note your IDs**
    - **Tenant ID**: found on the app's Overview page
    - **Client ID** (Application ID): found on the app's Overview page
    - **Client Secret**: the value you just copied

    Paste all three below and click **Test Connection**.
    """)

col_ms_a, col_ms_b = st.columns(2)
with col_ms_a:
    ms_tenant = st.text_input(
        "Tenant ID",
        value=st.session_state.get("_ms_tenant_id", ""),
        key="ms_tenant_input",
        type="default",
    )
    ms_client = st.text_input(
        "Client ID (Application ID)",
        value=st.session_state.get("_ms_client_id", ""),
        key="ms_client_input",
    )
    ms_secret = st.text_input(
        "Client Secret",
        value=st.session_state.get("_ms_client_secret", ""),
        key="ms_secret_input",
        type="password",
    )

with col_ms_b:
    st.markdown("**Fetch Options**")
    ms_since_days = st.number_input(
        "Fetch messages from last N days (0 = all)",
        min_value=0, max_value=3650, value=365,
        key="ms_since_days",
    )
    ms_max_per_user = st.number_input(
        "Max messages per user",
        min_value=1000, max_value=500000, value=50000,
        key="ms_max_per_user",
    )

# Save credentials to session state
if ms_tenant:
    st.session_state._ms_tenant_id = ms_tenant
if ms_client:
    st.session_state._ms_client_id = ms_client
if ms_secret:
    st.session_state._ms_client_secret = ms_secret

# Connection test + user list
ms_col1, ms_col2 = st.columns(2)

with ms_col1:
    if st.button("Test Connection", disabled=not (ms_tenant and ms_client and ms_secret)):
        try:
            from src.ingest.msgraph import GraphConfig, list_users
            gc = GraphConfig(tenant_id=ms_tenant, client_id=ms_client, client_secret=ms_secret)
            with st.spinner("Connecting to Microsoft Graph..."):
                users = list_users(gc)
            st.success(f"Connected. Found **{len(users)} users** with mailboxes.")
            st.session_state._ms_users = users
            # Show first 20
            if users:
                user_display = pl.DataFrame(users).head(20)
                st.dataframe(user_display.to_pandas(), width="stretch")
                if len(users) > 20:
                    st.caption(f"Showing 20 of {len(users)} users.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

with ms_col2:
    if st.button(
        "Fetch Email Metadata",
        type="primary",
        disabled=not (ms_tenant and ms_client and ms_secret),
    ):
        try:
            import datetime as _dt
            from src.ingest.msgraph import GraphConfig, run_graph_ingestion
            from src.cache_manager import write_parquet

            gc = GraphConfig(tenant_id=ms_tenant, client_id=ms_client, client_secret=ms_secret)

            since = None
            if ms_since_days > 0:
                since = _dt.datetime.now() - _dt.timedelta(days=ms_since_days)

            progress = st.progress(0, text="Starting fetch...")

            def _progress(frac, text):
                progress.progress(min(frac, 1.0), text=text)

            with st.spinner("Fetching email metadata from Microsoft 365..."):
                df = run_graph_ingestion(
                    gc,
                    since=since,
                    max_per_user=ms_max_per_user,
                    progress_callback=_progress,
                )

            progress.empty()

            if len(df) == 0:
                st.warning("No messages retrieved. Check date range and permissions.")
            else:
                # Save as parquet in data directory
                out_path = config.data_dir / "microsoft365_messages.parquet"
                config.data_dir.mkdir(parents=True, exist_ok=True)
                write_parquet(df, out_path)

                # Also save as the message_fact cache directly
                cache_path = config.cache_path(config.message_fact_file)
                write_parquet(df, cache_path)

                st.success(f"Fetched **{len(df):,} messages** from {df['from_email'].n_unique():,} senders.")
                st.info("Click **Reload Pipeline** below to process the data.")

                # Auto-detect internal domains
                all_emails = df["from_email"].unique().to_list()
                detected = AppConfig.detect_internal_domains(all_emails)
                if detected:
                    st.session_state._internal_domains = detected
                    st.write(f"Auto-detected internal domains: **{', '.join(detected)}**")

        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.exception(e)

st.divider()

# =========================================================================
# Section 5: Apply Settings & Reload
# =========================================================================
st.header("Apply & Reload")

if st.button("Reload Pipeline with Current Settings", type="primary"):
    # Update the DatasetConfig with user settings
    st.session_state._internal_domains = new_domains
    st.session_state._column_mapping = new_mapping
    st.session_state._date_format = date_format
    _clear_all_caches(config)
    st.rerun()

# Show current data stats
st.divider()
st.header("Current Data Summary")
try:
    mf = load_message_fact()
    st.write(f"**Messages loaded:** {len(mf):,}")
    st.write(f"**Date range:** {mf['timestamp'].min()} to {mf['timestamp'].max()}")
    st.write(f"**Unique senders:** {mf['from_email'].n_unique():,}")

    # Show domain distribution
    person_dim = load_person_dim()
    domain_counts = (
        person_dim.group_by("domain")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    st.subheader("Email Domains")
    st.dataframe(domain_counts.to_pandas(), width="stretch")
except Exception:
    st.info("No data loaded yet. Upload CSV files and click Reload.")
