"""Page 0: Settings — Data onboarding, internal domains, column mapping."""

import os
import shutil
from pathlib import Path

import streamlit as st
import polars as pl

from src.state import get_config, load_message_fact, load_person_dim
from src.config import AppConfig

# Load .env.local if it exists (for Microsoft credentials etc.)
_env_path = Path(__file__).resolve().parent.parent / ".env.local"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


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
# Section 0: Engagement Profiles
# =========================================================================
st.header("Engagement Profiles")
st.caption(
    "Save and load per-client settings (internal domains, departments, thresholds, key dates). "
    "Profiles persist across sessions so you can quickly reload a client's configuration."
)

from src.engagement import (
    list_profiles, save_profile, load_profile, delete_profile,
    collect_current_settings, apply_profile_to_session,
)

profiles = list_profiles()

col_load, col_save = st.columns(2)

with col_load:
    st.subheader("Load Profile")
    if profiles:
        selected_profile = st.selectbox("Select profile", profiles, key="profile_select")
        load_cols = st.columns(2)
        with load_cols[0]:
            if st.button("Load", type="primary"):
                try:
                    settings, dept_df = load_profile(selected_profile)
                    apply_profile_to_session(settings, dept_df, st.session_state)
                    st.success(f"Loaded profile: **{selected_profile}**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load: {e}")
        with load_cols[1]:
            if st.button("Delete"):
                delete_profile(selected_profile)
                st.success(f"Deleted profile: **{selected_profile}**")
                st.rerun()
    else:
        st.info("No saved profiles yet. Save your current settings below.")

with col_save:
    st.subheader("Save Current Settings")
    org_name = st.text_input(
        "Organization name",
        value=st.session_state.get("_org_name", "Organization"),
        key="org_name_input",
    )
    st.session_state._org_name = org_name

    profile_name = st.text_input(
        "Profile name",
        value=org_name.lower().replace(" ", "_") if org_name else "default",
        key="profile_name_input",
    )
    if st.button("Save Profile", type="primary"):
        settings = collect_current_settings(st.session_state)
        dept_df = st.session_state.get("_department_mapping")
        save_profile(profile_name, settings, dept_df)
        st.success(f"Saved profile: **{profile_name}**")
        st.rerun()

st.divider()

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
# Section 1b: PST / MBOX Import
# =========================================================================
st.header("Import PST / MBOX Files")
st.caption(
    "Import email metadata from Outlook PST or Unix MBOX files. "
    "Only headers (Date, From, To, Size) are read — message bodies are not accessed."
)

from src.ingest.mailbox_import import HAS_PST, detect_file_type, import_mbox, import_pst

mailbox_files = st.file_uploader(
    "Upload PST or MBOX files",
    type=["pst", "mbox"],
    accept_multiple_files=True,
    key="mailbox_upload",
)

if not HAS_PST:
    st.caption(
        "PST support requires `libpff-python`. "
        "Install with `pip install libpff-python` (MBOX works without it)."
    )

if mailbox_files:
    for uploaded in mailbox_files:
        # Save to temp location
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getvalue())

        file_type = detect_file_type(tmp_path)
        if file_type is None:
            st.error(f"Could not determine file type for {uploaded.name}.")
            continue

        progress = st.progress(0, text=f"Importing {uploaded.name}...")

        def _progress(frac, name):
            progress.progress(min(frac, 1.0), text=f"Importing {name}...")

        try:
            if file_type == "mbox":
                df, _, errors = import_mbox(tmp_path, progress_callback=_progress)
            elif file_type == "pst":
                df, _, errors = import_pst(tmp_path, progress_callback=_progress)
            else:
                st.error(f"Unsupported file type: {file_type}")
                continue

            progress.empty()

            if len(df) == 0:
                st.warning(f"No messages extracted from {uploaded.name}.")
            else:
                # Save as parquet in data directory
                from src.cache_manager import write_parquet
                out_name = Path(uploaded.name).stem + ".parquet"
                out_path = config.data_dir / out_name
                config.data_dir.mkdir(parents=True, exist_ok=True)
                write_parquet(df, out_path)

                # Also write as CSV-compatible cache
                cache_path = config.cache_path(config.message_fact_file)
                if cache_path.exists():
                    existing = pl.read_parquet(cache_path)
                    df = df.with_columns(
                        (pl.lit(len(existing)) + pl.arange(0, pl.len())).cast(pl.Int64).alias("msg_id")
                    )
                    df = pl.concat([existing, df])
                write_parquet(df, cache_path)

                st.success(
                    f"Imported **{len(df):,} messages** from {uploaded.name} "
                    f"({errors} parse errors skipped)."
                )
                st.info("Click **Reload Pipeline** below to refresh all analytics.")

        except ImportError as e:
            progress.empty()
            st.error(str(e))
        except Exception as e:
            progress.empty()
            st.error(f"Import failed: {e}")

        # Clean up temp file
        try:
            tmp_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            pass

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
# Section 2b: Department Enrichment
# =========================================================================
st.header("Department Enrichment")
st.caption(
    "Upload a CSV mapping email addresses to departments. "
    "This adds a 'department' column to person data across all analytics pages."
)

dept_file = st.file_uploader(
    "Upload department mapping CSV",
    type=["csv"],
    key="dept_upload",
    help="CSV with columns: email, department (header row required)",
)

if dept_file:
    try:
        dept_df = pl.read_csv(dept_file)
        # Normalize column names
        cols_lower = {c: c.lower().strip() for c in dept_df.columns}
        dept_df = dept_df.rename(cols_lower)

        if "email" not in dept_df.columns or "department" not in dept_df.columns:
            st.error("CSV must have 'email' and 'department' columns.")
        else:
            dept_df = dept_df.select(["email", "department"]).with_columns(
                pl.col("email").str.to_lowercase().str.strip_chars()
            )
            st.session_state._department_mapping = dept_df
            n_depts = dept_df["department"].n_unique()
            st.success(
                f"Loaded **{len(dept_df):,} mappings** across **{n_depts} departments**."
            )
            st.dataframe(
                dept_df.group_by("department")
                .agg(pl.len().alias("people"))
                .sort("people", descending=True)
                .to_pandas(),
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Show current mapping status
dept_mapping = st.session_state.get("_department_mapping")
if dept_mapping is not None and not dept_file:
    n_mapped = len(dept_mapping)
    n_depts = dept_mapping["department"].n_unique()
    st.info(f"Department mapping active: **{n_mapped:,} people** across **{n_depts} departments**.")
    if st.button("Clear department mapping"):
        del st.session_state._department_mapping
        st.rerun()
elif dept_mapping is None and not dept_file:
    st.info("No department mapping uploaded. Analytics will use email domain as a department proxy.")

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

# Data Profiler
st.subheader("Data Profiler")
st.caption("Auto-detect file format, encoding, date format, and column mapping.")
csv_files = config.discover_csv_files()
if csv_files:
    preview_file = st.selectbox("Select file to profile", [f.name for f in csv_files])
    if preview_file:
        file_path = config.data_dir / preview_file

        from src.ingest.profiler import profile_csv
        profile = profile_csv(file_path)

        if "error" in profile:
            st.error(profile["error"])
        else:
            # Detection results
            det_cols = st.columns(4)
            with det_cols[0]:
                st.metric("Encoding", profile["encoding"])
            with det_cols[1]:
                st.metric("Delimiter", profile["delimiter_label"])
            with det_cols[2]:
                st.metric("Data Lines", f"{profile['n_data_lines']:,}")
            with det_cols[3]:
                st.metric("Columns", profile["n_columns"])

            # Date format detection
            if profile["date_format"]:
                st.success(f"Date format detected: **{profile['date_format_label']}** (`{profile['date_format']}`)")
                if st.button("Apply detected date format"):
                    st.session_state._date_format = profile["date_format"]
                    st.rerun()
            else:
                st.warning("Could not auto-detect date format. Configure manually above.")

            # Column role detection
            roles = profile["column_roles"]
            header = profile["header"]
            role_labels = []
            for role_name, col_idx in roles.items():
                if col_idx is not None and col_idx < len(header):
                    role_labels.append(f"**{role_name.replace('_col', '').title()}**: column {col_idx} (`{header[col_idx]}`)")
            if role_labels:
                st.info("Detected columns: " + " | ".join(role_labels))

            if profile["is_custom_format"]:
                st.info(
                    "This file appears to use the custom format where recipients spill "
                    "across CSV columns. The built-in parser handles this automatically."
                )

            # Warnings
            for w in profile["warnings"]:
                st.warning(w)

            # Sample data preview
            with st.expander("Raw Data Preview", expanded=False):
                try:
                    with open(file_path, "r", encoding=profile["encoding"], errors="replace") as f:
                        lines = [f.readline() for _ in range(6)]
                    st.code("".join(lines), language="csv")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

            # Parsed sample preview
            if profile["sample_rows"]:
                with st.expander("Parsed Columns Preview", expanded=True):
                    import pandas as pd
                    display_header = profile["header"][:8]  # limit columns
                    display_rows = [row[:8] for row in profile["sample_rows"][:10]]
                    pdf = pd.DataFrame(display_rows, columns=display_header[:len(display_rows[0])] if display_rows else display_header)
                    st.dataframe(pdf, use_container_width=True)
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

    **Step 5: Store credentials securely**

    Create a `.env.local` file in the project root (it's gitignored):
    ```
    MS_TENANT_ID=your-tenant-id-here
    MS_APP_ID=your-app-id-here
    MS_APP_SECRET=your-secret-here
    ```

    The app loads these automatically. You can also paste them below manually.
    """)

# Load credentials: .env.local > environment variables > manual UI input
_env_tenant = os.environ.get("MS_TENANT_ID", "")
_env_client = os.environ.get("MS_APP_ID", "")
_env_secret = os.environ.get("MS_APP_SECRET", "")

if _env_tenant and _env_client and _env_secret:
    st.success("Microsoft credentials loaded from environment / `.env.local`")

col_ms_a, col_ms_b = st.columns(2)
with col_ms_a:
    ms_tenant = st.text_input(
        "Tenant ID",
        value=st.session_state.get("_ms_tenant_id", _env_tenant),
        key="ms_tenant_input",
        type="default",
    )
    ms_client = st.text_input(
        "Application (App) ID",
        value=st.session_state.get("_ms_app_id", _env_client),
        key="ms_app_id_input",
    )
    ms_secret = st.text_input(
        "App Secret",
        value=st.session_state.get("_ms_app_secret", _env_secret),
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
    st.session_state._ms_app_id = ms_client
if ms_secret:
    st.session_state._ms_app_secret = ms_secret

# Connection test + user list
ms_col1, ms_col2 = st.columns(2)

with ms_col1:
    if st.button("Test Connection", disabled=not (ms_tenant and ms_client and ms_secret)):
        try:
            from src.ingest.msgraph import GraphConfig, list_users
            gc = GraphConfig(tenant_id=ms_tenant, app_id=ms_client, app_secret=ms_secret)
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

            gc = GraphConfig(tenant_id=ms_tenant, app_id=ms_client, app_secret=ms_secret)

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
