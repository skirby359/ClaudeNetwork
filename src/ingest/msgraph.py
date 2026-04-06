"""Microsoft Graph API connector for email metadata ingestion.

Pulls email headers (Date, Size, From, To) using Mail.ReadBasic.All scope.
This scope CANNOT access email bodies — enforced at the API level.

Auth: Client credentials flow (app-only, no per-user OAuth).
Requires: Azure AD app registration with Mail.ReadBasic.All application permission
          + admin consent from tenant administrator.
"""

import time
import datetime as dt
from dataclasses import dataclass

import httpx
import msal
import polars as pl


GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Only request the metadata fields we need
MESSAGE_SELECT = "sentDateTime,from,toRecipients,ccRecipients,internetMessageId,size"
MESSAGE_TOP = 500  # max per page (Graph API allows up to 999 but 500 is safer)


@dataclass
class GraphConfig:
    """Azure AD app registration credentials."""
    tenant_id: str
    client_id: str
    client_secret: str


def _get_access_token(config: GraphConfig) -> str:
    """Acquire an access token using MSAL client credentials flow."""
    app = msal.ConfidentialClientApplication(
        client_id=config.client_id,
        client_credential=config.client_secret,
        authority=f"https://login.microsoftonline.com/{config.tenant_id}",
    )
    result = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )
    if "access_token" not in result:
        error = result.get("error_description", result.get("error", "Unknown error"))
        raise RuntimeError(f"Failed to acquire token: {error}")
    return result["access_token"]


def _graph_get(client: httpx.Client, url: str, token: str, retries: int = 3) -> dict:
    """GET request to Graph API with retry on 429/503."""
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(retries):
        resp = client.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            # Throttled — respect Retry-After header
            wait = int(resp.headers.get("Retry-After", 5))
            time.sleep(wait)
            continue
        if resp.status_code in (503, 504):
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Graph API request failed after {retries} retries: {url}")


def list_users(config: GraphConfig) -> list[dict]:
    """List all licensed users in the tenant.

    Returns list of dicts with 'id', 'mail', 'displayName'.
    """
    token = _get_access_token(config)
    users = []
    url = f"{GRAPH_BASE}/users?$select=id,mail,displayName&$top=999&$filter=assignedLicenses/$count ne 0&$count=true"

    with httpx.Client(timeout=30) as client:
        headers = {"Authorization": f"Bearer {token}", "ConsistencyLevel": "eventual"}
        while url:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                # Fall back to unfiltered list if $count/$filter not supported
                if "$filter" in url:
                    url = f"{GRAPH_BASE}/users?$select=id,mail,displayName&$top=999"
                    continue
                resp.raise_for_status()
            data = resp.json()
            for u in data.get("value", []):
                if u.get("mail"):  # skip users without email
                    users.append({
                        "id": u["id"],
                        "mail": u["mail"].lower(),
                        "displayName": u.get("displayName", ""),
                    })
            url = data.get("@odata.nextLink")

    return users


def fetch_user_messages(
    config: GraphConfig,
    user_id: str,
    since: dt.datetime | None = None,
    max_messages: int = 50000,
) -> list[dict]:
    """Fetch email metadata for a single user.

    Args:
        config: Azure AD credentials.
        user_id: User ID or UPN (email address).
        since: Only fetch messages after this datetime.
        max_messages: Safety cap on messages per user.

    Returns list of raw message dicts from Graph API.
    """
    token = _get_access_token(config)
    messages = []

    url = (
        f"{GRAPH_BASE}/users/{user_id}/messages"
        f"?$select={MESSAGE_SELECT}"
        f"&$top={MESSAGE_TOP}"
        f"&$orderby=sentDateTime asc"
    )
    if since:
        iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        url += f"&$filter=sentDateTime ge {iso}"

    with httpx.Client(timeout=30) as client:
        while url and len(messages) < max_messages:
            data = _graph_get(client, url, token)
            batch = data.get("value", [])
            messages.extend(batch)
            url = data.get("@odata.nextLink")

    return messages[:max_messages]


def _extract_email(recipient: dict) -> str:
    """Extract email address from a Graph API recipient object."""
    addr = recipient.get("emailAddress", {})
    return (addr.get("address") or "").strip().lower()


def _extract_name(recipient: dict) -> str:
    """Extract display name from a Graph API recipient object."""
    addr = recipient.get("emailAddress", {})
    return (addr.get("name") or "").strip()


def graph_messages_to_dataframe(
    raw_messages: list[dict],
    start_msg_id: int = 0,
    dataset_config=None,
) -> pl.DataFrame:
    """Convert raw Graph API messages to the message_fact schema.

    Matches the same schema as CSV ingestion so all downstream analytics work unchanged.
    """
    records = []
    msg_id = start_msg_id

    for msg in raw_messages:
        sent = msg.get("sentDateTime")
        if not sent:
            continue

        # Parse ISO timestamp
        try:
            ts = dt.datetime.fromisoformat(sent.replace("Z", "+00:00"))
            # Convert to naive datetime (drop timezone) for consistency with CSV pipeline
            ts = ts.replace(tzinfo=None)
        except (ValueError, TypeError):
            continue

        # From
        from_obj = msg.get("from", {})
        from_email = _extract_email(from_obj)
        from_name = _extract_name(from_obj)
        if not from_email:
            continue

        # To + CC recipients
        to_list = msg.get("toRecipients", []) or []
        cc_list = msg.get("ccRecipients", []) or []
        all_recipients = to_list + cc_list

        to_emails = []
        to_names = []
        for r in all_recipients:
            email = _extract_email(r)
            name = _extract_name(r)
            if email:
                to_emails.append(email)
                to_names.append(name)

        if not to_emails:
            continue

        size_bytes = msg.get("size", 0) or 0

        records.append({
            "msg_id": msg_id,
            "timestamp": ts,
            "size_bytes": size_bytes,
            "from_email": from_email,
            "from_name": from_name,
            "to_emails": to_emails,
            "to_names": to_names,
            "n_recipients": len(to_emails),
        })
        msg_id += 1

    if not records:
        return pl.DataFrame()

    df = pl.DataFrame(records)

    # Add time-derived columns (same as CSV pipeline)
    after_hours_start = 18
    after_hours_end = 7
    weekend_days = [5, 6]
    if dataset_config:
        after_hours_start = dataset_config.after_hours_start
        after_hours_end = dataset_config.after_hours_end
        weekend_days = dataset_config.weekend_days

    df = df.with_columns([
        pl.col("timestamp").dt.strftime("%G-W%V").alias("week_id"),
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.weekday() - 1).cast(pl.Int32).alias("day_of_week"),
    ])
    df = df.with_columns([
        ((pl.col("hour") >= after_hours_start) | (pl.col("hour") < after_hours_end)).alias("is_after_hours"),
        pl.col("day_of_week").is_in(weekend_days).alias("is_weekend"),
    ])

    return df


def run_graph_ingestion(
    config: GraphConfig,
    user_ids: list[str] | None = None,
    since: dt.datetime | None = None,
    max_per_user: int = 50000,
    progress_callback=None,
    dataset_config=None,
) -> pl.DataFrame:
    """Full ingestion run: list users, fetch messages, build DataFrame.

    Args:
        config: Azure AD credentials.
        user_ids: Specific user IDs/emails to fetch. If None, fetches all users.
        since: Only fetch messages after this datetime.
        max_per_user: Max messages per user (safety cap).
        progress_callback: Optional callback(fraction, status_text).
        dataset_config: DatasetConfig for time-derived column settings.

    Returns a message_fact DataFrame matching the CSV pipeline schema.
    """
    if user_ids is None:
        if progress_callback:
            progress_callback(0.0, "Listing users...")
        users = list_users(config)
        user_ids = [u["id"] for u in users]

    all_messages = []
    total = len(user_ids)

    for i, uid in enumerate(user_ids):
        if progress_callback:
            progress_callback(i / total, f"Fetching user {i+1}/{total}...")
        try:
            msgs = fetch_user_messages(config, uid, since=since, max_messages=max_per_user)
            all_messages.extend(msgs)
        except Exception as e:
            # Log but continue — don't let one failed user stop the whole run
            print(f"Warning: failed to fetch user {uid}: {e}")

    if progress_callback:
        progress_callback(1.0, "Processing messages...")

    return graph_messages_to_dataframe(
        all_messages,
        dataset_config=dataset_config,
    )
