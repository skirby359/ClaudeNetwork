"""PST and MBOX file import: extract email headers into message_fact format."""

import email
import email.utils
import mailbox
from datetime import datetime
from pathlib import Path
from typing import Callable

import polars as pl

from src.ingest.email_parser import parse_email_address, parse_recipients
from src.ingest.normalizer import normalize_email, normalize_name
from src.ingest.size_parser import parse_size

try:
    import pypff
    HAS_PST = True
except ImportError:
    HAS_PST = False


def _parse_header_date(date_str: str | None) -> datetime | None:
    """Parse RFC 2822 date header into datetime."""
    if not date_str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        # Strip timezone info for consistency with CSV pipeline
        return parsed.replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _extract_headers(msg) -> dict | None:
    """Extract Date, From, To, Size from an email.message.Message object."""
    date_str = msg.get("Date", "")
    ts = _parse_header_date(date_str)
    if ts is None:
        return None

    from_raw = msg.get("From", "")
    from_name, from_email_addr = parse_email_address(from_raw)
    if not from_email_addr:
        return None
    from_email_addr = normalize_email(from_email_addr)
    from_name = normalize_name(from_name)

    # Combine To, Cc
    to_raw = msg.get("To", "")
    cc_raw = msg.get("Cc", "")
    combined_to = to_raw
    if cc_raw:
        combined_to = f"{to_raw}, {cc_raw}" if to_raw else cc_raw

    recipients = parse_recipients(combined_to)
    if not recipients:
        return None

    to_emails = []
    to_names = []
    for rname, remail in recipients:
        remail = normalize_email(remail)
        rname = normalize_name(rname)
        if remail:
            to_emails.append(remail)
            to_names.append(rname)

    if not to_emails:
        return None

    # Size: use Content-Length header or estimate from payload
    size_str = msg.get("Content-Length", "")
    size_bytes = parse_size(size_str) if size_str else 0
    if not size_bytes:
        try:
            payload = msg.as_bytes() if hasattr(msg, "as_bytes") else str(msg).encode("utf-8", errors="replace")
            size_bytes = len(payload)
        except Exception:
            size_bytes = 0

    return {
        "timestamp": ts,
        "size_bytes": size_bytes,
        "from_email": from_email_addr,
        "from_name": from_name,
        "to_emails": to_emails,
        "to_names": to_names,
        "n_recipients": len(to_emails),
    }


def import_mbox(
    mbox_path: Path,
    start_msg_id: int = 0,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[pl.DataFrame, int, int]:
    """Import an MBOX file into message_fact format.

    Returns (DataFrame, next_msg_id, error_count).
    """
    mbox = mailbox.mbox(str(mbox_path))
    total = len(mbox)
    records = []
    msg_id = start_msg_id
    errors = 0

    for i, msg in enumerate(mbox):
        if progress_callback and total > 0 and i % 1000 == 0:
            progress_callback(i / total, mbox_path.name)

        headers = _extract_headers(msg)
        if headers is None:
            errors += 1
            continue

        headers["msg_id"] = msg_id
        records.append(headers)
        msg_id += 1

    mbox.close()

    if not records:
        return pl.DataFrame(), msg_id, errors

    df = pl.DataFrame(records)
    df = _add_time_columns(df)
    return df, msg_id, errors


def import_pst(
    pst_path: Path,
    start_msg_id: int = 0,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[pl.DataFrame, int, int]:
    """Import a PST file into message_fact format.

    Requires pypff (libpff-python). Returns (DataFrame, next_msg_id, error_count).
    """
    if not HAS_PST:
        raise ImportError(
            "pypff is required for PST import. "
            "Install with: pip install libpff-python"
        )

    pst = pypff.file()
    pst.open(str(pst_path))
    root = pst.get_root_folder()

    records = []
    msg_id = start_msg_id
    errors = 0

    def _walk_folder(folder, depth=0):
        nonlocal msg_id, errors

        for i in range(folder.number_of_sub_messages):
            try:
                msg = folder.get_sub_message(i)
                headers = _extract_pst_message(msg)
                if headers:
                    headers["msg_id"] = msg_id
                    records.append(headers)
                    msg_id += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

        for i in range(folder.number_of_sub_folders):
            try:
                sub = folder.get_sub_folder(i)
                _walk_folder(sub, depth + 1)
            except Exception:
                errors += 1

    _walk_folder(root)
    pst.close()

    if not records:
        return pl.DataFrame(), msg_id, errors

    df = pl.DataFrame(records)
    df = _add_time_columns(df)
    return df, msg_id, errors


def _extract_pst_message(msg) -> dict | None:
    """Extract headers from a pypff message object."""
    try:
        ts = msg.delivery_time
        if ts is None:
            return None
        # pypff returns datetime with tzinfo; strip it
        ts = ts.replace(tzinfo=None)
    except Exception:
        return None

    sender = msg.sender_name or ""
    sender_email = ""
    # Try to extract email from transport headers
    try:
        headers_text = msg.transport_headers
        if headers_text:
            parsed = email.message_from_string(headers_text)
            from_raw = parsed.get("From", "")
            _, sender_email = parse_email_address(from_raw)
    except Exception:
        pass

    if not sender_email:
        # Fallback: try sender_email_address attribute
        try:
            sender_email = msg.sender_email_address or ""
        except Exception:
            pass

    if not sender_email:
        return None

    sender_email = normalize_email(sender_email)
    sender_name = normalize_name(sender)

    # Recipients from transport headers
    to_emails = []
    to_names = []
    try:
        headers_text = msg.transport_headers
        if headers_text:
            parsed = email.message_from_string(headers_text)
            to_raw = parsed.get("To", "")
            cc_raw = parsed.get("Cc", "")
            combined = f"{to_raw}, {cc_raw}" if cc_raw else to_raw
            recipients = parse_recipients(combined)
            for rname, remail in recipients:
                remail = normalize_email(remail)
                rname = normalize_name(rname)
                if remail:
                    to_emails.append(remail)
                    to_names.append(rname)
    except Exception:
        pass

    if not to_emails:
        return None

    size_bytes = msg.size if hasattr(msg, "size") else 0

    return {
        "timestamp": ts,
        "size_bytes": size_bytes or 0,
        "from_email": sender_email,
        "from_name": sender_name,
        "to_emails": to_emails,
        "to_names": to_names,
        "n_recipients": len(to_emails),
    }


def _add_time_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add standard time-derived columns to match CSV pipeline output."""
    df = df.with_columns([
        pl.col("timestamp").dt.strftime("%G-W%V").alias("week_id"),
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.weekday() - 1).cast(pl.Int32).alias("day_of_week"),
    ])
    df = df.with_columns([
        ((pl.col("hour") >= 18) | (pl.col("hour") < 7)).alias("is_after_hours"),
        pl.col("day_of_week").is_in([5, 6]).alias("is_weekend"),
    ])
    return df


def detect_file_type(path: Path) -> str | None:
    """Detect if a file is MBOX or PST based on extension and magic bytes."""
    suffix = path.suffix.lower()
    if suffix == ".mbox":
        return "mbox"
    if suffix == ".pst":
        return "pst"
    # Check magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if header == b"!BDN":  # PST magic bytes
            return "pst"
        if header[:5] == b"From ":  # MBOX typically starts with "From "
            return "mbox"
    except Exception:
        pass
    return None
