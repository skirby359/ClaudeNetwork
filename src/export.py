"""Export utilities for downloading DataFrames as CSV or Excel."""

import io

import streamlit as st
import polars as pl

from src.anonymize import anon_df


def download_csv_button(df: pl.DataFrame, filename: str, label: str = "Download CSV") -> None:
    """Render a Streamlit download button for a Polars DataFrame as CSV."""
    csv_bytes = anon_df(df).write_csv().encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def download_excel_button(df: pl.DataFrame, filename: str, label: str = "Download Excel") -> None:
    """Render a Streamlit download button for a Polars DataFrame as Excel."""
    buf = io.BytesIO()
    anon_df(df).to_pandas().to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
