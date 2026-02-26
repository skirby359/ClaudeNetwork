"""Build edge_fact and person_dim from message_fact."""

import polars as pl

from src.config import AppConfig, DatasetConfig
from src.cache_manager import cached_parquet


def build_edge_fact(message_fact: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    """Explode message_fact to one row per sender-recipient pair."""
    cache_path = config.cache_path(config.edge_fact_file)
    source_paths = [config.cache_path(config.message_fact_file)]

    def _build():
        # Explode the to_emails list column
        edge = message_fact.select([
            "msg_id", "timestamp", "size_bytes", "from_email",
            "to_emails", "n_recipients", "week_id", "hour",
            "day_of_week", "is_after_hours", "is_weekend",
        ]).explode("to_emails").rename({"to_emails": "to_email"})
        return edge

    return cached_parquet(cache_path, source_paths, _build)


def build_person_dim(
    edge_fact: pl.DataFrame,
    message_fact: pl.DataFrame,
    config: AppConfig,
    dataset: DatasetConfig = None,
) -> pl.DataFrame:
    """Build person dimension table with per-person metrics."""
    if dataset is None:
        dataset = config.default_dataset

    cache_path = config.cache_path(config.person_dim_file)
    source_paths = [config.cache_path(config.edge_fact_file)]

    def _build():
        # Sent counts
        sent = (
            message_fact.group_by("from_email")
            .agg(pl.len().alias("total_sent"))
            .rename({"from_email": "email"})
        )

        # Get from_name for each sender (most common name)
        from_names = (
            message_fact.select(["from_email", "from_name"])
            .filter(pl.col("from_name") != "")
            .group_by("from_email")
            .agg(pl.col("from_name").first().alias("display_name"))
            .rename({"from_email": "email"})
        )

        # Received counts
        received = (
            edge_fact.group_by("to_email")
            .agg(pl.len().alias("total_received"))
            .rename({"to_email": "email"})
        )

        # Combine: full outer join
        person = sent.join(received, on="email", how="full", coalesce=True)
        person = person.join(from_names, on="email", how="left")

        # Fill nulls
        person = person.with_columns([
            pl.col("total_sent").fill_null(0),
            pl.col("total_received").fill_null(0),
        ])

        # Fill missing display_name
        if "display_name" not in person.columns:
            person = person.with_columns(pl.lit("").alias("display_name"))
        person = person.with_columns(
            pl.col("display_name").fill_null("")
        )

        # Add derived columns using vectorized Polars expressions
        internal_domains_lower = [d.lower() for d in dataset.internal_domains]
        domain_expr = (
            pl.when(pl.col("email").str.contains("@"))
            .then(pl.col("email").str.split("@").list.last().str.to_lowercase())
            .otherwise(pl.lit(""))
        )
        dl_regex = r"(?i)^(?:all[-_.]?|everyone|staff|team|group|dept|department)|(?:[-_.](?:list|all|group|team|dept))@|^(?i)dl[-_.]|(?i)undisclosed"
        person = person.with_columns([
            domain_expr.alias("domain"),
            domain_expr.is_in(internal_domains_lower).alias("is_internal"),
            pl.col("email").str.contains(dl_regex).alias("is_distribution_list"),
        ])

        return person

    return cached_parquet(cache_path, source_paths, _build)
