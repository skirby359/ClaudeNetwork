"""Configuration dataclasses for the Email Metadata Analytics Platform."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for a single dataset/CSV file."""
    name: str
    csv_paths: list[Path] = field(default_factory=list)
    date_col: str = "Date"
    size_col: str = "Size"
    from_col: str = "From"
    to_col: str = "To"
    date_format: str = "%m/%d/%Y %H:%M"
    internal_domains: list[str] = field(default_factory=list)
    after_hours_start: int = 18  # 6 PM
    after_hours_end: int = 7     # 7 AM
    weekend_days: list[int] = field(default_factory=lambda: [5, 6])  # Sat, Sun


@dataclass
class AppConfig:
    """Top-level application configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    cache_dir: Path = field(default=None)
    data_dir: Path = field(default=None)

    # Cache file names
    message_fact_file: str = "message_fact.parquet"
    edge_fact_file: str = "edge_fact.parquet"
    person_dim_file: str = "person_dim.parquet"
    weekly_agg_file: str = "weekly_agg.parquet"
    graph_metrics_file: str = "graph_metrics.parquet"
    network_graph_file: str = "network_graph.pickle"
    dyad_analysis_file: str = "dyad_analysis.parquet"
    timing_metrics_file: str = "timing_metrics.parquet"
    broadcast_metrics_file: str = "broadcast_metrics.parquet"
    anomaly_file: str = "anomalies.parquet"

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = self.project_root / "cache"
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, filename: str) -> Path:
        return self.cache_dir / filename

    def csv_cache_path(self, csv_path: Path) -> Path:
        """Return per-file chunk cache path: cache/chunk_{stem}.parquet."""
        return self.cache_dir / f"chunk_{csv_path.stem}.parquet"

    def discover_csv_files(self) -> list[Path]:
        """Find all CSV files in the data directory, sorted by name."""
        if not self.data_dir.exists():
            return []
        return sorted(self.data_dir.glob("*.csv"))

    def discover_datasets(self) -> dict[str, list[Path]]:
        """Discover named datasets: subdirectories of data/ containing CSVs.

        Returns dict mapping dataset name to list of CSV paths.
        Top-level CSVs in data/ become the 'default' dataset.
        """
        datasets = {}
        # Top-level CSVs
        top_csvs = sorted(self.data_dir.glob("*.csv"))
        if top_csvs:
            datasets["default"] = top_csvs
        # Subdirectories
        if self.data_dir.exists():
            for subdir in sorted(self.data_dir.iterdir()):
                if subdir.is_dir():
                    csvs = sorted(subdir.glob("*.csv"))
                    if csvs:
                        datasets[subdir.name] = csvs
        return datasets

    @property
    def default_dataset(self) -> DatasetConfig:
        csv_files = self.discover_csv_files()
        return DatasetConfig(
            name="all-data",
            csv_paths=csv_files,
        )

    @staticmethod
    def detect_internal_domains(person_emails: list[str], top_n: int = 3) -> list[str]:
        """Auto-detect internal domains from the most common email domains."""
        from collections import Counter
        domains = []
        for email in person_emails:
            if "@" in email:
                domains.append(email.split("@")[1].lower())
        if not domains:
            return []
        counts = Counter(domains)
        # Return the top N domains that together cover >50% of addresses,
        # or just the top N most common
        total = len(domains)
        result = []
        running = 0
        for domain, count in counts.most_common(top_n):
            result.append(domain)
            running += count
            if running / total > 0.5:
                break
        return result
