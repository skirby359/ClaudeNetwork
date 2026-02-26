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
    internal_domains: list[str] = field(default_factory=lambda: ["spokanecounty.org"])
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

    def discover_csv_files(self) -> list[Path]:
        """Find all CSV files in the data directory, sorted by name."""
        if not self.data_dir.exists():
            return []
        return sorted(self.data_dir.glob("*.csv"))

    @property
    def default_dataset(self) -> DatasetConfig:
        csv_files = self.discover_csv_files()
        return DatasetConfig(
            name="all-data",
            csv_paths=csv_files,
        )
