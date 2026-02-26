"""Configurable column mapping for multi-file CSV support."""

from dataclasses import dataclass


@dataclass
class ColumnMapping:
    """Maps CSV column names to expected internal names."""
    date: str = "Date"
    size: str = "Size"
    from_addr: str = "From"
    to_addr: str = "To"

    def map_row(self, row: dict) -> dict:
        """Map a raw row dict to internal field names."""
        return {
            "date": row.get(self.date, ""),
            "size": row.get(self.size, ""),
            "from_raw": row.get(self.from_addr, ""),
            "to_raw": row.get(self.to_addr, ""),
        }
