"""Storage backend using Polars native scan_parquet — local filesystem only."""

from pathlib import Path
from typing import Any

import polars as pl

from quantdl.api.exceptions import StorageError


class StorageBackend:
    """Local-only storage backend using Polars parquet I/O.

    Reads data from the local filesystem using Polars' scan_parquet
    and read_parquet for efficient lazy/eager evaluation.
    """

    def __init__(self, data_path: str | Path) -> None:
        """Initialize storage backend.

        Args:
            data_path: Path to local data directory
        """
        self._data_path = Path(data_path)

    def _resolve_path(self, path: str) -> str:
        """Resolve relative path to absolute local path."""
        return str(self._data_path / path.lstrip("/"))

    def scan_parquet(
        self,
        path: str,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Scan parquet file as LazyFrame.

        Args:
            path: Path within data directory (e.g., "data/master/security_master.parquet")
            columns: Optional list of columns to select

        Returns:
            LazyFrame for lazy evaluation with predicate pushdown
        """
        resolved = self._resolve_path(path)
        try:
            lf = pl.scan_parquet(resolved)
            if columns:
                lf = lf.select(columns)
            return lf
        except Exception as e:
            raise StorageError("scan_parquet", path, e) from e

    def read_parquet(
        self,
        path: str,
        columns: list[str] | None = None,
        filters: list[Any] | None = None,
    ) -> pl.DataFrame:
        """Read parquet file into DataFrame.

        Args:
            path: Path within data directory
            columns: Optional list of columns to select
            filters: Optional list of filter expressions to apply

        Returns:
            DataFrame with data
        """
        lf = self.scan_parquet(path, columns)
        if filters:
            for f in filters:
                lf = lf.filter(f)
        try:
            return lf.collect()
        except Exception as e:
            raise StorageError("read_parquet", path, e) from e

    def exists(self, path: str) -> bool:
        """Check if a path exists on local filesystem."""
        resolved = self._resolve_path(path)
        return Path(resolved).exists()
