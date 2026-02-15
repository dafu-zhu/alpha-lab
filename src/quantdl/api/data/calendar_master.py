"""Trading calendar lookup with O(1) membership check."""

from datetime import date

import polars as pl

from quantdl.api.storage.backend import StorageBackend
from quantdl.api.storage.cache import DiskCache


class CalendarMaster:
    """Trading calendar for US equity markets.

    Provides O(1) lookup for trading day checks and efficient
    range queries for trading days between dates.
    """

    CALENDAR_MASTER_PATH = "data/meta/master/calendar_master.parquet"

    def __init__(self, storage: StorageBackend, cache: DiskCache | None = None) -> None:
        self._storage = storage
        self._cache = cache
        self._df: pl.DataFrame | None = None
        self._trading_days: set[date] | None = None

    def _load(self) -> pl.DataFrame:
        """Load calendar master with caching."""
        if self._df is not None:
            return self._df

        # Try cache first
        if self._cache:
            cached = self._cache.get(self.CALENDAR_MASTER_PATH)
            if cached is not None:
                self._df = cached
                # Handle both "date" and "timestamp" column names
                date_col = "date" if "date" in self._df.columns else "timestamp"
                self._trading_days = set(self._df[date_col].to_list())
                return self._df

        # Fetch from storage
        self._df = self._storage.read_parquet(self.CALENDAR_MASTER_PATH)

        # Cache for next time
        if self._cache:
            self._cache.put(self.CALENDAR_MASTER_PATH, self._df)

        # Build set for O(1) lookup - handle both column names
        date_col = "date" if "date" in self._df.columns else "timestamp"
        self._trading_days = set(self._df[date_col].to_list())
        return self._df

    def is_trading_day(self, dt: date) -> bool:
        """Check if date is a trading day.

        Args:
            dt: Date to check

        Returns:
            True if trading day, False otherwise
        """
        self._load()
        return dt in self._trading_days  # type: ignore[operator]

    def get_trading_days(self, start: date, end: date) -> list[date]:
        """Get trading days in date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Sorted list of trading days
        """
        df = self._load()
        date_col = "date" if "date" in df.columns else "timestamp"
        filtered = df.filter((pl.col(date_col) >= start) & (pl.col(date_col) <= end))
        return sorted(filtered[date_col].to_list())
