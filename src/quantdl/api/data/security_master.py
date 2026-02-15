"""Security master lookup with point-in-time resolution."""

from datetime import date, timedelta

import polars as pl

from quantdl.api.storage.backend import StorageBackend
from quantdl.api.storage.cache import DiskCache
from quantdl.api.types import SecurityInfo


class SecurityMaster:
    """Point-in-time security master lookup.

    Resolves symbols, CIKs, or security IDs to SecurityInfo at a given date,
    handling symbol changes and corporate actions.
    """

    SECURITY_MASTER_PATH = "data/master/security_master.parquet"

    def __init__(self, storage: StorageBackend, cache: DiskCache | None = None) -> None:
        self._storage = storage
        self._cache = cache
        self._df: pl.DataFrame | None = None

    def _load(self) -> pl.DataFrame:
        """Load security master with caching."""
        if self._df is not None:
            return self._df

        # Try cache first
        if self._cache:
            cached = self._cache.get(self.SECURITY_MASTER_PATH)
            if cached is not None:
                self._df = cached
                return self._df

        # Fetch from storage
        self._df = self._storage.read_parquet(self.SECURITY_MASTER_PATH)

        # Cache for next time
        if self._cache:
            self._cache.put(self.SECURITY_MASTER_PATH, self._df)

        return self._df

    def _to_security_info(self, row: dict[str, object]) -> SecurityInfo:
        """Convert row dict to SecurityInfo."""
        return SecurityInfo(
            security_id=str(row["security_id"]),
            symbol=str(row["symbol"]),
            company=str(row["company"]),
            cik=str(row["cik"]) if row.get("cik") is not None else None,
            cusip=str(row["cusip"]) if row.get("cusip") is not None else None,
            start_date=row["start_date"],  # type: ignore[arg-type]
            end_date=row["end_date"] if row.get("end_date") is not None else None,  # type: ignore[arg-type]
        )

    def resolve(
        self,
        identifier: str,
        as_of: date | None = None,
    ) -> SecurityInfo | None:
        """Resolve identifier to SecurityInfo at point-in-time.

        Args:
            identifier: Symbol, CIK, or security_id
            as_of: Date for point-in-time lookup (default: today)

        Returns:
            SecurityInfo if found, None otherwise
        """
        df = self._load()
        as_of = as_of or date.today() - timedelta(days=1)

        # Build filter: start_date <= as_of AND (end_date is null OR end_date >= as_of)
        pit_filter = (pl.col("start_date") <= as_of) & (
            pl.col("end_date").is_null() | (pl.col("end_date") >= as_of)
        )

        # Try different identifier types
        for col in ["symbol", "security_id", "cik", "cusip"]:
            if col not in df.columns:
                continue
            # Cast column to string for comparison (handles mixed int/string columns)
            result = df.filter(pit_filter & (pl.col(col).cast(pl.Utf8) == identifier))
            if len(result) > 0:
                row = result.row(0, named=True)
                return self._to_security_info(row)

        return None

    def resolve_batch(
        self,
        identifiers: list[str],
        as_of: date | None = None,
    ) -> dict[str, SecurityInfo | None]:
        """Resolve multiple identifiers.

        Args:
            identifiers: List of symbols, CIKs, etc.
            as_of: Date for point-in-time lookup

        Returns:
            Dict mapping identifier to SecurityInfo (or None if not found)
        """
        return {ident: self.resolve(ident, as_of) for ident in identifiers}

    def get_by_security_id(self, security_id: str) -> SecurityInfo | None:
        """Get security by internal ID (no date filtering)."""
        df = self._load()
        result = df.filter(pl.col("security_id") == security_id)
        if len(result) > 0:
            row = result.row(0, named=True)
            return self._to_security_info(row)
        return None

    def search(
        self,
        query: str,
        as_of: date | None = None,
        limit: int = 10,
    ) -> list[SecurityInfo]:
        """Search securities by partial match on symbol or company name.

        Args:
            query: Search string
            as_of: Date for point-in-time lookup
            limit: Max results to return

        Returns:
            List of matching SecurityInfo
        """
        df = self._load()
        as_of = as_of or date.today() - timedelta(days=1)

        query_lower = query.lower()

        # Filter by PIT and match
        pit_filter = (pl.col("start_date") <= as_of) & (
            pl.col("end_date").is_null() | (pl.col("end_date") >= as_of)
        )

        result = df.filter(
            pit_filter
            & (
                pl.col("symbol").str.to_lowercase().str.contains(query_lower)
                | pl.col("company").str.to_lowercase().str.contains(query_lower)
            )
        ).head(limit)

        return [self._to_security_info(row) for row in result.iter_rows(named=True)]
