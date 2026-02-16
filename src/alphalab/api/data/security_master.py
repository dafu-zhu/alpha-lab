"""Security master lookup with point-in-time resolution."""

from datetime import date, timedelta

import polars as pl

from alphalab.api.backend import StorageBackend
from alphalab.api.types import SecurityInfo


class SecurityMaster:
    """Point-in-time security master lookup.

    Resolves symbols, CIKs, or security IDs to SecurityInfo at a given date,
    handling symbol changes and corporate actions.
    """

    SECURITY_MASTER_PATH = "data/meta/master/security_master.parquet"

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage
        self._df: pl.DataFrame | None = None

    def _load(self) -> pl.DataFrame:
        """Load security master with in-memory caching."""
        if self._df is not None:
            return self._df
        self._df = self._storage.read_parquet(self.SECURITY_MASTER_PATH)
        return self._df

    def _to_security_info(self, row: dict[str, object]) -> SecurityInfo:
        """Convert row dict to SecurityInfo."""
        return SecurityInfo(
            security_id=str(row["security_id"]),
            symbol=str(row["symbol"]),
            company=str(row["company"]),
            cik=str(row["cik"]) if row.get("cik") is not None else None,
            start_date=row["start_date"],  # type: ignore[arg-type]
            end_date=row["end_date"] if row.get("end_date") is not None else None,  # type: ignore[arg-type]
            exchange=str(row["exchange"]) if row.get("exchange") is not None else None,
            sector=str(row["sector"]) if row.get("sector") is not None else None,
            industry=str(row["industry"]) if row.get("industry") is not None else None,
            subindustry=str(row["subindustry"]) if row.get("subindustry") is not None else None,
        )

    def resolve(
        self,
        identifier: str,
        as_of: date | None = None,
    ) -> SecurityInfo | None:
        """Resolve identifier to SecurityInfo with smart date handling.

        If the identifier matches a single row, returns it immediately (no date needed).
        If multiple rows match, applies point-in-time filtering.

        Args:
            identifier: Symbol, CIK, or security_id
            as_of: Date for point-in-time lookup (default: yesterday)

        Returns:
            SecurityInfo if found, None otherwise
        """
        df = self._load()

        pit_filter = None
        if as_of is not None:
            pit_filter = (pl.col("start_date") <= as_of) & (
                pl.col("end_date").is_null() | (pl.col("end_date") >= as_of)
            )

        for col in ["symbol", "security_id", "cik"]:
            if col not in df.columns:
                continue
            matches = df.filter(pl.col(col).cast(pl.Utf8) == identifier)
            if len(matches) == 0:
                continue
            # Single row + no explicit as_of → skip PIT
            if len(matches) == 1 and as_of is None:
                return self._to_security_info(matches.row(0, named=True))
            # Apply point-in-time filter
            if pit_filter is None:
                effective = date.today() - timedelta(days=1)
                pit_filter = (pl.col("start_date") <= effective) & (
                    pl.col("end_date").is_null() | (pl.col("end_date") >= effective)
                )
            pit = matches.filter(pit_filter)
            if len(pit) > 0:
                return self._to_security_info(pit.row(0, named=True))

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
