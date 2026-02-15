"""QuantDL client - main entry point for financial data access (local-only)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from quantdl.api.data.calendar_master import CalendarMaster
from quantdl.api.data.security_master import SecurityMaster
from quantdl.api.exceptions import ConfigurationError, DataNotFoundError, ValidationError
from quantdl.api.backend import StorageBackend
from quantdl.api.types import SecurityInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

# Duration concepts: income statement and cash flow items measured over time
# These default to TTM (trailing twelve months) instead of quarterly raw values
DURATION_CONCEPTS = {
    "sales",
    "cogs",
    "operating_income",
    "income",
    "pretax_income",
    "income_tax",
    "int_exp",
    "rnd",
    "sga_expense",
    "depre_amort",
    "cashflow_op",
    "cashflow_invest",
    "cashflow_fin",
    "capex",
    "dividend",
    "sto_isu",
}


class QuantDLClient:
    """Client for fetching financial data from local storage with optional caching.

    Example:
        ```python
        client = QuantDLClient(data_path="/path/to/data")

        # Get daily prices as wide table
        prices = client.ticks(["AAPL", "MSFT", "GOOGL"], "close", "2024-01-01", "2024-12-31")

        # Get fundamentals
        fundamentals = client.fundamentals(["AAPL"], "Revenue", "2024-01-01", "2024-12-31")
        ```
    """

    def __init__(
        self,
        data_path: str,
        max_concurrency: int = 10,
    ) -> None:
        """Initialize QuantDL client.

        Args:
            data_path: Path to local data directory (required)
            max_concurrency: Max concurrent requests (default: 10)
        """
        if not data_path:
            raise ConfigurationError("data_path is required")

        self._storage = StorageBackend(data_path=data_path)
        self._security_master = SecurityMaster(self._storage)
        self._calendar_master = CalendarMaster(self._storage)
        self._max_concurrency = max_concurrency
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency)

    @property
    def security_master(self) -> SecurityMaster:
        """Access security master for direct lookups."""
        return self._security_master

    @property
    def calendar_master(self) -> CalendarMaster:
        """Access calendar master for trading day lookups."""
        return self._calendar_master

    def resolve(self, identifier: str, as_of: date | None = None) -> SecurityInfo | None:
        """Resolve symbol/identifier to SecurityInfo.

        Args:
            identifier: Symbol, CIK, or security_id
            as_of: Point-in-time date (default: today)

        Returns:
            SecurityInfo if found, None otherwise
        """
        return self._security_master.resolve(identifier, as_of)

    def _resolve_securities(
        self,
        symbols: Sequence[str],
        as_of: date | None = None,
    ) -> list[tuple[str, SecurityInfo]]:
        """Resolve symbols and return list of (symbol, info) pairs."""
        result: list[tuple[str, SecurityInfo]] = []
        for sym in symbols:
            info = self._security_master.resolve(sym, as_of)
            if info is not None:
                result.append((sym, info))
        return result

    def _align_to_calendar(
        self, wide: pl.DataFrame, start: date, end: date, forward_fill: bool = False
    ) -> pl.DataFrame:
        """Align wide table rows to trading calendar."""
        trading_days = self._calendar_master.get_trading_days(start, end)
        calendar_df = pl.DataFrame({"Date": trading_days})
        aligned = calendar_df.join(wide, on="Date", how="left").sort("Date")
        if forward_fill:
            # Forward fill all columns except timestamp
            value_cols = [c for c in aligned.columns if c != "Date"]
            aligned = aligned.with_columns([pl.col(c).forward_fill() for c in value_cols])
        return aligned

    def _fetch_ticks_single(
        self,
        security_id: str,
        start: date,
        end: date,
    ) -> pl.DataFrame | None:
        """Fetch daily ticks for single security."""
        path = f"data/raw/ticks/daily/{security_id}/ticks.parquet"
        try:
            df = self._storage.read_parquet(path)
            if df.schema["Date"] == pl.String:
                df = df.with_columns(pl.col("Date").str.to_date())
            return df.filter(
                (pl.col("Date") >= start) & (pl.col("Date") <= end)
            )
        except Exception:
            return None

    async def _fetch_ticks_async(
        self,
        securities: list[tuple[str, SecurityInfo]],
        start: date,
        end: date,
    ) -> list[tuple[str, pl.DataFrame]]:
        """Fetch daily data for multiple securities concurrently."""
        loop = asyncio.get_event_loop()
        futures: list[tuple[str, asyncio.Future[pl.DataFrame | None]]] = []

        for symbol, info in securities:
            future = loop.run_in_executor(
                self._executor,
                self._fetch_ticks_single,
                info.security_id,
                start,
                end,
            )
            futures.append((symbol, future))

        results: list[tuple[str, pl.DataFrame]] = []
        for symbol, future in futures:
            df = await future
            if df is not None and len(df) > 0:
                results.append((symbol, df))

        return results

    def ticks(
        self,
        symbols: Sequence[str] | str,
        field: str = "close",
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pl.DataFrame:
        """Get daily price data as wide table.

        Args:
            symbols: Symbol(s) to fetch
            field: Price field (open, high, low, close, volume)
            start: Start date
            end: End date (default: today)

        Returns:
            Wide DataFrame with timestamp as first column, symbols as other columns
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Parse dates
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        start = start or date(2000, 1, 1)
        end = end or date.today() - timedelta(days=1)

        # Resolve symbols to security IDs
        resolved = self._resolve_securities(symbols, as_of=start)
        if not resolved:
            raise DataNotFoundError("ticks", ", ".join(symbols))

        # Fetch data concurrently
        results = asyncio.run(self._fetch_ticks_async(resolved, start, end))

        if not results:
            raise DataNotFoundError("ticks", ", ".join(symbols))

        # Build wide table
        dfs: list[pl.DataFrame] = []
        for symbol, df in results:
            if field not in df.columns:
                continue
            dfs.append(
                df.select(
                    pl.col("Date"),
                    pl.lit(symbol).alias("symbol"),
                    pl.col(field).alias("value"),
                )
            )

        if not dfs:
            raise DataNotFoundError("ticks", f"field={field}")

        # Concat and pivot
        combined = pl.concat(dfs)
        wide = combined.pivot(values="value", index="Date", on="symbol")

        # Align to trading calendar
        return self._align_to_calendar(wide, start, end)

    def _fetch_fundamentals_single(
        self, security_id: str, end: date, source: str = "raw"
    ) -> pl.DataFrame | None:
        """Fetch fundamentals for single security by security_id."""
        path = f"data/raw/fundamental/{security_id}/fundamental.parquet"
        date_filter = pl.col("as_of_date") <= end
        try:
            df = self._storage.read_parquet(path)
            if df.schema["as_of_date"] == pl.String:
                df = df.with_columns(pl.col("as_of_date").str.to_date())
            return df.filter(date_filter)
        except Exception:
            return None

    async def _fetch_fundamentals_async(
        self,
        securities: list[tuple[str, SecurityInfo]],
        end: date,
        source: str = "raw",
    ) -> list[tuple[str, pl.DataFrame]]:
        """Fetch fundamentals for multiple securities concurrently."""
        loop = asyncio.get_event_loop()
        futures: list[tuple[str, asyncio.Future[pl.DataFrame | None]]] = []

        for symbol, info in securities:
            future = loop.run_in_executor(
                self._executor, self._fetch_fundamentals_single, info.security_id, end, source
            )
            futures.append((symbol, future))

        results: list[tuple[str, pl.DataFrame]] = []
        for symbol, future in futures:
            df = await future
            if df is not None and len(df) > 0:
                results.append((symbol, df))

        return results

    def _extract_fundamental_values(
        self, df: pl.DataFrame, concept: str, symbol: str
    ) -> pl.DataFrame | None:
        """Extract fundamental concept values from DataFrame."""
        filtered = df.filter(pl.col("concept") == concept)
        if len(filtered) == 0:
            return None

        result = filtered.select(
            pl.col("as_of_date").alias("Date"),
            pl.lit(symbol).alias("symbol"),
            pl.col("value").cast(pl.Float64),
        )
        # Deduplicate: take first value per timestamp
        return result.group_by(["Date", "symbol"]).agg(pl.col("value").first())

    def fundamentals(
        self,
        symbols: Sequence[str] | str,
        concept: str,
        start: date | str | None = None,
        end: date | str | None = None,
        source: str | None = None,
    ) -> pl.DataFrame:
        """Get fundamental data as wide table.

        Args:
            symbols: Symbol(s) to fetch
            concept: Fundamental concept (e.g., "sales", "income", "assets")
            start: Start date
            end: End date
            source: Data source - "raw" for quarterly filings, "ttm" for trailing
                    twelve months. Defaults to "ttm" for duration concepts.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        start = start or date(2000, 1, 1)
        end = end or date.today() - timedelta(days=1)

        # Default to TTM for duration concepts, raw for balance sheet items
        if source is None:
            source = "ttm" if concept in DURATION_CONCEPTS else "raw"

        resolved = self._resolve_securities(symbols, as_of=start)
        if not resolved:
            raise DataNotFoundError("fundamentals", ", ".join(symbols))

        results = asyncio.run(self._fetch_fundamentals_async(resolved, end, source))
        if not results:
            raise DataNotFoundError("fundamentals", ", ".join(symbols))

        dfs: list[pl.DataFrame] = []
        for symbol, df in results:
            extracted = self._extract_fundamental_values(df, concept, symbol)
            if extracted is not None:
                dfs.append(extracted)

        if not dfs:
            raise DataNotFoundError("fundamentals", f"concept={concept}")

        combined = pl.concat(dfs)
        wide = combined.pivot(values="value", index="Date", on="symbol")

        # Align from earliest data to allow forward-fill into requested range
        earliest_val = wide["Date"].min()
        earliest = earliest_val if isinstance(earliest_val, date) else None
        align_start = min(earliest, start) if earliest else start
        aligned = self._align_to_calendar(wide, align_start, end, forward_fill=True)

        return aligned.filter(pl.col("Date") >= start)

    def metrics(
        self,
        symbols: Sequence[str] | str,
        metric: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pl.DataFrame:
        """Get derived metrics as wide table.

        Args:
            symbols: Symbol(s) to fetch
            metric: Metric name (e.g., "pe_ratio", "pb_ratio", "roe", "roa")
            start: Start date
            end: End date
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        start = start or date(2000, 1, 1)
        end = end or date.today() - timedelta(days=1)

        resolved = self._resolve_securities(symbols, as_of=start)
        if not resolved:
            raise DataNotFoundError("metrics", ", ".join(symbols))

        results = asyncio.run(self._fetch_fundamentals_async(resolved, end))
        if not results:
            raise DataNotFoundError("metrics", ", ".join(symbols))

        dfs: list[pl.DataFrame] = []
        for symbol, df in results:
            extracted = self._extract_metric_values(df, metric, symbol)
            if extracted is not None:
                dfs.append(extracted)

        if not dfs:
            raise DataNotFoundError("metrics", f"metric={metric}")

        combined = pl.concat(dfs)
        wide = combined.pivot(values="value", index="Date", on="symbol")

        # Align from earliest data to allow forward-fill into requested range
        earliest_val = wide["Date"].min()
        earliest = earliest_val if isinstance(earliest_val, date) else None
        align_start = min(earliest, start) if earliest else start
        aligned = self._align_to_calendar(wide, align_start, end, forward_fill=True)

        return aligned.filter(pl.col("Date") >= start)

    def _extract_metric_values(
        self, df: pl.DataFrame, metric: str, symbol: str
    ) -> pl.DataFrame | None:
        """Extract metric values from DataFrame, handling long or wide format."""
        is_long_format = "metric" in df.columns and "value" in df.columns

        if is_long_format:
            filtered = df.filter(pl.col("metric") == metric)
            if len(filtered) == 0:
                return None
            return filtered.select(
                pl.col("as_of_date").alias("Date"),
                pl.lit(symbol).alias("symbol"),
                pl.col("value"),
            )

        if metric in df.columns:
            return df.select(
                pl.col("as_of_date").alias("Date"),
                pl.lit(symbol).alias("symbol"),
                pl.col(metric).alias("value"),
            )

        return None

    def features(
        self,
        field: str,
        symbols: Sequence[str] | None = None,
        universe: str | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pl.DataFrame:
        """Get pre-built feature wide table.

        Args:
            field: Feature field name (e.g., "close", "assets", "returns")
            symbols: Symbol(s) to include (columns). If None, all columns returned.
            universe: Universe name to resolve symbols from (e.g., "top3000")
            start: Start date filter
            end: End date filter

        Returns:
            Wide DataFrame with timestamp + symbol columns
        """
        from quantdl.features.registry import VALID_FIELD_NAMES

        if field not in VALID_FIELD_NAMES:
            raise ValidationError(
                f"Unknown feature field: '{field}'. "
                f"Valid fields: {sorted(VALID_FIELD_NAMES)[:10]}..."
            )

        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        start = start or date(2000, 1, 1)
        end = end or date.today() - timedelta(days=1)

        # Resolve symbols
        if universe:
            symbol_list = self.universe(universe)
        elif symbols is not None:
            symbol_list = list(symbols)
        else:
            symbol_list = None

        # Read feature file
        path = f"data/features/{field}.arrow"
        df = self._storage.read_ipc(path)

        # Cast timestamp if needed
        if "Date" in df.columns and df.schema["Date"] == pl.String:
            df = df.with_columns(pl.col("Date").str.to_date())

        # Filter date range
        if "Date" in df.columns:
            df = df.filter(
                (pl.col("Date") >= start) & (pl.col("Date") <= end)
            )

        # Map security_id columns to symbols
        if symbol_list is not None:
            sid_to_symbol: dict[str, str] = {}
            for sym in symbol_list:
                info = self._security_master.resolve(sym, as_of=start)
                if info is not None:
                    sid_to_symbol[info.security_id] = sym

            # Select matching columns and rename to symbols
            sid_cols = [sid for sid in sid_to_symbol if sid in df.columns]
            if not sid_cols:
                raise DataNotFoundError("features", f"field={field}")
            df = df.select("Date", *sid_cols)
            rename_map = {sid: sid_to_symbol[sid] for sid in sid_cols}
            df = df.rename(rename_map)

        return df.sort("Date")

    def universe(self, name: str = "top3000") -> list[str]:
        """Load universe of symbols.

        Args:
            name: Universe name (default: "top3000")

        Returns:
            List of symbols in the universe
        """
        path = f"data/meta/universe/{name}.parquet"
        try:
            df = self._storage.read_parquet(path)
            return df["symbol"].to_list()
        except Exception as e:
            raise DataNotFoundError("universe", name) from e

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)

    def __enter__(self) -> QuantDLClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
