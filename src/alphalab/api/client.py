"""AlphaLab client - main entry point for financial data access (local-only)."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from alphalab.api.data.calendar_master import CalendarMaster
from alphalab.api.data.security_master import SecurityMaster
from alphalab.api.exceptions import ConfigurationError, DataNotFoundError, ValidationError
from alphalab.api.backend import StorageBackend
from alphalab.api.types import SecurityInfo

if TYPE_CHECKING:
    from collections.abc import Sequence


class AlphaLabClient:
    """Client for accessing pre-built feature wide tables.

    Example:
        ```python
        client = AlphaLabClient(data_path="/path/to/data")

        # Look up a symbol
        info = client.lookup("AAPL")

        # Get feature data
        df = client.get("close", symbols=["AAPL", "MSFT"], start="2024-01-01")
        ```
    """

    def __init__(self, data_path: str, max_concurrency: int = 10) -> None:
        if not data_path:
            raise ConfigurationError("data_path is required")

        self._storage = StorageBackend(data_path=data_path)
        self._security_master = SecurityMaster(self._storage)
        self._calendar_master = CalendarMaster(self._storage)
        self._max_concurrency = max_concurrency

    @property
    def security_master(self) -> SecurityMaster:
        """Access security master for direct lookups."""
        return self._security_master

    @property
    def calendar_master(self) -> CalendarMaster:
        """Access calendar master for trading day lookups."""
        return self._calendar_master

    def lookup(self, identifier: str, as_of: date | None = None) -> SecurityInfo | None:
        """Look up symbol/identifier to SecurityInfo.

        Args:
            identifier: Symbol, CIK, or security_id
            as_of: Point-in-time date (auto for single-match symbols)

        Returns:
            SecurityInfo if found, None otherwise
        """
        return self._security_master.resolve(identifier, as_of)

    def get(
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
            Wide DataFrame with Date + symbol columns
        """
        from alphalab.features.registry import VALID_FIELD_NAMES

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

        # Cast date if needed
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

            sid_cols = [sid for sid in sid_to_symbol if sid in df.columns]
            if not sid_cols:
                raise DataNotFoundError("features", f"field={field}")
            df = df.select("Date", *sid_cols)
            rename_map = {sid: sid_to_symbol[sid] for sid in sid_cols}
            df = df.rename(rename_map)

        return df.sort("Date")

    def _extract_field_refs(self, expr: str) -> set[str]:
        """Parse expression AST to find field names that need auto-loading."""
        from alphalab.features.registry import VALID_FIELD_NAMES
        import alphalab.api.operators as operators

        tree = ast.parse(expr, mode="exec")
        names: set[str] = set()
        assigned: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)

        op_names = {n for n in dir(operators) if not n.startswith("_")}
        builtins = {"abs", "min", "max", "log", "sqrt", "sign"}
        return (names & VALID_FIELD_NAMES) - assigned - op_names - builtins

    def query(
        self,
        expr: str,
        symbols: Sequence[str] | None = None,
        universe: str | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pl.DataFrame:
        """Evaluate multi-line alpha expression with auto-field loading.

        Fields referenced in the expression are automatically loaded via get().
        Supports variable assignment and semicolons for multi-statement expressions.

        Args:
            expr: Alpha expression (e.g., "rank(-ts_delta(close, 5))")
            symbols: Symbol(s) to include
            universe: Universe name to resolve symbols from
            start: Start date filter
            end: End date filter

        Returns:
            Wide DataFrame with Date + symbol columns
        """
        import alphalab.api.operators as operators
        from alphalab.alpha.parser import alpha_query

        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        start = start or date(2000, 1, 1)
        end = end or date.today() - timedelta(days=1)

        # Auto-detect field references
        fields = self._extract_field_refs(expr)

        # Load each field
        variables: dict[str, pl.DataFrame] = {}
        for f in fields:
            variables[f] = self.get(f, symbols=symbols, universe=universe, start=start, end=end)

        result = alpha_query(expr, variables, ops=operators)
        return result.data

    def universe(self, name: str = "top3000") -> list[str]:
        """Load universe of symbols."""
        path = f"data/meta/universe/{name}.parquet"
        try:
            df = self._storage.read_parquet(path)
            return df["symbol"].to_list()
        except Exception as e:
            raise DataNotFoundError("universe", name) from e

    def close(self) -> None:
        """Clean up resources."""

    def __enter__(self) -> AlphaLabClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
