"""
Ticks Client for Querying Daily Ticks Data
===========================================

Provides user-friendly API for querying daily ticks with:
- Symbol-based queries (transparent security_id resolution)
- Session-based caching for performance
- Single file per security_id
"""

import datetime as dt
from typing import Optional, Dict, Tuple
import polars as pl
import logging
from botocore.exceptions import ClientError
from quantdl.storage.utils import NoSuchKeyError

from quantdl.master.security_master import SecurityMaster


class TicksClient:
    """
    Client for querying daily ticks data with symbol resolution.

    Features:
    - Transparent symbol → security_id resolution
    - Session-based cache for lookups
    - Date range filtering

    Storage: data/raw/ticks/daily/{security_id}/ticks.parquet
    """

    def __init__(
        self,
        s3_client,
        bucket_name: str,
        security_master: SecurityMaster,
        logger: Optional[logging.Logger] = None
    ):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.security_master = security_master
        self.logger = logger or logging.getLogger(__name__)

        # Session-based cache: (symbol, year) → security_id
        self._cache: Dict[Tuple[str, int], int] = {}

    def get_daily_ticks(
        self,
        symbol: str,
        year: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch daily ticks for a symbol and year.

        :param symbol: Ticker symbol (e.g., 'AAPL', 'BRK.B')
        :param year: Year to fetch
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        security_id = self._resolve_symbol(symbol, year)
        return self._fetch_by_security_id(security_id, start_date, end_date)

    def get_daily_ticks_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch full historical daily ticks for a symbol.

        :param symbol: Ticker symbol
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :return: Polars DataFrame with historical data
        """
        if end_date:
            year = int(end_date[:4])
        else:
            year = dt.date.today().year

        security_id = self._resolve_symbol(symbol, year)
        return self._fetch_by_security_id(security_id, start_date, end_date)

    def _resolve_symbol(self, symbol: str, year: int) -> int:
        """Resolve symbol to security_id with session caching."""
        cache_key = (symbol, year)

        if cache_key not in self._cache:
            date = f"{year}-12-31"
            self._cache[cache_key] = self.security_master.get_security_id(symbol, date)
            self.logger.debug(f"Resolved {symbol} ({year}) → security_id={self._cache[cache_key]}")

        return self._cache[cache_key]

    def _fetch_by_security_id(
        self,
        security_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """Fetch data from single ticks.parquet file."""
        s3_key = f"data/raw/ticks/daily/{security_id}/ticks.parquet"

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            df = pl.read_parquet(response['Body'])

            if start_date or end_date:
                df = self._apply_date_filter(df, start_date, end_date)

            return df

        except (ClientError, NoSuchKeyError) as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise ValueError(
                    f"No data found for security_id={security_id}. "
                    f"Check if data has been uploaded."
                )
            else:
                raise

    def _apply_date_filter(
        self,
        df: pl.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pl.DataFrame:
        """Apply date range filter to DataFrame."""
        if start_date:
            df = df.filter(pl.col('timestamp') >= start_date)
        if end_date:
            df = df.filter(pl.col('timestamp') <= end_date)
        return df

    def clear_cache(self):
        """Clear the symbol resolution cache."""
        self._cache.clear()
        self.logger.debug("Symbol resolution cache cleared")
