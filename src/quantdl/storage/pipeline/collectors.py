"""
Data collection functionality for market data.

This module handles fetching market data from various sources:
- Daily ticks from Alpaca API
- Minute ticks from Alpaca API
- Fundamental data from SEC EDGAR API
"""

import datetime as dt
import time
import logging
import threading
from collections import OrderedDict
from typing import List, Dict, Optional
from pathlib import Path
import yaml
import polars as pl

from quantdl.collection.models import DataCollector
from quantdl.collection.fundamental import Fundamental
from quantdl.universe.current import fetch_all_stocks
from quantdl.utils.logger import setup_logger
from quantdl.utils.mapping import align_calendar


class TicksDataCollector(DataCollector):
    """Handles tick data collection (daily and minute resolution)."""

    def __init__(
        self,
        alpaca_ticks,
        alpaca_headers: dict,
        logger: logging.Logger,
        crsp_ticks=None,
    ):
        super().__init__(logger=logger)
        self.alpaca_ticks = alpaca_ticks
        self.alpaca_headers = alpaca_headers

    def _normalize_daily_df(self, df: pl.DataFrame) -> pl.DataFrame:
        if len(df) > 0:
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                df = df.with_columns([pl.lit(None).alias(col) for col in missing_cols])

            df = df.with_columns([
                pl.col('timestamp').cast(pl.Utf8),
                pl.col('open').cast(pl.Float64).round(4),
                pl.col('high').cast(pl.Float64).round(4),
                pl.col('low').cast(pl.Float64).round(4),
                pl.col('close').cast(pl.Float64).round(4),
                pl.col('volume').cast(pl.Int64)
            ]).sort('timestamp')

        return df

    def _bars_to_daily_df(self, bars: List[Dict]) -> pl.DataFrame:
        if not bars:
            return pl.DataFrame()

        from dataclasses import asdict
        parsed_ticks = self.alpaca_ticks.parse_ticks(bars)
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        df = pl.DataFrame(ticks_data).with_columns([
            pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S').dt.date(),
            pl.col('open').cast(pl.Float64),
            pl.col('high').cast(pl.Float64),
            pl.col('low').cast(pl.Float64),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Int64)
        ]).select(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        return df

    def collect_daily_ticks_year(self, sym: str, year: int) -> pl.DataFrame:
        """
        Fetch daily ticks for entire year from Alpaca and return as Polars DataFrame.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            bars_map = self.alpaca_ticks.fetch_daily_year_bulk(
                symbols=[sym],
                year=year,
                adjusted=True
            )
            df = self._bars_to_daily_df(bars_map.get(sym, []))
        except Exception as e:
            self.logger.warning(f"Failed to fetch {sym} for {year}: {e}")
            return pl.DataFrame()

        return self._normalize_daily_df(df)

    def collect_daily_ticks_year_bulk(self, symbols: List[str], year: int) -> Dict[str, pl.DataFrame]:
        """
        Bulk fetch daily ticks for a full year and return a mapping of symbol -> DataFrame.
        Always uses Alpaca bulk fetch.
        """
        symbol_bars = self.alpaca_ticks.fetch_daily_year_bulk(symbols, year, adjusted=True)
        result = {}
        for sym in symbols:
            df = self._bars_to_daily_df(symbol_bars.get(sym, []))
            result[sym] = self._normalize_daily_df(df)
        return result

    def collect_daily_ticks_range_bulk(
        self, symbols: List[str], start: str, end: str
    ) -> Dict[str, pl.DataFrame]:
        """
        Bulk fetch daily ticks for an arbitrary date range.

        :param symbols: List of symbols in Alpaca format
        :param start: Start date (YYYY-MM-DD)
        :param end: End date (YYYY-MM-DD)
        :return: Dict mapping symbol -> normalized DataFrame
        """
        start_str = f"{start}T00:00:00Z"
        end_str = f"{end}T23:59:59Z"
        symbol_bars = self.alpaca_ticks.fetch_daily_range_bulk(
            symbols, start_str, end_str, adjusted=True
        )
        result = {}
        for sym in symbols:
            df = self._bars_to_daily_df(symbol_bars.get(sym, []))
            result[sym] = self._normalize_daily_df(df)
        return result

    def collect_daily_ticks_month(
        self,
        sym: str,
        year: int,
        month: int,
        year_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Fetch daily ticks for a specific month from appropriate source and return as Polars DataFrame.
        Directly fetches only the requested month from the source API (no year-level fetch).
        If year_df is provided, filters the month from that DataFrame instead of refetching.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if year_df is not None:
            if len(year_df) == 0:
                return pl.DataFrame()
            month_prefix = f"{year}-{month:02d}"
            df = year_df.filter(
                pl.col('timestamp').cast(pl.Utf8).str.slice(0, 7).eq(month_prefix)
            )
            if len(df) == 0:
                return pl.DataFrame()
        else:
            try:
                symbol_bars = self.alpaca_ticks.fetch_daily_month_bulk(
                    symbols=[sym],
                    year=year,
                    month=month,
                    adjusted=True
                )
                df = self._bars_to_daily_df(symbol_bars.get(sym, []))
            except Exception as e:
                self.logger.warning(f"Failed to fetch {sym} for {year}-{month:02d}: {e}")
                return pl.DataFrame()

        return self._normalize_daily_df(df)

    def collect_daily_ticks_month_bulk(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> Dict[str, pl.DataFrame]:
        """
        Bulk fetch daily ticks for a specific month and return a mapping of symbol -> DataFrame.
        Always uses Alpaca bulk fetch.
        """
        symbol_bars = self.alpaca_ticks.fetch_daily_month_bulk(
            symbols=symbols,
            year=year,
            month=month,
            sleep_time=sleep_time,
            adjusted=True
        )
        result = {}
        for sym in symbols:
            df = self._bars_to_daily_df(symbol_bars.get(sym, []))
            result[sym] = self._normalize_daily_df(df)
        return result



class FundamentalDataCollector(DataCollector):
    """Handles fundamental data collection from SEC EDGAR."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        sec_rate_limiter=None,
        fundamental_cache: Optional[OrderedDict] = None,
        fundamental_cache_lock: Optional[threading.Lock] = None,
        fundamental_cache_size: int = 128
    ):
        # Create logger if not provided (for backward compatibility)
        if logger is None:
            logger = setup_logger(
                name="storage.fundamental_data_collector",
                log_dir="data/logs/fundamental",
                level=logging.INFO
            )
        super().__init__(logger=logger)
        self.sec_rate_limiter = sec_rate_limiter

        # Use shared cache or create new one
        if fundamental_cache is not None and fundamental_cache_lock is not None:
            self._fundamental_cache = fundamental_cache
            self._fundamental_cache_lock = fundamental_cache_lock
            self._fundamental_cache_size = fundamental_cache_size
        else:
            self._fundamental_cache_size = max(int(fundamental_cache_size), 0)
            self._fundamental_cache = OrderedDict()
            self._fundamental_cache_lock = threading.Lock()

    def _load_concepts(
        self,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> List[str]:
        if concepts is not None:
            return concepts
        if config_path is None:
            config_path = Path("configs/sec_mapping.yaml")

        with open(config_path) as f:
            mappings = yaml.safe_load(f)
            return list(mappings.keys())

    def _get_or_create_fundamental(
        self,
        cik: str,
        symbol: Optional[str] = None
    ) -> Fundamental:
        """
        Get cached Fundamental object or create new one.
        Avoids redundant SEC API calls by reusing fetched data.

        :param cik: Company CIK number
        :param symbol: Optional symbol for logging
        :return: Fundamental object (cached or newly created)
        """
        if self._fundamental_cache_size <= 0:
            return Fundamental(
                cik=cik,
                symbol=symbol,
                rate_limiter=self.sec_rate_limiter
            )

        with self._fundamental_cache_lock:
            cached = self._fundamental_cache.get(cik)
            if cached is not None:
                self._fundamental_cache.move_to_end(cik)
                return cached

        created = Fundamental(
            cik=cik,
            symbol=symbol,
            rate_limiter=self.sec_rate_limiter
        )

        with self._fundamental_cache_lock:
            cached = self._fundamental_cache.get(cik)
            if cached is not None:
                self._fundamental_cache.move_to_end(cik)
                return cached
            self._fundamental_cache[cik] = created
            if len(self._fundamental_cache) > self._fundamental_cache_size:
                self._fundamental_cache.popitem(last=False)
        return created

    def collect_fundamental_long(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        """
        Fetch long-format fundamental data for a filing date range.

        Returns columns:
        [symbol, as_of_date, accn, form, concept, value, start, end, frame, is_instant]
        """
        try:
            concepts = self._load_concepts(concepts, config_path)

            # Use cached Fundamental object to avoid redundant API calls
            fnd = self._get_or_create_fundamental(cik=cik, symbol=symbol)
            start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

            records = []
            concepts_found = []
            concepts_missing = []

            for concept in concepts:
                try:
                    dps = fnd.get_concept_data(
                        concept,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if dps:
                        concepts_found.append(concept)
                        for dp in dps:
                            if not (start_dt <= dp.timestamp <= end_dt):
                                continue
                            records.append(
                                {
                                    "symbol": symbol,
                                    "as_of_date": dp.timestamp.isoformat(),
                                    "accn": dp.accn,
                                    "form": dp.form,
                                    "concept": concept,
                                    "value": dp.value,
                                    "start": dp.start_date.isoformat() if dp.start_date else None,
                                    "end": dp.end_date.isoformat(),
                                    "frame": dp.frame,
                                    "is_instant": dp.is_instant,
                                }
                            )
                    else:
                        concepts_missing.append(concept)
                except Exception as e:
                    self.logger.debug(f"Failed to extract concept '{concept}' for CIK {cik}: {e}")
                    concepts_missing.append(concept)

            if not records:
                self.logger.warning(
                    f"No fundamental data found for CIK {cik} ({symbol}) "
                    f"from {start_date} to {end_date}"
                )
                return pl.DataFrame()

            self.logger.debug(
                f"CIK {cik} ({symbol}): {len(concepts_found)}/{len(concepts)} concepts available "
                f"({len(concepts_missing)} missing)"
            )

            return pl.DataFrame(records)

        except Exception as e:
            self.logger.error(
                f"Failed to collect fundamental data for CIK {cik} ({symbol}) "
                f"from {start_date} to {end_date}: {e}"
            )
            return pl.DataFrame()



class UniverseDataCollector(DataCollector):
    """Handles stock universe data collection."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Create logger if not provided (for backward compatibility)
        if logger is None:
            logger = setup_logger(
                name="storage.universe_data_collector",
                log_dir="data/logs/symbols",
                level=logging.INFO
            )
        super().__init__(logger=logger)

    def collect_current_universe(self, with_filter: bool = True, refresh: bool = False):
        """Fetch current universe of stocks."""
        return fetch_all_stocks(with_filter=with_filter, refresh=refresh)


class DataCollectors:
    """
    Orchestrator for data collection from various market data sources.
    Delegates to specialized collector instances.
    """

    def __init__(
        self,
        alpaca_ticks,
        alpaca_headers: dict,
        logger: logging.Logger,
        sec_rate_limiter=None,
        fundamental_cache_size: int = 128,
        crsp_ticks=None,
    ):
        """
        Initialize data collectors.

        :param alpaca_ticks: Alpaca Ticks instance
        :param alpaca_headers: Headers for Alpaca API requests
        :param logger: Logger instance
        :param sec_rate_limiter: Optional rate limiter for SEC API calls
        :param fundamental_cache_size: Size of LRU cache for Fundamental objects
        :param crsp_ticks: Deprecated, ignored (kept for backward compatibility)
        """
        self.logger = logger

        # Shared fundamental cache (managed centrally)
        self._fundamental_cache_size = max(int(fundamental_cache_size), 0)
        self._fundamental_cache: "OrderedDict[str, Fundamental]" = OrderedDict()
        self._fundamental_cache_lock = threading.Lock()

        # Create specialized collectors with dependency injection
        self.ticks_collector = TicksDataCollector(
            alpaca_ticks=alpaca_ticks,
            alpaca_headers=alpaca_headers,
            logger=logger,
        )

        self.fundamental_collector = FundamentalDataCollector(
            logger=logger,
            sec_rate_limiter=sec_rate_limiter,
            fundamental_cache=self._fundamental_cache,
            fundamental_cache_lock=self._fundamental_cache_lock,
            fundamental_cache_size=fundamental_cache_size
        )

        self.universe_collector = UniverseDataCollector(logger=logger)

    # Delegation methods for fundamental collection
    def _load_concepts(
        self,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> List[str]:
        return self.fundamental_collector._load_concepts(concepts, config_path)

    def collect_fundamental_long(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        return self.fundamental_collector.collect_fundamental_long(
            cik, start_date, end_date, symbol, concepts, config_path
        )

    # Delegation methods for ticks collection
    def collect_daily_ticks_range_bulk(
        self, symbols: List[str], start: str, end: str
    ) -> Dict[str, pl.DataFrame]:
        return self.ticks_collector.collect_daily_ticks_range_bulk(symbols, start, end)

    def collect_daily_ticks_year(self, sym: str, year: int) -> pl.DataFrame:
        return self.ticks_collector.collect_daily_ticks_year(sym, year)

    def collect_daily_ticks_month(
        self,
        sym: str,
        year: int,
        month: int,
        year_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        return self.ticks_collector.collect_daily_ticks_month(sym, year, month, year_df=year_df)

    def collect_daily_ticks_year_bulk(self, symbols: List[str], year: int) -> Dict[str, pl.DataFrame]:
        return self.ticks_collector.collect_daily_ticks_year_bulk(symbols, year)

    def collect_daily_ticks_month_bulk(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> Dict[str, pl.DataFrame]:
        return self.ticks_collector.collect_daily_ticks_month_bulk(symbols, year, month, sleep_time)

