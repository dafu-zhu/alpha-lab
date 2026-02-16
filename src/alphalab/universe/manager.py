from pathlib import Path
import shutil
import time
import datetime as dt
import polars as pl
import logging
from typing import Dict, Optional
from alphalab.collection.alpaca_ticks import Ticks
from alphalab.universe.current import fetch_all_stocks
from alphalab.universe.historical import get_hist_universe_local
from alphalab.master.security_master import SecurityMaster
from alphalab.utils.logger import setup_logger

class UniverseManager:
    def __init__(
        self,
        security_master: Optional[SecurityMaster] = None
    ):
        self.store_dir = Path("data/meta/universe")
        self.store_dir.mkdir(parents=True, exist_ok=True)

        log_dir = Path("data/logs/symbols")
        self.logger = setup_logger("symbols", log_dir, logging.INFO, console_output=True)

        self.alpaca_fetcher = Ticks()

        if security_master is None:
            self.security_master = SecurityMaster()
        else:
            self.security_master = security_master

        # Cache for current symbols to avoid re-reading CSV
        self._current_symbols_cache: Optional[list[str]] = None

        # Cache for historical universe
        self._historical_cache: Dict[int, list[str]] = {}

    def get_current_symbols(self, refresh=False) -> list[str]:
        """
        Get the current list of common stocks from Nasdaq Trader.

        :param refresh: If True, fetches fresh data from Nasdaq. If False, reads from cache.
        """
        # If we have cached symbols and not refreshing, return cache
        if not refresh and self._current_symbols_cache is not None:
            return self._current_symbols_cache

        # Otherwise fetch from file/FTP (only one thread will do this)
        pd_df = fetch_all_stocks(refresh=refresh, logger=self.logger)
        if pd_df is not None and not pd_df.empty:
            symbols = pd_df['Ticker'].tolist()
        else:
            raise ValueError("Failed to fetch symbols from Nasdaq Trader.")

        self.logger.debug(f"Market Universe Size: {len(symbols)} tickers")

        # Cache the result
        self._current_symbols_cache = symbols
        return symbols

    def load_symbols_for_year(self, year: int, sym_type: str = "alpaca") -> list[str]:
        """
        Load symbol list for a given year with format support.
        Returns all stocks that were active at any point during the year.

        For year >= 2025: uses current Nasdaq FTP list.
        For year < 2025: uses local security master to filter by date range.

        :param year: Year (e.g., 2024)
        :param sym_type: "sec" or "alpaca" (default: "alpaca")
        :return: List of symbols in the specified format
        """
        try:
            # Check cache first
            cache_key = (year, sym_type)
            if cache_key in self._historical_cache:
                symbols = self._historical_cache[cache_key]
                self.logger.debug(f"Loaded {len(symbols)} symbols for {year} from cache (format={sym_type})")
                return symbols

            if year >= 2025:
                if self._current_symbols_cache is None:
                    df = fetch_all_stocks(with_filter=True, refresh=True, logger=self.logger)
                    self._current_symbols_cache = df['Ticker'].to_list()
                nasdaq_symbols = self._current_symbols_cache
                self.logger.debug(f"Using current ticker list for {year} ({len(nasdaq_symbols)} symbols)")
            else:
                # Use local security master for historical years
                df = get_hist_universe_local(year, security_master=self.security_master)
                nasdaq_symbols = df['Ticker'].to_list()
                self.logger.debug(f"Using local security master for {year} ({len(nasdaq_symbols)} symbols)")

            if sym_type == "alpaca":
                symbols = nasdaq_symbols
            elif sym_type == "sec":
                symbols = [sym.replace('.', '-') for sym in nasdaq_symbols]
            else:
                raise ValueError(f"Expected sym_type: 'sec' or 'alpaca', get {sym_type}")

            # Cache the result
            self._historical_cache[cache_key] = symbols

            self.logger.debug(f"Loaded {len(symbols)} symbols for {year} (format={sym_type})")
            return symbols

        except Exception as e:
            self.logger.error(f"Failed to load symbols for {year}: {e}", exc_info=True)
            return []

    def get_top_3000(
        self,
        day: str,
        symbols: list[str],
        source: str = 'alpaca',
        auto_resolve: bool = True
    ) -> list[str]:
        """
        Fetch recent data and calculate top 3000 most liquid stocks (in-memory).

        :param day: Date string in format "YYYY-MM-DD"
        :param symbols: List of symbols to analyze
        :param source: 'alpaca' (only supported source)
        :param auto_resolve: Unused, kept for API compatibility
        :return: List of top 3000 symbols ranked by average dollar volume
        """
        self.logger.info(f"Fetching recent data on {day} for {len(symbols)} symbols using alpaca...")

        recent_data = self.alpaca_fetcher.recent_daily_ticks(symbols, end_day=day)

        self.logger.info(f"Data fetched for {len(recent_data)} symbols, calculating liquidity...")

        # Calculate average dollar volume for each symbol
        liquidity_data = []
        for symbol, df in recent_data.items():
            if len(df) > 0:
                # Calculate average dollar volume: avg(close * volume)
                avg_dollar_vol = (df['close'] * df['volume']).mean()
                liquidity_data.append({
                    'symbol': symbol,
                    'avg_dollar_vol': avg_dollar_vol
                })
        
        if len(liquidity_data) == 0:
            self.logger.error("No symbols passed liquidity filter")
            return []

        # Create DataFrame and rank by liquidity
        liquidity_df = (
            pl.DataFrame(liquidity_data)
            .filter(pl.col('avg_dollar_vol') > 1000)
            .sort('avg_dollar_vol', descending=True)
            .head(3000)
        )

        # Log top and bottom stocks
        top_stock = liquidity_df.row(0)
        bottom_stock = liquidity_df.row(-1)
        self.logger.info(f"Top Liquid Stock: {top_stock[0]} (ADV: ${top_stock[1]:,.0f})")
        self.logger.info(f"Rank {len(liquidity_df)} Stock: {bottom_stock[0]} (ADV: ${bottom_stock[1]:,.0f})")

        result = liquidity_df['symbol'].to_list()

        return result

    def close(self) -> None:
        """Clean up resources."""
        pass
