from pathlib import Path
import datetime as dt
import os
import polars as pl
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
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

    def _get_local_storage_path(self) -> Path:
        """Get LOCAL_STORAGE_PATH from environment."""
        base = os.getenv("LOCAL_STORAGE_PATH", "")
        if not base:
            raise ValueError("LOCAL_STORAGE_PATH environment variable required")
        return Path(os.path.expanduser(base))

    def _verify_ticks_exist(self, sample_size: int = 10) -> bool:
        """
        Verify that local ticks data exists by checking a sample of files.

        :param sample_size: Number of files to check
        :return: True if ticks data exists, False otherwise
        """
        try:
            storage = self._get_local_storage_path()
            ticks_dir = storage / "data" / "raw" / "ticks" / "daily"

            if not ticks_dir.exists():
                return False

            # Check if at least some security directories exist with parquet files
            security_dirs = [d for d in ticks_dir.iterdir() if d.is_dir()][:sample_size]
            if not security_dirs:
                return False

            # Check if parquet files exist in sample directories
            found = 0
            for sec_dir in security_dirs:
                ticks_file = sec_dir / "ticks.parquet"
                if ticks_file.exists():
                    found += 1

            # Require at least 3 files to consider data "exists"
            return found >= 3
        except (ValueError, OSError):
            return False

    def _calculate_adv_single(
        self,
        args: Tuple[int, str, List[str]]
    ) -> Tuple[int, str, float]:
        """
        Calculate ADV for a single security from local ticks.

        :param args: Tuple of (security_id, symbol, trading_days_str)
        :return: Tuple of (security_id, symbol, avg_dollar_volume)
        """
        security_id, symbol, trading_days_str = args
        try:
            storage = self._get_local_storage_path()
            ticks_path = storage / "data" / "raw" / "ticks" / "daily" / str(security_id) / "ticks.parquet"

            if not ticks_path.exists():
                return (security_id, symbol, 0.0)

            # Read parquet and filter to trading days
            df = pl.read_parquet(ticks_path)

            # Filter to trading days (timestamp column is string format YYYY-MM-DD)
            df = df.filter(pl.col("timestamp").is_in(trading_days_str))

            if len(df) == 0:
                return (security_id, symbol, 0.0)

            # Calculate ADV = mean(close * volume)
            adv = (df["close"] * df["volume"]).mean()
            return (security_id, symbol, adv if adv is not None else 0.0)

        except Exception as e:
            self.logger.debug(f"Error calculating ADV for {symbol} (sid={security_id}): {e}")
            return (security_id, symbol, 0.0)

    def _calculate_adv_from_local_ticks(
        self,
        symbols: List[str],
        end_date: dt.date,
        lookback_days: int = 60,
        max_workers: int = 8
    ) -> Dict[str, float]:
        """
        Calculate average dollar volume from local ticks storage.

        :param symbols: List of ticker symbols
        :param end_date: End date (month-end trading day)
        :param lookback_days: Number of trading days to look back (default 60)
        :param max_workers: Max parallel workers (default 8)
        :return: Dict mapping symbol -> average dollar volume
        """
        from alphalab.utils.calendar import TradingCalendar

        # Load trading calendar
        calendar = TradingCalendar()
        start_date = end_date - dt.timedelta(days=90)  # Buffer for ~60 trading days
        trading_days = calendar.get_trading_days(start_date, end_date)

        # Take last N trading days
        trading_days = trading_days[-lookback_days:]
        trading_days_str = [d.strftime("%Y-%m-%d") for d in trading_days]

        # Resolve symbols to security IDs
        end_date_str = end_date.strftime("%Y-%m-%d")
        args_list = []
        for sym in symbols:
            try:
                sid = self.security_master.get_security_id(sym, end_date_str)
                args_list.append((sid, sym, trading_days_str))
            except ValueError:
                continue

        # Calculate ADV in parallel
        results: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = executor.map(self._calculate_adv_single, args_list)
            for security_id, symbol, adv in futures:
                results[symbol] = adv

        return results

    def get_top_3000(
        self,
        day: str,
        symbols: list[str],
        source: str = 'alpaca',
        auto_resolve: bool = True
    ) -> list[str]:
        """
        Calculate top 3000 most liquid stocks.

        Prefers local ticks storage when available (faster, no API calls).
        Falls back to Alpaca API for real-time data.

        REQUIRES: For local source, daily ticks must be downloaded first (`alab --ticks`).

        :param day: Date string in format "YYYY-MM-DD"
        :param symbols: List of symbols to analyze
        :param source: 'local' (from ticks storage) or 'alpaca' (real-time API)
        :param auto_resolve: Unused, kept for API compatibility
        :return: List of top 3000 symbols ranked by average dollar volume
        """
        end_date = dt.datetime.strptime(day, "%Y-%m-%d").date()

        # Check if local ticks exist and use them preferentially
        # Also check that security_master is available (needed for symbol resolution)
        use_local = hasattr(self, 'security_master') and self._verify_ticks_exist()

        if use_local:
            self.logger.debug(f"Calculating ADV from local ticks for {len(symbols)} symbols...")
            adv_map = self._calculate_adv_from_local_ticks(symbols, end_date)
        else:
            self.logger.debug(f"Fetching recent data on {day} for {len(symbols)} symbols using alpaca...")
            recent_data = self.alpaca_fetcher.recent_daily_ticks(symbols, end_day=day)
            self.logger.debug(f"Data fetched for {len(recent_data)} symbols, calculating liquidity...")

            adv_map = {}
            for symbol, df in recent_data.items():
                if len(df) > 0:
                    adv_map[symbol] = (df['close'] * df['volume']).mean()

        # Convert to list for ranking
        liquidity_data = [
            {'symbol': sym, 'avg_dollar_vol': adv}
            for sym, adv in adv_map.items()
            if adv > 0
        ]

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
        if len(liquidity_df) > 0:
            top_stock = liquidity_df.row(0)
            bottom_stock = liquidity_df.row(-1)
            self.logger.debug(f"Top Liquid Stock: {top_stock[0]} (ADV: ${top_stock[1]:,.0f})")
            self.logger.debug(f"Rank {len(liquidity_df)} Stock: {bottom_stock[0]} (ADV: ${bottom_stock[1]:,.0f})")

        result = liquidity_df['symbol'].to_list()

        return result

    def close(self) -> None:
        """Clean up resources."""
        pass
