"""
Fundamental data upload handlers.

Handles raw fundamental, TTM, and derived metrics uploads.
"""

from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple

from tqdm import tqdm
import polars as pl

from quantdl.storage.handlers.base import BaseHandler

if TYPE_CHECKING:
    from quantdl.storage.pipeline import DataPublishers, DataCollectors, Validator
    from quantdl.storage.utils import CIKResolver, RateLimiter
    from quantdl.universe.manager import UniverseManager


class FundamentalHandler(BaseHandler):
    """
    Handles raw fundamental data upload.

    Storage: data/raw/fundamental/{cik}/fundamental.parquet
    """

    def __init__(
        self,
        data_publishers: DataPublishers,
        data_collectors: DataCollectors,
        cik_resolver: CIKResolver,
        universe_manager: UniverseManager,
        validator: Validator,
        sec_rate_limiter: RateLimiter,
        logger: logging.Logger
    ):
        super().__init__(logger)
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.cik_resolver = cik_resolver
        self.universe_manager = universe_manager
        self.validator = validator
        self.sec_rate_limiter = sec_rate_limiter

    def upload(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Upload raw fundamental data for all symbols in date range."""
        start_time = time.time()

        # Build symbol list and prefetch CIKs
        symbols_with_cik, cik_map, prefetch_time = self._prepare_symbols(start_date, end_date)

        if not symbols_with_cik:
            self.logger.warning("No symbols with CIKs found, skipping fundamental upload")
            return self._build_result(start_time, prefetch_time, 0, 0)

        total = len(symbols_with_cik)
        self.logger.info(f"Step 3/3: Fetching fundamental data for {total} symbols...")
        fetch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_symbol, sym, start_date, end_date, overwrite, cik_map.get(sym)
                ): sym
                for sym in symbols_with_cik
            }

            pbar = tqdm(as_completed(futures), total=total, desc="Fundamental", unit="sym")
            for future in pbar:
                result = future.result()
                self._update_stats(result)
                pbar.set_postfix(
                    ok=self.stats['success'], fail=self.stats['failed'],
                    skip=self.stats['skipped'], cancel=self.stats['canceled']
                )

        return self._build_result(start_time, prefetch_time, time.time() - fetch_start, total)

    def _process_symbol(
        self,
        sym: str,
        start_date: str,
        end_date: str,
        overwrite: bool,
        cik: Optional[str]
    ) -> Dict[str, Any]:
        """Process fundamental data for a single symbol."""
        if cik is None:
            reference_year = int(end_date[:4])
            cik = self.cik_resolver.get_cik(sym, f"{reference_year}-06-30", year=reference_year)

        if not overwrite and self.validator.data_exists(sym, 'fundamental', cik=cik):
            self.logger.debug(f"Fundamental for {cik} exists, refreshing date range")

        return self.publishers.publish_fundamental(
            sym=sym,
            start_date=start_date,
            end_date=end_date,
            cik=cik,
            sec_rate_limiter=self.sec_rate_limiter
        )

    def _prepare_symbols(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[List[str], Dict[str, str], float]:
        """Build symbol list and prefetch CIKs."""
        start_year, end_year = int(start_date[:4]), int(end_date[:4])

        # Build symbol -> reference year mapping
        symbol_ref_year = {}
        for year in range(start_year, end_year + 1):
            for sym in self.universe_manager.load_symbols_for_year(year, sym_type='alpaca'):
                if sym not in symbol_ref_year:
                    symbol_ref_year[sym] = year

        total = len(symbol_ref_year)
        self.logger.info(
            f"Starting fundamental upload for {total} symbols "
            f"from {start_date} to {end_date}"
        )
        self.logger.info("Storage: data/raw/fundamental/{cik}/fundamental.parquet")

        # Batch prefetch CIKs
        self.logger.info(f"Step 1/3: Pre-fetching CIKs for {total} symbols...")
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_syms = [s for s, y in symbol_ref_year.items() if y == year]
            if year_syms:
                cik_map.update(self.cik_resolver.batch_prefetch_ciks(year_syms, year, batch_size=100))
        prefetch_time = time.time() - prefetch_start
        self.logger.info(f"CIK pre-fetch completed in {prefetch_time:.1f}s")

        # Filter symbols with valid CIKs
        self.logger.info("Step 2/3: Filtering symbols with valid CIKs...")
        symbols_with_cik = [s for s in symbol_ref_year if cik_map.get(s)]
        symbols_without = [s for s in symbol_ref_year if not cik_map.get(s)]

        self.logger.info(
            f"Symbol filtering: {len(symbols_with_cik)}/{total} have CIKs, "
            f"{len(symbols_without)} non-SEC filers"
        )
        if symbols_without and len(symbols_without) <= 30:
            self.logger.info(f"Non-SEC filers: {sorted(symbols_without)}")

        return symbols_with_cik, cik_map, prefetch_time

    def _update_stats(self, result: Dict[str, Any]) -> None:
        """Update statistics from result."""
        status = result.get('status', 'failed')
        if status == 'success':
            self.stats['success'] += 1
        elif status == 'canceled':
            self.stats['canceled'] += 1
        elif status == 'skipped':
            self.stats['skipped'] += 1
        else:
            self.stats['failed'] += 1

    def _build_result(
        self,
        start_time: float,
        prefetch_time: float,
        fetch_time: float,
        total: int
    ) -> Dict[str, Any]:
        """Build result dictionary."""
        total_time = time.time() - start_time
        avg_rate = total / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"Fundamental upload completed in {total_time:.1f}s: "
            f"{self.stats['success']} success, {self.stats['failed']} failed, "
            f"{self.stats['skipped']} skipped, {self.stats['canceled']} canceled"
        )
        self.logger.info(
            f"Performance: CIK fetch={prefetch_time:.1f}s, "
            f"Data fetch={fetch_time:.1f}s, Avg rate={avg_rate:.2f} sym/sec"
        )

        return {
            **self.stats,
            'total_time': total_time,
            'prefetch_time': prefetch_time,
            'fetch_time': fetch_time,
            'avg_rate': avg_rate
        }


class TTMHandler(BaseHandler):
    """
    Handles TTM fundamental data upload.

    Storage: data/derived/features/fundamental/{cik}/ttm.parquet
    """

    def __init__(
        self,
        data_publishers: DataPublishers,
        data_collectors: DataCollectors,
        cik_resolver: CIKResolver,
        universe_manager: UniverseManager,
        validator: Validator,
        sec_rate_limiter: RateLimiter,
        logger: logging.Logger
    ):
        super().__init__(logger)
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.cik_resolver = cik_resolver
        self.universe_manager = universe_manager
        self.validator = validator
        self.sec_rate_limiter = sec_rate_limiter

    def upload(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Upload TTM fundamental data for all symbols in date range."""
        start_time = time.time()

        symbols_with_cik, cik_map, prefetch_time = self._prepare_symbols(start_date, end_date)

        if not symbols_with_cik:
            self.logger.warning("No symbols with CIKs found, skipping TTM upload")
            return {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

        total = len(symbols_with_cik)
        self.logger.info(f"Step 3/3: Computing TTM data for {total} symbols...")
        fetch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_symbol, sym, start_date, end_date, overwrite, cik_map.get(sym)
                ): sym
                for sym in symbols_with_cik
            }

            pbar = tqdm(as_completed(futures), total=total, desc="TTM", unit="sym")
            for future in pbar:
                result = future.result()
                self._update_stats(result)
                pbar.set_postfix(
                    ok=self.stats['success'], fail=self.stats['failed'],
                    skip=self.stats['skipped'], cancel=self.stats['canceled']
                )

        fetch_time = time.time() - fetch_start
        total_time = time.time() - start_time
        avg_rate = total / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"TTM upload completed in {total_time:.1f}s: "
            f"{self.stats['success']} success, {self.stats['failed']} failed, "
            f"{self.stats['skipped']} skipped"
        )

        return {**self.stats, 'total_time': total_time, 'avg_rate': avg_rate}

    def _process_symbol(
        self,
        sym: str,
        start_date: str,
        end_date: str,
        overwrite: bool,
        cik: Optional[str]
    ) -> Dict[str, Any]:
        """Process TTM data for a single symbol."""
        if cik is None:
            reference_year = int(end_date[:4])
            cik = self.cik_resolver.get_cik(sym, f"{reference_year}-06-30", year=reference_year)

        if not overwrite and self.validator.data_exists(sym, 'ttm', data_tier='derived', cik=cik):
            self.logger.debug(f"TTM for {sym} exists, refreshing date range")

        return self.publishers.publish_ttm_fundamental(
            sym=sym,
            start_date=start_date,
            end_date=end_date,
            cik=cik,
            sec_rate_limiter=self.sec_rate_limiter
        )

    def _prepare_symbols(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[List[str], Dict[str, str], float]:
        """Build symbol list and prefetch CIKs."""
        start_year, end_year = int(start_date[:4]), int(end_date[:4])

        symbol_ref_year = {}
        for year in range(start_year, end_year + 1):
            for sym in self.universe_manager.load_symbols_for_year(year, sym_type='alpaca'):
                if sym not in symbol_ref_year:
                    symbol_ref_year[sym] = year

        total = len(symbol_ref_year)
        self.logger.info(
            f"Starting TTM upload for {total} symbols from {start_date} to {end_date}"
        )
        self.logger.info("Storage: data/derived/features/fundamental/{cik}/ttm.parquet")

        self.logger.info(f"Step 1/3: Pre-fetching CIKs...")
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_syms = [s for s, y in symbol_ref_year.items() if y == year]
            if year_syms:
                cik_map.update(self.cik_resolver.batch_prefetch_ciks(year_syms, year, batch_size=100))
        prefetch_time = time.time() - prefetch_start

        self.logger.info("Step 2/3: Filtering symbols...")
        symbols_with_cik = [s for s in symbol_ref_year if cik_map.get(s)]
        self.logger.info(f"Found {len(symbols_with_cik)}/{total} symbols with CIKs")

        return symbols_with_cik, cik_map, prefetch_time

    def _update_stats(self, result: Dict[str, Any]) -> None:
        status = result.get('status', 'failed')
        if status == 'success':
            self.stats['success'] += 1
        elif status == 'canceled':
            self.stats['canceled'] += 1
        elif status == 'skipped':
            self.stats['skipped'] += 1
        else:
            self.stats['failed'] += 1


class DerivedHandler(BaseHandler):
    """
    Handles derived fundamental metrics upload.

    Storage: data/derived/features/fundamental/{cik}/metrics.parquet
    """

    def __init__(
        self,
        data_publishers: DataPublishers,
        data_collectors: DataCollectors,
        cik_resolver: CIKResolver,
        universe_manager: UniverseManager,
        validator: Validator,
        logger: logging.Logger
    ):
        super().__init__(logger)
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.cik_resolver = cik_resolver
        self.universe_manager = universe_manager
        self.validator = validator

    def upload(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Upload derived fundamental data for all symbols in date range."""
        start_time = time.time()

        symbols_with_cik, cik_map, prefetch_time = self._prepare_symbols(start_date, end_date)

        if not symbols_with_cik:
            self.logger.warning("No symbols with CIKs found, skipping derived upload")
            return {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

        total = len(symbols_with_cik)
        self.logger.info(f"Step 3/3: Computing derived metrics for {total} symbols...")
        fetch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_symbol, sym, start_date, end_date, overwrite, cik_map.get(sym)
                ): sym
                for sym in symbols_with_cik
            }

            pbar = tqdm(as_completed(futures), total=total, desc="Derived", unit="sym")
            for future in pbar:
                result = future.result()
                self._update_stats(result)
                pbar.set_postfix(
                    ok=self.stats['success'], fail=self.stats['failed'],
                    skip=self.stats['skipped'], cancel=self.stats['canceled']
                )

        fetch_time = time.time() - fetch_start
        total_time = time.time() - start_time
        avg_rate = total / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"Derived upload completed in {total_time:.1f}s: "
            f"{self.stats['success']} success, {self.stats['failed']} failed, "
            f"{self.stats['skipped']} skipped"
        )

        return {**self.stats, 'total_time': total_time, 'avg_rate': avg_rate}

    def _process_symbol(
        self,
        sym: str,
        start_date: str,
        end_date: str,
        overwrite: bool,
        cik: Optional[str]
    ) -> Dict[str, Any]:
        """Process derived metrics for a single symbol."""
        if cik is None:
            reference_year = int(end_date[:4])
            cik = self.cik_resolver.get_cik(sym, f"{reference_year}-06-30", year=reference_year)

        if cik is None:
            return {'symbol': sym, 'cik': None, 'status': 'skipped', 'error': f'No CIK for {sym}'}

        if not overwrite and self.validator.data_exists(sym, 'fundamental', data_tier='derived', cik=cik):
            self.logger.debug(f"Derived for {sym} exists, refreshing date range")

        # Collect derived metrics
        derived_df, reason = self.collectors.collect_derived_long(
            cik=cik,
            start_date=start_date,
            end_date=end_date,
            symbol=sym
        )

        if len(derived_df) == 0:
            return {'symbol': sym, 'cik': cik, 'status': 'skipped', 'error': reason or 'No derived data'}

        return self.publishers.publish_derived_fundamental(
            sym=sym,
            start_date=start_date,
            end_date=end_date,
            derived_df=derived_df,
            cik=cik
        )

    def _prepare_symbols(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[List[str], Dict[str, str], float]:
        """Build symbol list and prefetch CIKs."""
        start_year, end_year = int(start_date[:4]), int(end_date[:4])

        symbol_ref_year = {}
        for year in range(start_year, end_year + 1):
            symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
            if not symbols:
                self.logger.warning(f"No symbols for {year}")
                continue
            for sym in symbols:
                if sym not in symbol_ref_year:
                    symbol_ref_year[sym] = year

        total = len(symbol_ref_year)
        self.logger.info(
            f"Starting derived upload for {total} symbols from {start_date} to {end_date}"
        )
        self.logger.info("Storage: data/derived/features/fundamental/{cik}/metrics.parquet")

        self.logger.info(f"Step 1/3: Pre-fetching CIKs...")
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_syms = [s for s, y in symbol_ref_year.items() if y == year]
            if year_syms:
                cik_map.update(self.cik_resolver.batch_prefetch_ciks(year_syms, year, batch_size=100))
        prefetch_time = time.time() - prefetch_start

        self.logger.info("Step 2/3: Filtering symbols...")
        symbols_with_cik = [s for s in symbol_ref_year if cik_map.get(s)]
        self.logger.info(f"Found {len(symbols_with_cik)}/{total} symbols with CIKs")

        return symbols_with_cik, cik_map, prefetch_time

    def _update_stats(self, result: Dict[str, Any]) -> None:
        status = result.get('status', 'failed')
        if status == 'success':
            self.stats['success'] += 1
        elif status == 'canceled':
            self.stats['canceled'] += 1
        elif status == 'skipped':
            self.stats['skipped'] += 1
        else:
            self.stats['failed'] += 1
