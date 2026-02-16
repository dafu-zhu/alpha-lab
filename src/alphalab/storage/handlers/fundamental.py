"""
Fundamental data upload handler.

Handles raw fundamental data uploads.
"""

from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple

from tqdm import tqdm
import polars as pl

from alphalab.storage.handlers.base import BaseHandler

if TYPE_CHECKING:
    from alphalab.storage.pipeline import DataPublishers, DataCollectors, Validator
    from alphalab.storage.utils import CIKResolver, RateLimiter
    from alphalab.universe.manager import UniverseManager
    from alphalab.master.security_master import SecurityMaster


class FundamentalHandler(BaseHandler):
    """
    Handles raw fundamental data upload.

    Storage: data/raw/fundamental/{security_id}/fundamental.parquet
    """

    def __init__(
        self,
        data_publishers: DataPublishers,
        data_collectors: DataCollectors,
        cik_resolver: CIKResolver,
        security_master: SecurityMaster,
        universe_manager: UniverseManager,
        validator: Validator,
        sec_rate_limiter: RateLimiter,
        logger: logging.Logger
    ):
        super().__init__(logger)
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.cik_resolver = cik_resolver
        self.security_master = security_master
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

        # Build symbol list, prefetch CIKs, and resolve security IDs
        symbols_with_cik, cik_map, security_id_cache, prefetch_time = self._prepare_symbols(start_date, end_date)

        if not symbols_with_cik:
            self.logger.warning("No symbols with CIKs found, skipping fundamental upload")
            return self._build_result(start_time, prefetch_time, 0, 0)

        total = len(symbols_with_cik)
        self.logger.debug(f"Step 4/4: Fetching fundamental data for {total} symbols...")
        fetch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_symbol, sym, start_date, end_date, overwrite,
                    cik_map.get(sym), security_id_cache.get(sym)
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
        cik: Optional[str],
        security_id: Optional[int]
    ) -> Dict[str, Any]:
        """Process fundamental data for a single symbol."""
        if cik is None:
            reference_year = int(end_date[:4])
            cik = self.cik_resolver.get_cik(sym, f"{reference_year}-06-30", year=reference_year)

        if security_id is None:
            self.logger.debug(f"Skipping {sym}: no security_id")
            return {'symbol': sym, 'status': 'skipped', 'error': 'No security_id'}

        if not overwrite and self.validator.data_exists(sym, 'fundamental', security_id=security_id):
            self.logger.debug(f"Fundamental for {sym} (sid={security_id}) exists, refreshing date range")

        return self.publishers.publish_fundamental(
            sym=sym,
            start_date=start_date,
            end_date=end_date,
            cik=cik,
            security_id=security_id,
            sec_rate_limiter=self.sec_rate_limiter
        )

    def _prepare_symbols(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[int]], float]:
        """Build symbol list, prefetch CIKs, and resolve security IDs."""
        start_year, end_year = int(start_date[:4]), int(end_date[:4])

        # Build symbol -> reference year mapping
        symbol_ref_year = {}
        for year in range(start_year, end_year + 1):
            for sym in self.universe_manager.load_symbols_for_year(year, sym_type='alpaca'):
                if sym not in symbol_ref_year:
                    symbol_ref_year[sym] = year

        total = len(symbol_ref_year)
        self.logger.debug(
            f"Starting fundamental upload for {total} symbols "
            f"from {start_date} to {end_date}"
        )
        self.logger.debug("Storage: data/raw/fundamental/{security_id}/fundamental.parquet")

        # Batch prefetch CIKs
        self.logger.debug(f"Step 1/4: Pre-fetching CIKs for {total} symbols...")
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_syms = [s for s, y in symbol_ref_year.items() if y == year]
            if year_syms:
                cik_map.update(self.cik_resolver.batch_prefetch_ciks(year_syms, year, batch_size=100))
        prefetch_time = time.time() - prefetch_start
        self.logger.debug(f"CIK pre-fetch completed in {prefetch_time:.1f}s")

        # Filter symbols with valid CIKs
        self.logger.debug("Step 2/4: Filtering symbols with valid CIKs...")
        symbols_with_cik = [s for s in symbol_ref_year if cik_map.get(s)]
        symbols_without = [s for s in symbol_ref_year if not cik_map.get(s)]

        self.logger.debug(
            f"Symbol filtering: {len(symbols_with_cik)}/{total} have CIKs, "
            f"{len(symbols_without)} non-SEC filers"
        )
        if symbols_without and len(symbols_without) <= 30:
            self.logger.debug(f"Non-SEC filers: {sorted(symbols_without)}")

        # Build security_id cache
        self.logger.debug(f"Step 3/4: Resolving security IDs for {len(symbols_with_cik)} symbols...")
        security_id_cache: Dict[str, Optional[int]] = {}
        for sym in symbols_with_cik:
            ref_year = symbol_ref_year[sym]
            try:
                security_id_cache[sym] = self.security_master.get_security_id(sym, f"{ref_year}-12-31")
            except ValueError:
                security_id_cache[sym] = None

        return symbols_with_cik, cik_map, security_id_cache, prefetch_time

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
        self.logger.debug(
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
