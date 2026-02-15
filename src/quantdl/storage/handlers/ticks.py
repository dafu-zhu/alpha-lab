"""
Ticks upload handlers.

Handles daily ticks uploads using Alpaca as the sole data source.
"""

from __future__ import annotations

import datetime as dt
import time
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from tqdm import tqdm
import polars as pl

from quantdl.storage.handlers.base import BaseHandler

if TYPE_CHECKING:
    from quantdl.storage.pipeline import DataPublishers, DataCollectors, Validator
    from quantdl.universe.manager import UniverseManager
    from quantdl.master.security_master import SecurityMaster


class DailyTicksHandler(BaseHandler):
    """
    Handles daily ticks upload using Alpaca (2017+).

    Storage: data/raw/ticks/daily/{security_id}/ticks.parquet
    """

    def __init__(
        self,
        s3_client,
        bucket_name: str,
        data_publishers: DataPublishers,
        data_collectors: DataCollectors,
        security_master: SecurityMaster,
        universe_manager: UniverseManager,
        validator: Validator,
        logger: logging.Logger,
    ):
        super().__init__(logger)
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.security_master = security_master
        self.universe_manager = universe_manager
        self.validator = validator

    def upload_year(
        self,
        year: int,
        overwrite: bool = False,
        chunk_size: int = 200,
        sleep_time: float = 0.2,
    ) -> Dict[str, Any]:
        """Upload daily ticks for a single year (Alpaca source)."""
        # Load symbols and resolve security IDs
        symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
        security_id_cache = self._build_security_id_cache(symbols, year)

        self.logger.debug(
            f"Starting {year} daily ticks upload for {len(symbols)} symbols "
            f"(source=alpaca)"
        )
        self.logger.debug("Storage: data/raw/ticks/daily/{security_id}/ticks.parquet")
        self.reset_stats()

        total = len(symbols)
        pbar = tqdm(total=total, desc=f"Uploading {year} daily ticks", unit="sym")

        for i in range(0, total, chunk_size):
            chunk = symbols[i:i + chunk_size]

            # Filter existing
            if not overwrite:
                chunk = self._filter_existing_symbols(
                    chunk, year, security_id_cache, pbar
                )
                if not chunk:
                    continue

            # Bulk fetch year data
            symbol_map = self.collectors.collect_daily_ticks_year_bulk(chunk, year)

            # Publish each symbol
            for sym in chunk:
                security_id = security_id_cache.get(sym)
                df = symbol_map.get(sym, pl.DataFrame())

                result = self.publishers.publish_daily_ticks(
                    sym, year, security_id, df
                )
                self._update_stats_from_result(result)
                pbar.update(1)
                pbar.set_postfix(
                    ok=self.stats['success'], fail=self.stats['failed'],
                    skip=self.stats['skipped'], cancel=self.stats['canceled']
                )

        pbar.close()
        self.log_summary(f"{year} daily ticks (alpaca)", total, time.time())
        return self.stats

    def _build_security_id_cache(
        self,
        symbols: List[str],
        year: int
    ) -> Dict[str, Optional[int]]:
        """Pre-resolve security IDs for all symbols."""
        cache = {}
        for sym in symbols:
            try:
                cache[sym] = self.security_master.get_security_id(sym, f"{year}-12-31")
            except ValueError:
                cache[sym] = None
        return cache

    def _filter_existing_symbols(
        self,
        chunk: List[str],
        year: int,
        security_id_cache: Dict[str, Optional[int]],
        pbar,
    ) -> List[str]:
        """Filter out already-existing symbols."""
        symbols_to_fetch = []
        for sym in chunk:
            sec_id = security_id_cache.get(sym)
            if sec_id is None:
                self.stats['skipped'] += 1
                pbar.update(1)
            elif self.validator.data_exists(sym, 'ticks', year, security_id=sec_id):
                self.stats['canceled'] += 1
                pbar.update(1)
            else:
                symbols_to_fetch.append(sym)
        pbar.set_postfix(
            ok=self.stats['success'], fail=self.stats['failed'],
            skip=self.stats['skipped'], cancel=self.stats['canceled']
        )
        return symbols_to_fetch

    def _update_stats_from_result(self, result: Dict[str, Any]):
        status = result.get('status', 'failed')
        if status == 'success':
            self.stats['success'] += 1
        elif status == 'canceled':
            self.stats['canceled'] += 1
        elif status == 'skipped':
            self.stats['skipped'] += 1
        else:
            self.stats['failed'] += 1
