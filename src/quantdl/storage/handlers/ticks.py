"""
Ticks upload handlers.

Handles daily ticks uploads using Alpaca as the sole data source.
SecurityMaster-driven: downloads full date range for all securities at once.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple

from tqdm import tqdm
import polars as pl

from quantdl.storage.handlers.base import BaseHandler

if TYPE_CHECKING:
    from quantdl.storage.pipeline import DataPublishers, DataCollectors, Validator
    from quantdl.master.security_master import SecurityMaster


class DailyTicksHandler(BaseHandler):
    """
    Handles daily ticks upload using Alpaca (2017+).

    Uses SecurityMaster as the definitive source of all symbols (including
    delisted), batches symbols per Alpaca API call, and downloads the entire
    date range at once.

    Storage: data/raw/ticks/daily/{security_id}/ticks.parquet
    """

    def __init__(
        self,
        data_publishers: DataPublishers,
        data_collectors: DataCollectors,
        security_master: SecurityMaster,
        validator: Validator,
        logger: logging.Logger,
    ):
        super().__init__(logger)
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.security_master = security_master
        self.validator = validator

    def upload_range(
        self,
        start_year: int,
        end_year: int,
        overwrite: bool = False,
        chunk_size: int = 50,
        sleep_time: float = 0.2,
    ) -> Dict[str, Any]:
        """Upload daily ticks for all securities active in [start_year, end_year]."""
        # 1. Get all (symbol, security_id) from SecurityMaster
        securities = self.security_master.get_securities_in_range(start_year, end_year)

        self.logger.debug(
            f"Found {len(securities)} securities for {start_year}-{end_year}"
        )
        self.reset_stats()

        total = len(securities)
        pbar = tqdm(total=total, desc=f"Uploading {start_year}-{end_year} daily ticks", unit="sym")

        # 2. Filter already-downloaded
        if not overwrite:
            securities = self._filter_existing(securities, end_year, pbar)

        # 3. Batch and download
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        for i in range(0, len(securities), chunk_size):
            batch = securities[i:i + chunk_size]
            syms = [s for s, _ in batch]

            symbol_map = self.collectors.collect_daily_ticks_range_bulk(
                syms, start_date, end_date
            )

            for sym, sid in batch:
                df = symbol_map.get(sym, pl.DataFrame())
                result = self.publishers.publish_daily_ticks(
                    sym, sid, df, start_year, end_year
                )
                self._update_stats_from_result(result)
                pbar.update(1)
                pbar.set_postfix(
                    ok=self.stats['success'], fail=self.stats['failed'],
                    skip=self.stats['skipped'], cancel=self.stats['canceled']
                )

            if sleep_time > 0 and i + chunk_size < len(securities):
                time.sleep(sleep_time)

        pbar.close()
        self.log_summary(f"{start_year}-{end_year} daily ticks (alpaca)", total, time.time())
        return self.stats

    def _filter_existing(
        self,
        securities: List[Tuple[str, int]],
        end_year: int,
        pbar,
    ) -> List[Tuple[str, int]]:
        """Filter out securities that already have end_year data."""
        to_fetch = []
        for sym, sid in securities:
            if sid is None:
                self.stats['skipped'] += 1
                pbar.update(1)
            elif self.validator.data_exists(sym, 'ticks', year=end_year, security_id=sid):
                self.stats['canceled'] += 1
                pbar.update(1)
            else:
                to_fetch.append((sym, sid))
        pbar.set_postfix(
            ok=self.stats['success'], fail=self.stats['failed'],
            skip=self.stats['skipped'], cancel=self.stats['canceled']
        )
        return to_fetch

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
