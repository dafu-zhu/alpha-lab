"""
Top 3000 symbols upload handler.

Handles monthly top 3000 liquid stocks list uploads.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Any

from tqdm import tqdm

from alphalab.storage.handlers.base import BaseHandler

if TYPE_CHECKING:
    from alphalab.storage.pipeline import DataPublishers, DataCollectors, Validator
    from alphalab.universe.manager import UniverseManager
    from alphalab.utils.calendar import TradingCalendar


class Top3000Handler(BaseHandler):
    """
    Handles top 3000 symbols upload.

    Storage: data/meta/universe/{YYYY}/{MM}/top3000.txt
    """

    def __init__(
        self,
        data_publishers: DataPublishers,
        data_collectors: DataCollectors,
        universe_manager: UniverseManager,
        validator: Validator,
        calendar: TradingCalendar,
        logger: logging.Logger,
        alpaca_start_year: int = 2025
    ):
        super().__init__(logger)
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.universe_manager = universe_manager
        self.validator = validator
        self.calendar = calendar
        self.alpaca_start_year = alpaca_start_year

    def upload_year(
        self,
        year: int,
        overwrite: bool = False,
        auto_resolve: bool = True
    ) -> Dict[str, Any]:
        """Upload top 3000 symbols for each month in a year."""
        start_time = time.time()
        symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')

        if not symbols:
            self.logger.warning(f"No symbols for {year}, skipping top3000")
            return self.stats

        source = 'crsp' if year < self.alpaca_start_year else 'alpaca'
        self.logger.debug(
            f"Starting {year} top3000 upload for {len(symbols)} symbols "
            f"(source={source}, overwrite={overwrite})"
        )

        self.reset_stats()
        today = dt.date.today()
        months_processed = 0

        pbar = tqdm(range(1, 13), desc="Top3000", unit="month", leave=False)
        for month in pbar:
            # Skip existing
            if not overwrite and self.validator.top_3000_exists(year, month):
                self.logger.debug(f"Skipping {year}-{month:02d}: exists")
                self.stats['skipped'] += 1
                pbar.set_postfix(ok=self.stats['success'], skip=self.stats['skipped'])
                continue

            trading_days = self.calendar.load_trading_days(year, month)
            if not trading_days:
                self.logger.debug(f"No trading days for {year}-{month:02d}")
                self.stats['skipped'] += 1
                pbar.set_postfix(ok=self.stats['success'], skip=self.stats['skipped'])
                continue

            as_of = trading_days[-1]
            as_of_date = dt.datetime.strptime(as_of, "%Y-%m-%d").date()

            # Stop at future months
            if as_of_date > today and (as_of_date.year, as_of_date.month) > (today.year, today.month):
                self.logger.debug(f"Stopping at {year}-{month:02d}: future month")
                break

            top_3000 = self.universe_manager.get_top_3000(
                as_of, symbols, source, auto_resolve=auto_resolve
            )

            result = self.publishers.publish_top_3000(
                year=year,
                month=month,
                as_of=as_of,
                symbols=top_3000,
                source=source
            )

            if result['status'] == 'success':
                self.stats['success'] += 1
                months_processed += 1
                self.logger.debug(
                    f"Uploaded top3000 for {year}-{month:02d} (as_of={as_of}, count={len(top_3000)})"
                )
            elif result['status'] == 'skipped':
                self.stats['skipped'] += 1
                self.logger.debug(f"Skipped {year}-{month:02d}: {result.get('error')}")
            else:
                self.stats['failed'] += 1
                self.logger.error(f"Failed {year}-{month:02d}: {result.get('error')}")

            pbar.set_postfix(ok=self.stats['success'], skip=self.stats['skipped'])

        pbar.close()
        elapsed = time.time() - start_time
        self.logger.info(
            f"Successfully generated top3000 in {elapsed:.1f}s "
            f"({self.stats['success']} months, {self.stats['skipped']} skipped)"
        )
        return self.stats
