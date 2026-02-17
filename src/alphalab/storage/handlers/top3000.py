"""
Top 3000 symbols upload handler.

Handles monthly top 3000 liquid stocks list uploads.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Any

from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

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
        """Upload top 3000 symbols for all months in a single year."""
        return self.upload_range(year, year, overwrite, auto_resolve)

    def upload_range(
        self,
        start_year: int,
        end_year: int,
        overwrite: bool = False,
        auto_resolve: bool = True
    ) -> Dict[str, Any]:
        """Upload top 3000 symbols for all months in year range."""
        start_time = time.time()
        self.reset_stats()
        today = dt.date.today()

        # Build list of (year, month) tuples to process
        year_months = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                year_months.append((year, month))

        # Cache symbols per year
        symbols_cache: Dict[int, List[str]] = {}

        with Progress(
            TextColumn("Downloading top3000"),
            BarColumn(bar_width=30, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            transient=True
        ) as progress:
            task = progress.add_task("", total=len(year_months))

            for year, month in year_months:
                # Skip existing
                if not overwrite and self.validator.top_3000_exists(year, month):
                    self.stats['skipped'] += 1
                    progress.advance(task)
                    continue

                # Load symbols for year (cached)
                if year not in symbols_cache:
                    symbols_cache[year] = self.universe_manager.load_symbols_for_year(
                        year, sym_type='alpaca'
                    )
                symbols = symbols_cache[year]

                if not symbols:
                    self.stats['skipped'] += 1
                    progress.advance(task)
                    continue

                trading_days = self.calendar.load_trading_days(year, month)
                if not trading_days:
                    self.stats['skipped'] += 1
                    progress.advance(task)
                    continue

                as_of = trading_days[-1]
                as_of_date = dt.datetime.strptime(as_of, "%Y-%m-%d").date()

                # Stop at future months
                if as_of_date > today:
                    self.stats['skipped'] += 1
                    progress.advance(task)
                    continue

                source = 'crsp' if year < self.alpaca_start_year else 'alpaca'
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
                elif result['status'] == 'skipped':
                    self.stats['skipped'] += 1
                else:
                    self.stats['failed'] += 1
                    self.logger.error(f"Failed {year}-{month:02d}: {result.get('error')}")

                progress.advance(task)
        elapsed = time.time() - start_time
        print(
            f"Downloading top3000... done "
            f"({self.stats['success']} ok, {self.stats['skipped']} skip, {elapsed:.1f}s)"
        )
        return self.stats
