"""
Ticks upload handlers.

Handles daily and minute ticks uploads with support for CRSP/Alpaca sources.
"""

from __future__ import annotations

import datetime as dt
import queue
import threading
import time
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from tqdm import tqdm
import polars as pl

from quantdl.storage.handlers.base import BaseHandler
from quantdl.storage.utils import UploadProgressTracker

if TYPE_CHECKING:
    from quantdl.storage.pipeline import DataPublishers, DataCollectors, Validator
    from quantdl.universe.manager import UniverseManager
    from quantdl.master.security_master import SecurityMaster
    from quantdl.collection.crsp_ticks import CRSPDailyTicks
    from quantdl.utils.calendar import TradingCalendar


class DailyTicksHandler(BaseHandler):
    """
    Handles daily ticks upload with support for:
    - CRSP bulk history (permno-centric)
    - Alpaca year-by-year
    - Monthly partitions for current year

    Storage:
    - History: data/raw/ticks/daily/{security_id}/history.parquet
    - Monthly: data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet
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
        crsp_ticks: CRSPDailyTicks,
        logger: logging.Logger,
        wrds_available: bool = True,
        alpaca_start_year: int = 2025
    ):
        super().__init__(logger)
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.security_master = security_master
        self.universe_manager = universe_manager
        self.validator = validator
        self.crsp_ticks = crsp_ticks
        self.wrds_available = wrds_available
        self.alpaca_start_year = alpaca_start_year

    def upload_year(
        self,
        year: int,
        overwrite: bool = False,
        chunk_size: int = 200,
        sleep_time: float = 0.2,
        current_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """Upload daily ticks for a single year."""
        if current_year is None:
            current_year = dt.datetime.now().year

        if year < self.alpaca_start_year and not self.wrds_available:
            raise RuntimeError(
                f"Cannot upload year {year}: WRDS required for CRSP data (years < {self.alpaca_start_year})"
            )

        # Load symbols and resolve security IDs
        symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
        security_id_cache = self._build_security_id_cache(symbols, year)

        is_completed = year < current_year
        source = 'crsp' if year < self.alpaca_start_year else 'alpaca'

        self.logger.debug(
            f"Starting {year} daily ticks upload for {len(symbols)} symbols "
            f"(source={source}, completed={is_completed})"
        )

        if is_completed:
            # History mode for completed years
            return self._upload_history_mode(
                symbols, year, security_id_cache, overwrite, chunk_size, sleep_time, source
            )
        else:
            # Monthly partitions for current year
            return self._upload_monthly_mode(
                symbols, year, security_id_cache, overwrite, chunk_size, sleep_time, source
            )

    def upload_crsp_bulk_history(
        self,
        start_year: int = 2009,
        end_year: int = 2024,
        overwrite: bool = False,
        chunk_size: int = 500,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Upload CRSP daily ticks using permno-centric bulk fetch.
        Much more efficient than year-by-year symbol fetches.
        """
        if not self.wrds_available:
            raise RuntimeError("CRSP bulk history requires WRDS connection")

        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        self.logger.debug(f"Starting CRSP bulk history: {start_year}-{end_year}")

        # Build permno -> security_id mapping
        permno_sid_map, all_sids = self._build_permno_mapping()

        # Setup progress tracker
        tracker = UploadProgressTracker(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            task_name='daily_ticks_crsp_bulk'
        )

        # Filter for resume
        if resume and not overwrite:
            completed = tracker.load()
            pending_sids = all_sids - completed
            permnos_to_fetch = [
                p for p, sids in permno_sid_map.items()
                if any(sid in pending_sids for sid, _, _ in sids)
            ]
            self.logger.debug(f"Resuming: {len(completed)} done, {len(pending_sids)} pending")
        else:
            permnos_to_fetch = list(permno_sid_map.keys())
            pending_sids = all_sids
            if overwrite:
                tracker.reset()

        tracker.set_total(len(all_sids))

        if not permnos_to_fetch:
            self.logger.debug("All security_ids completed")
            return self.stats

        # Fetch and upload in batches
        self.reset_stats()
        master_tb = self.security_master.master_tb

        with tracker:
            pbar = tqdm(total=len(pending_sids), desc="CRSP history", unit="sid")

            for i in range(0, len(permnos_to_fetch), chunk_size):
                chunk = permnos_to_fetch[i:i + chunk_size]

                # Bulk fetch
                permno_data = self.crsp_ticks.collect_daily_ticks_full_history_bulk(
                    permnos=chunk,
                    start_date=start_date,
                    end_date=end_date,
                    chunk_size=chunk_size
                )

                # Split by security_id and publish
                for permno in chunk:
                    self._process_permno_data(
                        permno, permno_data, permno_sid_map, pending_sids,
                        master_tb, tracker, pbar
                    )

            pbar.close()

        self.log_summary("CRSP bulk history", len(pending_sids), time.time())
        return self.stats

    def _upload_history_mode(
        self,
        symbols: List[str],
        year: int,
        security_id_cache: Dict[str, Optional[int]],
        overwrite: bool,
        chunk_size: int,
        sleep_time: float,
        source: str
    ) -> Dict[str, Any]:
        """Upload completed year to history.parquet."""
        self.logger.debug(f"Storage: data/raw/ticks/daily/{{security_id}}/history.parquet")
        self.reset_stats()

        total = len(symbols)
        pbar = tqdm(total=total, desc=f"Uploading {year} history", unit="sym")

        for i in range(0, total, chunk_size):
            chunk = symbols[i:i + chunk_size]

            # Filter existing
            if not overwrite:
                chunk = self._filter_existing_symbols(
                    chunk, year, security_id_cache, pbar, month=None
                )
                if not chunk:
                    continue

            # Bulk fetch year data
            if year >= self.alpaca_start_year:
                symbol_map = self.collectors.collect_daily_ticks_year_bulk(chunk, year)
            else:
                symbol_map = self.collectors.collect_daily_ticks_year_bulk(chunk, year)

            # Publish each symbol
            for sym in chunk:
                security_id = security_id_cache.get(sym)
                df = symbol_map.get(sym, pl.DataFrame())

                result = self.publishers.publish_daily_ticks(
                    sym, year, security_id, df, month=None, by_year=False
                )
                self._update_stats_from_result(result)
                pbar.update(1)
                pbar.set_postfix(
                    ok=self.stats['success'], fail=self.stats['failed'],
                    skip=self.stats['skipped'], cancel=self.stats['canceled']
                )

        pbar.close()
        self.log_summary(f"{year} daily ticks ({source})", total, time.time())
        return self.stats

    def _upload_monthly_mode(
        self,
        symbols: List[str],
        year: int,
        security_id_cache: Dict[str, Optional[int]],
        overwrite: bool,
        chunk_size: int,
        sleep_time: float,
        source: str
    ) -> Dict[str, Any]:
        """Upload current year with monthly partitions."""
        self.logger.debug(f"Storage: data/raw/ticks/daily/{{security_id}}/{year}/{{MM}}/ticks.parquet")

        total = len(symbols)
        today = dt.date.today()

        for month in range(1, 13):
            # Skip future months
            if year == today.year and month > today.month:
                self.logger.debug(f"Skipping {year}-{month:02d}: future month")
                continue

            self.reset_stats()
            pbar = tqdm(total=total, desc=f"Uploading {year}-{month:02d}", unit="sym")

            for i in range(0, total, chunk_size):
                chunk = symbols[i:i + chunk_size]

                if not overwrite:
                    chunk = self._filter_existing_symbols(
                        chunk, year, security_id_cache, pbar, month=month
                    )
                    if not chunk:
                        continue

                # Bulk fetch month data
                symbol_map = self.collectors.collect_daily_ticks_month_bulk(
                    chunk, year, month, sleep_time=sleep_time
                )

                for sym in chunk:
                    security_id = security_id_cache.get(sym)
                    df = symbol_map.get(sym, pl.DataFrame())

                    result = self.publishers.publish_daily_ticks(
                        sym, year, security_id, df, month=month, by_year=False
                    )
                    self._update_stats_from_result(result)
                    pbar.update(1)
                    pbar.set_postfix(
                        ok=self.stats['success'], fail=self.stats['failed'],
                        skip=self.stats['skipped'], cancel=self.stats['canceled']
                    )

            pbar.close()
            self.logger.info(
                f"{year}-{month:02d}: {self.stats['success']} success, "
                f"{self.stats['failed']} failed, {self.stats['skipped']} skipped"
            )

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

    def _build_permno_mapping(self):
        """Build permno -> [(security_id, start, end), ...] mapping."""
        master_tb = self.security_master.master_tb
        permno_sid_map = {}
        all_sids = set()

        for row in master_tb.iter_rows(named=True):
            permno, sid = row['permno'], row['security_id']
            start, end = row['start_date'], row['end_date']

            if sid is None or permno is None:
                continue

            all_sids.add(sid)
            if permno not in permno_sid_map:
                permno_sid_map[permno] = []
            permno_sid_map[permno].append((sid, start, end))

        self.logger.debug(f"Found {len(permno_sid_map)} permnos, {len(all_sids)} security_ids")
        return permno_sid_map, all_sids

    def _filter_existing_symbols(
        self,
        chunk: List[str],
        year: int,
        security_id_cache: Dict[str, Optional[int]],
        pbar,
        month: Optional[int]
    ) -> List[str]:
        """Filter out already-existing symbols."""
        symbols_to_fetch = []
        for sym in chunk:
            sec_id = security_id_cache.get(sym)
            if sec_id is None:
                self.stats['skipped'] += 1
                pbar.update(1)
            elif self.validator.data_exists(sym, 'ticks', year, month, security_id=sec_id):
                self.stats['canceled'] += 1
                pbar.update(1)
            else:
                symbols_to_fetch.append(sym)
        pbar.set_postfix(
            ok=self.stats['success'], fail=self.stats['failed'],
            skip=self.stats['skipped'], cancel=self.stats['canceled']
        )
        return symbols_to_fetch

    def _process_permno_data(
        self,
        permno: int,
        permno_data: Dict[int, pl.DataFrame],
        permno_sid_map: Dict[int, List],
        pending_sids: set,
        master_tb: pl.DataFrame,
        tracker,
        pbar
    ):
        """Process data for a single permno."""
        if permno not in permno_data:
            for sid, _, _ in permno_sid_map.get(permno, []):
                if sid in pending_sids:
                    self.stats['skipped'] += 1
                    tracker.mark_skipped(sid)
                    pbar.update(1)
            return

        df = permno_data[permno]

        for sid, sid_start, sid_end in permno_sid_map.get(permno, []):
            if sid not in pending_sids:
                continue

            # Filter for this security_id's date range
            sid_df = df.filter(
                (pl.col('timestamp') >= str(sid_start)) &
                (pl.col('timestamp') <= str(sid_end))
            )

            if len(sid_df) == 0:
                self.stats['skipped'] += 1
                tracker.mark_skipped(sid)
                pbar.update(1)
                pbar.set_postfix(ok=self.stats['success'], fail=self.stats['failed'], skip=self.stats['skipped'])
                continue

            # Get symbol for logging
            sym_info = master_tb.filter(pl.col('security_id') == sid).select('symbol').head(1)
            symbol = sym_info['symbol'][0] if len(sym_info) > 0 else f"sid_{sid}"

            try:
                result = self.publishers.publish_daily_ticks_to_history(
                    security_id=sid, df=sid_df, symbol=symbol
                )
                if result.get('status') == 'success':
                    self.stats['success'] += 1
                    tracker.mark_completed(sid)
                else:
                    self.stats['failed'] += 1
                    tracker.mark_failed(sid)
            except Exception as e:
                self.logger.error(f"Failed sid={sid}: {e}")
                self.stats['failed'] += 1
                tracker.mark_failed(sid)

            pbar.update(1)
            pbar.set_postfix(ok=self.stats['success'], fail=self.stats['failed'], skip=self.stats['skipped'])

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


class MinuteTicksHandler(BaseHandler):
    """
    Handles minute ticks upload with parallel processing.

    Storage: data/raw/ticks/minute/{security_id}/{YYYY}/{MM}/{DD}/ticks.parquet
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
        calendar: TradingCalendar,
        logger: logging.Logger
    ):
        super().__init__(logger)
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.publishers = data_publishers
        self.collectors = data_collectors
        self.security_master = security_master
        self.universe_manager = universe_manager
        self.validator = validator
        self.calendar = calendar

    def upload_year(
        self,
        year: int,
        months: Optional[List[int]] = None,
        overwrite: bool = False,
        resume: bool = False,
        num_workers: int = 50,
        chunk_size: int = 500,
        sleep_time: float = 0.0
    ) -> Dict[str, Any]:
        """Upload minute ticks for a year with shared worker pool."""
        if months is None:
            months = list(range(1, 13))

        symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')

        # Shared stats and queue
        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        data_queue = queue.Queue(maxsize=200)

        self.logger.debug(f"Starting {year} minute ticks ({len(months)} months) | {num_workers} workers")

        # Start workers
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=self.publishers.minute_ticks_worker,
                args=(data_queue, stats, stats_lock),
                daemon=True
            )
            worker.start()
            workers.append(worker)

        try:
            for month in months:
                self._upload_month(
                    year, month, symbols, overwrite, resume, chunk_size, sleep_time,
                    data_queue, stats, stats_lock
                )
        finally:
            # Stop workers
            for _ in range(num_workers):
                data_queue.put(None)
            for worker in workers:
                worker.join()

        self.logger.info(
            f"Year {year} minute ticks: {stats['success']} success, "
            f"{stats['failed']} failed, {stats['canceled']} canceled"
        )
        return stats

    def _upload_month(
        self,
        year: int,
        month: int,
        symbols: List[str],
        overwrite: bool,
        resume: bool,
        chunk_size: int,
        sleep_time: float,
        data_queue: queue.Queue,
        stats: Dict[str, int],
        stats_lock: threading.Lock
    ):
        """Process a single month."""
        tracker = UploadProgressTracker(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            task_name=f'minute_ticks_{year}_{month:02d}',
            key_type='str'
        )

        # Filter completed symbols
        if resume and not overwrite:
            completed = tracker.load()
            symbols = [s for s in symbols if s not in completed]
            self.logger.info(f"{year}-{month:02d}: {len(completed)} done, {len(symbols)} pending")
        elif overwrite:
            tracker.reset()

        trading_days = self.calendar.load_trading_days(year, month)

        # Filter future dates
        today = dt.date.today()
        trading_days = [d for d in trading_days if d <= today.strftime("%Y-%m-%d")]

        if not trading_days:
            self.logger.info(f"Skipping {year}-{month:02d}: no trading days")
            return

        total_tasks = len(symbols) * len(trading_days)
        if total_tasks == 0:
            return

        pbar = tqdm(total=total_tasks, desc=f"Uploading {year}-{month:02d} minute", unit="task")

        try:
            with tracker:
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i:i + chunk_size]

                    # Pre-filter
                    if not overwrite:
                        chunk = self._filter_complete_symbols(
                            chunk, year, month, trading_days, tracker, stats, stats_lock, pbar
                        )
                        if not chunk:
                            continue

                    # Bulk fetch
                    symbol_bars = self.collectors.fetch_minute_month(chunk, year, month, sleep_time=sleep_time)
                    parsed = self.collectors.parse_minute_bars_to_daily(symbol_bars, trading_days)

                    # Queue for upload
                    symbols_with_data = set()
                    for (sym, day), df in parsed.items():
                        if len(df) == 0:
                            with stats_lock:
                                stats['skipped'] += 1
                                stats['completed'] += 1
                        else:
                            symbols_with_data.add(sym)
                            data_queue.put((sym, day, df))
                            with stats_lock:
                                stats['completed'] += 1

                        pbar.update(1)
                        pbar.set_postfix(
                            ok=stats['success'], fail=stats['failed'],
                            skip=stats['skipped'], cancel=stats['canceled']
                        )

                    for sym in symbols_with_data:
                        tracker.mark_completed(sym)

        except Exception as e:
            self.logger.error(f"Error in {year}-{month:02d}: {e}", exc_info=True)
        finally:
            pbar.close()

    def _filter_complete_symbols(
        self,
        chunk: List[str],
        year: int,
        month: int,
        trading_days: List[str],
        tracker,
        stats: Dict[str, int],
        stats_lock: threading.Lock,
        pbar
    ) -> List[str]:
        """Filter symbols that are already complete."""
        trading_day_set = {d.split('-')[2] for d in trading_days}
        symbols_to_fetch = []

        for sym in chunk:
            try:
                security_id = self.security_master.get_security_id(sym, trading_days[0])
            except ValueError:
                symbols_to_fetch.append(sym)
                continue

            if security_id is None:
                symbols_to_fetch.append(sym)
                continue

            existing = self.validator.get_existing_minute_days(security_id, year, month)
            if not trading_day_set.issubset(existing):
                symbols_to_fetch.append(sym)
            else:
                tracker.mark_completed(sym)
                count = len(trading_days)
                with stats_lock:
                    stats['canceled'] += count
                    stats['completed'] += count
                pbar.update(count)

        pbar.set_postfix(
            ok=stats['success'], fail=stats['failed'],
            skip=stats['skipped'], cancel=stats['canceled']
        )
        return symbols_to_fetch
