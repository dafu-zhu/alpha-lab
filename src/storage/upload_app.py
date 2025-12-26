import io
import json
import datetime as dt
from typing import List
from pathlib import Path
import logging
import requests
import queue
import threading
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from boto3.s3.transfer import TransferConfig
import polars as pl

from utils.logger import setup_logger
from utils.mapping import symbol_cik_mapping
from storage.config_loader import UploadConfig
from storage.s3_client import S3Client
from collection.fundamental import Fundamental
from collection.ticks import Ticks
from collection.models import TickField
from storage.validation import Validator


class UploadApp:
    def __init__(self, symbol_file: str="universe_top3000.txt"):
        # Load symbols
        self.symbol_path = Path(f"data/symbols/{symbol_file}")
        self.sec_symbols = self._load_symbols('sec')
        self.alpaca_symbols = self._load_symbols('alpaca')
        self.cik_map = symbol_cik_mapping()

        # Setup config and client
        self.config = UploadConfig()
        self.client = S3Client().client

        # Setup logger
        self.logger = setup_logger(
            name=f"uploadapp",
            log_dir=Path("data/logs/upload"),
            level=logging.INFO,
            console_output = True
        )

        self.validator = Validator(self.client)

        # Load trading calendar
        self.calendar_path = Path("data/calendar/master.parquet")
        self.trading_days = self._load_trading_days()

    def _load_symbols(self, sym_type: str) -> List[str]:
        """
        Load symbol as list from file
        SEC uses '-' as separator ('BRK-B'), Alpaca uses '.' ('BRK.B')

        :param sym_type: "sec" or "alpaca"
        """
        symbols = []
        with open(self.symbol_path, 'r') as file:
            if sym_type == "alpaca":
                for line in file:
                    symbol = line.strip()
                    symbols.append(symbol)
            elif sym_type == "sec":
                for line in file:
                    symbol = line.strip().replace('.', '-')
                    symbols.append(symbol)
            else:
                msg = f"Expected sym_type: 'sec' or 'alpaca', get {sym_type}"
                raise ValueError(msg)

        return symbols

    def _load_trading_days(self, year: int = 2024) -> List[str]:
        """
        Load trading days from master calendar for a specific year.

        :param year: Year to load trading days for (default: 2024)
        :return: List of trading days in 'YYYY-MM-DD' format
        """
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        df = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col('Date').is_between(start_date, end_date))
            .select('Date')
            .collect()
        )

        return [d.strftime('%Y-%m-%d') for d in df['Date'].to_list()]

    # ===========================
    # Upload ticks
    # ===========================
    def _process_symbol_daily_ticks(self, sym: str, year: int, overwrite: bool = False) -> dict:
        """
        Process daily ticks for a single symbol.
        Returns dict with status for progress tracking.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        # Convert to SEC format for S3 key (BRK.B -> BRK-B)
        sec_symbol = sym.replace('.', '-')

        if not overwrite and self.validator.data_exists(sec_symbol, 'ticks', year):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for year {year} already exists'}
        try:
            # Fetch from Alpaca API using Alpaca format (with '.')
            ticks = Ticks(sym)
            daily_df = ticks.collect_daily_ticks(year=year)

            # Check if all data columns are null (no actual data available)
            if daily_df['open'].is_null().all():
                return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

            # Setup S3 message
            buffer = io.BytesIO()
            daily_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            # Use SEC format (with '-') in S3 key
            s3_key = f"data/raw/ticks/daily/{sec_symbol}/{year}/ticks.parquet"
            s3_metadata = {
                'symbol': sec_symbol,
                'year': str(year),
                'data_type': 'ticks'
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch daily ticks for {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def daily_ticks(self, overwrite: bool = False):
        """
        Upload daily ticks for all symbols sequentially (no concurrency to avoid rate limits).

        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        year = 2024

        total = len(self.alpaca_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        self.logger.info(f"Starting daily ticks upload for {total} symbols (sequential processing, overwrite={overwrite})")

        for sym in self.alpaca_symbols:
            result = self._process_symbol_daily_ticks(sym, year, overwrite=overwrite)
            completed += 1

            if result['status'] == 'success':
                success += 1
            elif result['status'] == 'canceled':
                canceled += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

            # Progress logging every 10 symbols
            if completed % 10 == 0:
                self.logger.info(f"Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled, {skipped} skipped)")

        self.logger.info(f"Daily ticks upload completed: {success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total")

    def _process_symbol_minute_ticks(self, sym: str, trade_day: str, overwrite: bool = False) -> dict:
        """
        Process minute ticks for a single symbol and trading day.
        Returns dict with status for progress tracking.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        # Convert to SEC format for S3 key (BRK.B -> BRK-B)
        sec_symbol = sym.replace('.', '-')

        if not overwrite and self.validator.data_exists(sec_symbol, 'ticks', day=trade_day):
            return {'symbol': sym, 'day': trade_day, 'status': 'canceled', 'error': f'Symbol {sym} for day {trade_day} already exists'}
        try:
            # Fetch from Alpaca API using Alpaca format (with '.')
            ticks = Ticks(sym)
            minute_df = ticks.collect_minute_ticks(trade_day=trade_day)

            # Skip if DataFrame is empty (no data available)
            if len(minute_df) == 0:
                return {'symbol': sym, 'day': trade_day, 'status': 'skipped', 'error': 'No data available'}

            # Setup S3 message
            buffer = io.BytesIO()
            minute_df.write_parquet(buffer)
            buffer.seek(0)

            # Parse date for S3 key
            date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
            year = date_obj.strftime('%Y')
            month = date_obj.strftime('%m')
            day = date_obj.strftime('%d')

            s3_data = buffer
            # Use SEC format (with '-') in S3 key
            s3_key = f"data/raw/ticks/minute/{sec_symbol}/{year}/{month}/{day}/ticks.parquet"
            s3_metadata = {
                'symbol': sec_symbol,
                'trade_day': trade_day,
                'data_type': 'ticks'
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'day': trade_day, 'status': 'success', 'error': None}

        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch minute ticks for {sym} on {trade_day}: {e}')
            return {'symbol': sym, 'day': trade_day, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym} on {trade_day}: {e}', exc_info=True)
            return {'symbol': sym, 'day': trade_day, 'status': 'failed', 'error': str(e)}

    def _upload_minute_ticks_worker(
            self, 
            data_queue: queue.Queue, 
            stats: dict, 
            stats_lock: threading.Lock, 
            overwrite: bool
        ):
        """
        Worker thread that consumes fetched data and uploads to S3.

        :param data_queue: Queue containing (sym, trade_day, minute_df) tuples
        :param stats: Shared statistics dictionary
        :param stats_lock: Lock for updating statistics
        :param overwrite: Whether to overwrite existing data
        """
        while True:
            try:
                item = data_queue.get(timeout=1)
                if item is None:  # Poison pill to stop worker
                    break

                sym, trade_day, minute_df = item

                # Convert to SEC format for S3 key
                sec_symbol = sym.replace('.', '-')

                try:
                    # Check if already exists (unless overwriting)
                    if not overwrite and self.validator.data_exists(sec_symbol, 'ticks', day=trade_day):
                        with stats_lock:
                            stats['canceled'] += 1
                        continue

                    # Skip if DataFrame is empty
                    if len(minute_df) == 0:
                        with stats_lock:
                            stats['skipped'] += 1
                        continue

                    # Setup S3 message
                    buffer = io.BytesIO()
                    minute_df.write_parquet(buffer)
                    buffer.seek(0)

                    # Parse date for S3 key
                    date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
                    year = date_obj.strftime('%Y')
                    month = date_obj.strftime('%m')
                    day = date_obj.strftime('%d')

                    s3_key = f"data/raw/ticks/minute/{sec_symbol}/{year}/{month}/{day}/ticks.parquet"
                    s3_metadata = {
                        'symbol': sec_symbol,
                        'trade_day': trade_day,
                        'data_type': 'ticks'
                    }
                    s3_metadata_prepared = {
                        k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                        for k, v in s3_metadata.items()
                    }

                    # Upload to S3
                    self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

                    with stats_lock:
                        stats['success'] += 1

                except Exception as e:
                    self.logger.error(f'Upload error for {sym} on {trade_day}: {e}')
                    with stats_lock:
                        stats['failed'] += 1

                finally:
                    data_queue.task_done()

            except queue.Empty:
                continue

    def _fetch_single_symbol_minute(self, symbol: str, year: int = 2024) -> List[dict]:
        """
        Fetch minute data for a single symbol for entire year from Alpaca.

        :param symbol: Symbol in Alpaca format (e.g., 'AAPL')
        :param year: Year to fetch (default: 2024)
        :return: List of bars
        """
        # Get year range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)
        start_str = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

        # Prepare request
        base_url = "https://data.alpaca.markets/v2/stocks/bars"

        # Get headers from a Ticks instance
        ticks = Ticks(symbol)
        headers = ticks.headers

        bars = []

        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start_str,
            "end": end_str,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "sip",
            "sort": "asc"
        }

        try:
            # Initial request
            response = requests.get(base_url, headers=headers, params=params)
            time.sleep(0.3)  # Rate limiting

            if response.status_code != 200:
                self.logger.error(f"Single fetch error for {symbol}: {response.status_code}, {response.text}")
                return bars

            data = response.json()
            symbol_bars = data.get("bars", {}).get(symbol, [])
            bars.extend(symbol_bars)

            # Handle pagination
            page_count = 1
            while "next_page_token" in data and data["next_page_token"]:
                params["page_token"] = data["next_page_token"]
                response = requests.get(base_url, headers=headers, params=params)
                time.sleep(0.3)  # Rate limiting

                if response.status_code != 200:
                    self.logger.warning(f"Pagination error on page {page_count} for {symbol}: {response.status_code}")
                    break

                data = response.json()
                symbol_bars = data.get("bars", {}).get(symbol, [])
                bars.extend(symbol_bars)

                page_count += 1

            self.logger.info(f"Fetched {len(bars)} bars for {symbol} ({page_count} pages)")

        except Exception as e:
            self.logger.error(f"Exception during single fetch for {symbol}: {e}")

        return bars

    def _fetch_minute_bulk(self, symbols: List[str], year: int = 2024) -> dict:
        """
        Bulk fetch minute data for multiple symbols for entire year from Alpaca.
        If bulk fetch fails, retry by fetching symbols one by one.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param year: Year to fetch (default: 2024)
        :return: Dict mapping symbol -> list of bars
        """
        # Get year range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)
        start_str = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

        # Prepare request
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        symbols_str = ",".join(symbols)

        # Get headers from a Ticks instance
        ticks = Ticks(symbols[0])
        headers = ticks.headers

        all_bars = {sym: [] for sym in symbols}

        params = {
            "symbols": symbols_str,
            "timeframe": "1Min",
            "start": start_str,
            "end": end_str,
            "limit": 10000,
            "adjustment": "raw",  # Raw prices for minute data
            "feed": "sip",
            "sort": "asc"
        }

        # Retry logic for bulk fetch
        max_retries = 3
        bulk_success = False

        for retry in range(max_retries):
            try:
                # Initial request
                response = requests.get(base_url, headers=headers, params=params)
                time.sleep(0.3)  # Rate limiting

                if response.status_code == 429:
                    # Rate limit error - use exponential backoff
                    wait_time = 2 ** retry
                    self.logger.warning(f"Rate limit hit for bulk fetch (retry {retry + 1}/{max_retries}), waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code != 200:
                    self.logger.error(f"Bulk fetch error for {symbols}: {response.status_code}, {response.text}")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)
                        continue
                    break

                data = response.json()
                bars = data.get("bars", {})

                # Collect bars from initial response
                for sym in symbols:
                    if sym in bars:
                        all_bars[sym].extend(bars[sym])

                # Handle pagination
                page_count = 1
                while "next_page_token" in data and data["next_page_token"]:
                    params["page_token"] = data["next_page_token"]
                    response = requests.get(base_url, headers=headers, params=params)
                    time.sleep(0.3)  # Rate limiting

                    if response.status_code != 200:
                        self.logger.warning(f"Pagination error on page {page_count} for {symbols}: {response.status_code}")
                        break

                    data = response.json()
                    bars = data.get("bars", {})

                    for sym in symbols:
                        if sym in bars:
                            all_bars[sym].extend(bars[sym])

                    page_count += 1

                self.logger.info(f"Fetched {sum(len(v) for v in all_bars.values())} total bars for {len(symbols)} symbols ({page_count} pages)")
                bulk_success = True
                break

            except Exception as e:
                self.logger.error(f"Exception during bulk fetch for {symbols} (retry {retry + 1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)

        # If bulk fetch failed after retries, fetch symbols one by one
        if not bulk_success:
            self.logger.warning(f"Bulk fetch failed after {max_retries} retries, fetching symbols individually")
            failed_symbols = []

            for sym in symbols:
                try:
                    bars = self._fetch_single_symbol_minute(sym, year)
                    all_bars[sym] = bars
                    if not bars:
                        self.logger.warning(f"No data returned for {sym}")
                except Exception as e:
                    self.logger.error(f"Failed to fetch {sym} individually: {e}")
                    failed_symbols.append(sym)

            if failed_symbols:
                self.logger.error(f"Failed to fetch {len(failed_symbols)} symbols even after individual retry: {failed_symbols}")

        return all_bars

    def minute_ticks(self, overwrite: bool = False, num_workers: int = 40, chunk_size: int = 10):
        """
        Upload minute ticks using bulk fetch + concurrent processing.
        Fetches 10 symbols at a time for full year, then processes concurrently.

        :param overwrite: If True, overwrite existing data in S3 (default: False)
        :param num_workers: Number of concurrent processing workers (default: 40)
        :param chunk_size: Number of symbols to fetch at once (default: 10)
        """
        total_symbols = len(self.alpaca_symbols)
        total_days = len(self.trading_days)
        total_tasks = total_symbols * total_days

        # Shared statistics
        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()

        # Queue for passing data from parser to upload workers
        data_queue = queue.Queue(maxsize=200)

        self.logger.info(f"Starting minute ticks upload for {total_symbols} symbols × {total_days} days = {total_tasks} tasks")
        self.logger.info(f"Bulk fetching {chunk_size} symbols at a time | {num_workers} concurrent processors")

        # Start consumer threads
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=self._upload_minute_ticks_worker,
                args=(data_queue, stats, stats_lock, overwrite),
                daemon=True
            )
            worker.start()
            workers.append(worker)

        # Producer: Bulk fetch and parse
        try:
            for i in range(0, total_symbols, chunk_size):
                chunk = self.alpaca_symbols[i:i + chunk_size]
                chunk_num = i // chunk_size + 1

                self.logger.info(f"Fetching chunk {chunk_num}/{(total_symbols + chunk_size - 1) // chunk_size}: {len(chunk)} symbols")

                # Bulk fetch for entire year
                symbol_bars = self._fetch_minute_bulk(chunk, year=2024)

                # Parse and organize by (symbol, day)
                for sym in chunk:
                    bars = symbol_bars.get(sym, [])

                    if not bars:
                        # No data for this symbol
                        for day in self.trading_days:
                            with stats_lock:
                                stats['skipped'] += 1
                                stats['completed'] += 1
                        continue

                    # Group bars by day
                    bars_by_day = {}
                    for bar in bars:
                        # Parse timestamp to get date
                        bar_dt = dt.datetime.fromisoformat(bar[TickField.TIMESTAMP.value].replace('Z', '+00:00'))
                        trade_date = bar_dt.date().strftime('%Y-%m-%d')

                        if trade_date not in bars_by_day:
                            bars_by_day[trade_date] = []
                        bars_by_day[trade_date].append(bar)

                    # Process each day
                    for day in self.trading_days:
                        day_bars = bars_by_day.get(day, [])

                        if day_bars:
                            # Parse ticks and create DataFrame
                            ticks_obj = Ticks(sym)
                            # Time zone switched to 'ET' in parse_ticks method
                            parsed_ticks = ticks_obj.parse_ticks(day_bars)

                            ticks_data = [asdict(dp) for dp in parsed_ticks]

                            minute_df = pl.DataFrame(ticks_data).with_columns([
                                pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S'),
                                pl.col('open').cast(pl.Float64),
                                pl.col('high').cast(pl.Float64),
                                pl.col('low').cast(pl.Float64),
                                pl.col('close').cast(pl.Float64),
                                pl.col('volume').cast(pl.Int64),
                                pl.col('num_trades').cast(pl.Int64),
                                pl.col('vwap').cast(pl.Float64)
                            ])
                        else:
                            # Empty DataFrame
                            minute_df = pl.DataFrame()

                        # Put in queue for upload
                        data_queue.put((sym, day, minute_df))

                        with stats_lock:
                            stats['completed'] += 1
                            completed = stats['completed']

                        # Progress logging
                        if completed % 100 == 0:
                            with stats_lock:
                                self.logger.info(
                                    f"Progress: {completed}/{total_tasks} "
                                    f"({stats['success']} success, {stats['failed']} failed, "
                                    f"{stats['canceled']} canceled, {stats['skipped']} skipped)"
                                )

        except Exception as e:
            self.logger.error(f"Error in bulk fetch/parse: {e}", exc_info=True)

        finally:
            # Signal workers to stop
            for _ in range(num_workers):
                data_queue.put(None)

            # Wait for all workers to finish
            for worker in workers:
                worker.join()

        self.logger.info(
            f"Minute ticks upload completed: {stats['success']} success, {stats['failed']} failed, "
            f"{stats['canceled']} canceled, {stats['skipped']} skipped out of {total_tasks} total"
        )

    # ===========================
    # Upload fundamental
    # ===========================
    def _process_symbol_fundamental(self, sym: str, year: int, dei_fields: List[str], gaap_fields: List[str], overwrite: bool = False) -> dict:
        """
        Process fundamental data for a single symbol.
        Returns dict with status for progress tracking.

        :param overwrite: If True, skip existence check and overwrite existing data
        """
        if not overwrite and self.validator.data_exists(sym, 'fundamental', year):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for year {year} already exists'}
        try:
            cik = self.cik_map[sym]

            # Fetch from SEC EDGAR API
            fnd = Fundamental(cik, sym)

            # Load data on RAM
            dei_df = fnd.collect_fields(year=year, fields=dei_fields, location='dei')
            gaap_df = fnd.collect_fields(year=year, fields=gaap_fields, location='us-gaap')

            # Merge on Date column
            combined_df = dei_df.join(
                gaap_df,
                on='Date',
                how='inner'
            )

            # Setup S3 message
            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            s3_key = f"data/raw/fundamental/{sym}/{year}/fundamental.parquet"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'fundamental'
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch data for {sym} (CIK {cik}): {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except KeyError as e:
            self.logger.warning(f'Invalid symbol ({sym}) for CIK mapping: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except ValueError as e:
            self.logger.error(f'Invalid data for {sym} (CIK {cik}): {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym} (CIK {cik}): {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def fundamental(self, max_workers: int = 20, overwrite: bool = False):
        """
        Upload fundamental data for all symbols using threading.

        :param max_workers: Number of concurrent threads (default: 20)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        # Fields
        dei_fields = self.config.dei_fields
        gaap_fields = self.config.us_gaap_fields
        year = 2024

        total = len(self.sec_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0

        self.logger.info(f"Starting fundamental upload for {total} symbols with {max_workers} workers (overwrite={overwrite})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_symbol_fundamental, sym, year, dei_fields, gaap_fields, overwrite): sym
                for sym in self.sec_symbols
            }

            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                completed += 1

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                else:
                    failed += 1

                # Progress logging every 10 symbols
                if completed % 10 == 0:
                    self.logger.info(f"Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled)")

        self.logger.info(f"Fundamental upload completed: {success} success, {failed} failed, {canceled} canceled out of {total} total")

    def upload_fileobj(
            self, 
            data: io.BytesIO, 
            key: str, 
            metadata: dict=None
        ) -> None:
        """Upload file object to S3 with proper configuration"""

        # Define transfer config
        cfg = self.config.transfer
        transfer_config = TransferConfig(
            multipart_threshold=int(eval(cfg.get('multipart_threshold', 10*1024*1024))),
            max_concurrency=int(cfg.get('max_concurrency', 5)),
            multipart_chunksize=int(eval(cfg.get('multipart_chunksize', 10*1024*1024))),
            num_download_attempts=int(cfg.get('num_download_attempts', 5)),
            max_io_queue=int(cfg.get('max_io_queue', 100)),
            io_chunksize=int(cfg.get('io_chunksize', 262144)),
            use_threads=(str(cfg.get('use_threads', True)).lower() == "true")
        )

        # Determine content type
        content_type_map = {
            '.parquet': 'application/x-parquet',
            '.json': 'application/json',
            '.csv': 'text/csv'
        }
        file_ext = Path(key).suffix
        content_type = content_type_map.get(file_ext, 'application/octet-stream')

        # Build ExtraArgs
        extra_args = {
            'ContentType': content_type,
            'ServerSideEncryption': 'AES256',
            'StorageClass': 'INTELLIGENT_TIERING',
            'Metadata': metadata or {}
        }

        # Upload
        self.client.upload_fileobj(
            Fileobj=data,
            Bucket='us-equity-datalake',
            Key=key,
            Config=transfer_config,
            ExtraArgs=extra_args
        )

if __name__ == "__main__":
    import time
    app = UploadApp()
    start = time.time()
    app.minute_ticks()
    print(f"Execution time: {time.time() - start:.2f} seconds")