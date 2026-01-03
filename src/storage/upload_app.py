import io
import os
import json
import datetime as dt
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import requests
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from boto3.s3.transfer import TransferConfig
import polars as pl
from dotenv import load_dotenv

from utils.logger import setup_logger
from storage.config_loader import UploadConfig
from storage.s3_client import S3Client
from collection.fundamental import Fundamental
from collection.alpaca_ticks import Ticks
from collection.crsp_ticks import CRSPDailyTicks
from collection.models import TickField
from storage.validation import Validator

load_dotenv()


class UploadApp:
    def __init__(self):
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

        # Initialize data fetchers once to reuse connections
        self.alpaca_ticks = Ticks()
        self.crsp_ticks = CRSPDailyTicks()

        # Load trading calendar
        self.calendar_path = Path("data/calendar/master.parquet")

        # Load Alpaca Key
        ALPACA_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

    def _load_symbols(self, year: int, month: int, sym_type: str) -> List[str]:
        """
        Load symbol list from monthly file
        SEC uses '-' as separator ('BRK-B'), Alpaca uses '.' ('BRK.B')

        :param year: Year (e.g., 2024)
        :param month: Month (1-12)
        :param sym_type: "sec" or "alpaca"
        :return: List of symbols in the specified format
        """
        # Construct path: data/symbols/YYYY/MM/universe_top3000.txt
        symbol_path = Path(f"data/symbols/{year}/{month:02d}/universe_top3000.txt")

        if not symbol_path.exists():
            self.logger.warning(f"Symbol file not found: {symbol_path}")
            return []

        symbols = []
        with open(symbol_path, 'r') as file:
            if sym_type == "alpaca":
                for line in file:
                    symbol = line.strip()
                    if symbol:  # Skip empty lines
                        symbols.append(symbol)
            elif sym_type == "sec":
                for line in file:
                    symbol = line.strip()
                    if symbol:  # Skip empty lines
                        symbols.append(symbol.replace('.', '-'))
            else:
                msg = f"Expected sym_type: 'sec' or 'alpaca', get {sym_type}"
                raise ValueError(msg)

        return symbols

    def _load_trading_days(self, year: int, month: int) -> List[str]:
        """
        Load trading days from master calendar for a specific month.

        :param year: Year to load trading days for
        :param month: Month to load trading days for (1-12)
        :return: List of trading days in 'YYYY-MM-DD' format
        """
        start_date = dt.date(year, month, 1)

        # Get last day of month
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        df = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col('timestamp').is_between(start_date, end_date))
            .select('timestamp')
            .collect()
        )

        return [d.strftime('%Y-%m-%d') for d in df['timestamp'].to_list()]

    def _get_cik(self, symbol: str, date: str) -> Optional[str]:
        """
        Get CIK for a symbol at a specific date using SecurityMaster.
        Handles both current and historical symbols through CRSP data.

        :param symbol: Symbol in SEC format (e.g., 'BRK-B', 'AAPL')
        :param date: Date in 'YYYY-MM-DD' format (e.g., '2010-01-15')
        :return: CIK string (zero-padded to 10 digits) or None if not found
        """
        # Convert SEC format to CRSP format (BRK-B -> BRKB)
        crsp_symbol = symbol.replace('.', '').replace('-', '')

        try:
            # Use SecurityMaster to get security_id at the given date
            security_id = self.crsp_ticks.security_master.get_security_id(
                symbol=crsp_symbol,
                day=date,
                auto_resolve=True  # Handle symbol changes automatically
            )

            # Query master table for CIK at this date
            master_tb = self.crsp_ticks.security_master.master_tb
            cik_record = master_tb.filter(
                pl.col('security_id') == security_id,
                pl.col('start_date') <= dt.datetime.strptime(date, '%Y-%m-%d').date(),
                pl.col('end_date') >= dt.datetime.strptime(date, '%Y-%m-%d').date()
            ).select('cik').head(1)

            if cik_record.is_empty():
                self.logger.warning(f"No CIK found for {symbol} (security_id={security_id}) at {date}")
                return None

            # Get CIK and ensure it's a string (may be int or None)
            cik_value = cik_record.item()
            if cik_value is None:
                self.logger.warning(f"CIK is NULL for {symbol} (security_id={security_id}) at {date}")
                return None

            # Convert to zero-padded string
            cik_str = str(int(cik_value))
            return cik_str

        except ValueError as e:
            # SecurityMaster couldn't resolve the symbol
            self.logger.warning(f"SecurityMaster resolution failed for {symbol} at {date}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting CIK for {symbol} at {date}: {e}", exc_info=True)
            return None

    # ===========================
    # Upload ticks
    # ===========================
    def _collect_daily_ticks(self, sym: str, year: int, month: int) -> List[Dict[str, Any]]:
        """
        Fetch daily ticks from appropriate source and return as JSON format.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :return: A list of dictionary with keys: timestamp, open, high, low, close, volume
        """
        if year < 2025:
            # Use CRSP for years < 2025 (avoids survivorship bias)
            crsp_symbol = sym.replace('.', '').replace('-', '')
            json_list = self.crsp_ticks.collect_daily_ticks(
                symbol=crsp_symbol,
                year=year,
                month=month,
                adjusted=True,
                auto_resolve=True
            )
        else:
            # Use Alpaca for years >= 2025
            json_list = self.alpaca_ticks.collect_daily_ticks(
                symbol=sym,
                year=year,
                month=month,
                adjusted=True
            )

        return json_list

    def _publish_single_daily_ticks(
            self, 
            sym: str, 
            year: int, 
            month: int, 
            overwrite: bool = False
        ) -> Dict[str, Optional[str]]:
        """
        Process daily ticks for a single symbol.
        Returns dict with status for progress tracking.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        if not overwrite and self.validator.data_exists(sym, 'ticks', year, month):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for {year}-{month:02d} already exists'}
        try:
            # Fetch data from appropriate source
            json_list = self._collect_daily_ticks(sym, year, month)

            # Check if DataFrame is empty (no rows fetched)
            if len(json_list) == 0:
                return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

            # Round numeric fields to 6 decimal places
            for record in json_list:
                for field in ['open', 'high', 'low', 'close']:
                    if field in record and record[field] is not None:
                        record[field] = round(record[field], 4)

            # Setup S3 message
            bio = io.BytesIO(json.dumps(json_list, ensure_ascii=False).encode('utf-8'))

            s3_data = bio
            s3_key = f"data/raw/ticks/daily/{sym}/{year}/{month:02d}/ticks.json"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'ticks',
                'source': 'crsp' if year < 2025 else 'alpaca'
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except ValueError as e:
            # Handle expected conditions like security not active on date
            if "not active on" in str(e):
                self.logger.info(f'Skipping {sym}: {e}')
                return {'symbol': sym, 'status': 'skipped', 'error': str(e)}
            else:
                self.logger.error(f'ValueError for {sym}: {e}')
                return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch daily ticks for {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def upload_daily_ticks(self, year: int, month: int, overwrite: bool = False):
        """
        Upload daily ticks for all symbols sequentially (no concurrency to avoid rate limits).
        Uses crsp for years < 2025, Alpaca for years >= 2025.

        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        # Load symbols for this month
        alpaca_symbols = self._load_symbols(year, month, sym_type='alpaca')

        total = len(alpaca_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        data_source = 'crsp' if year < 2025 else 'alpaca'
        self.logger.info(f"Starting {year}-{month:02d} daily ticks upload for {total} symbols (source={data_source}, sequential processing, overwrite={overwrite})")

        for sym in alpaca_symbols:
            result = self._publish_single_daily_ticks(sym, year, month, overwrite=overwrite)
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
                self.logger.info(f"{year}-{month} Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled, {skipped} skipped)")

        self.logger.info(f"{year}-{month} Daily ticks upload completed ({data_source}): {success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total")

    def _upload_minute_ticks_worker(
            self,
            data_queue: queue.Queue,
            stats: dict,
            stats_lock: threading.Lock
        ):
        """
        Worker thread that consumes fetched data and uploads to S3.

        :param data_queue: Queue containing (sym, trade_day, minute_df) tuples
        :param stats: Shared statistics dictionary
        :param stats_lock: Lock for updating statistics
        """
        while True:
            try:
                item = data_queue.get(timeout=1)
                if item is None:  # Poison pill to stop worker
                    break

                sym, trade_day, minute_df = item

                try:
                    # Skip if DataFrame is empty (overwrite check already done before fetching)
                    if len(minute_df) == 0:
                        with stats_lock:
                            stats['skipped'] += 1
                        continue

                    # Round numeric fields to 6 decimal places
                    for field in ['open', 'high', 'low', 'close']:
                        if field in minute_df.columns:
                            minute_df = minute_df.with_columns(
                                minute_df[field].round(4)
                            )

                    # Setup S3 message
                    buffer = io.BytesIO()
                    minute_df.write_parquet(buffer)
                    buffer.seek(0)

                    # Parse date for S3 key
                    date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
                    year = date_obj.strftime('%Y')
                    month = date_obj.strftime('%m')
                    day = date_obj.strftime('%d')

                    s3_key = f"data/raw/ticks/minute/{sym}/{year}/{month}/{day}/ticks.parquet"
                    s3_metadata = {
                        'symbol': sym,
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

    def _fetch_single_symbol_minute(self, symbol: str, year: int, month: int, sleep_time: float = 0.1) -> List[dict]:
        """
        Fetch minute data for a single symbol for the specified month from Alpaca.

        :param symbol: Symbol in Alpaca format (e.g., 'AAPL')
        :param year: Year to fetch
        :param month: Month to fetch (1-12)
        :param sleep_time: Sleep time between requests in seconds (default: 0.1)
        :return: List of bars
        """
        # Get month range
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        start_str = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

        # Prepare request
        base_url = "https://data.alpaca.markets/v2/stocks/bars"

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

        session = requests.Session()
        session.headers.update(self.headers)

        try:
            # Initial request
            response = session.get(base_url, params=params)
            time.sleep(sleep_time)  # Rate limiting

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
                response = session.get(base_url, params=params)
                time.sleep(sleep_time)  # Rate limiting

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

    def _fetch_minute_bulk(self, symbols: List[str], year: int, month: int, sleep_time: float = 0.2) -> dict:
        """
        Bulk fetch minute data for multiple symbols for the specified month from Alpaca.
        If bulk fetch fails, retry by fetching symbols one by one.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param year: Year to fetch
        :param month: Month to fetch (1-12)
        :param sleep_time: Sleep time between requests in seconds (default: 0.2)
        :return: Dict mapping symbol -> list of bars
        """
        # Get month range
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        start_str = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

        # Prepare request
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        symbols_str = ",".join(symbols)
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

        # Use persistent session to reuse TCP connections (avoids handshake overhead)
        session = requests.Session()
        session.headers.update(self.headers)

        for retry in range(max_retries):
            try:
                # Initial request (using session for connection reuse)
                response = session.get(base_url, params=params)
                time.sleep(sleep_time)  # Rate limiting

                if response.status_code == 429:
                    # Rate limit error - use exponential backoff with longer waits
                    wait_time = min(60, (2 ** retry) * 5)  # 5s, 10s, 20s (capped at 60s)
                    self.logger.warning(f"Rate limit hit for bulk fetch (retry {retry + 1}/{max_retries}), waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code != 200:
                    self.logger.error(f"Bulk fetch error for {symbols}: {response.status_code}, {response.text}")
                    if retry < max_retries - 1:
                        wait_time = (2 ** retry) * 2  # 2s, 4s, 8s
                        self.logger.warning(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
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
                    response = session.get(base_url, params=params)
                    time.sleep(sleep_time)  # Rate limiting

                    if response.status_code == 429:
                        # Rate limit during pagination - wait and retry this page
                        wait_time = 5
                        self.logger.warning(f"Rate limit hit during pagination on page {page_count}, waiting {wait_time}s")
                        time.sleep(wait_time)
                        response = session.get(base_url, params=params)
                        time.sleep(sleep_time)

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
                    wait_time = (2 ** retry) * 2
                    self.logger.warning(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)

        # Close the session after retry loop
        session.close()

        # If bulk fetch failed after retries, fetch symbols one by one
        if not bulk_success:
            self.logger.warning(f"Bulk fetch failed after {max_retries} retries, fetching symbols individually")
            failed_symbols = []

            for sym in symbols:
                try:
                    bars = self._fetch_single_symbol_minute(sym, year, month, sleep_time=sleep_time)
                    all_bars[sym] = bars
                    if not bars:
                        self.logger.warning(f"No data returned for {sym}")
                except Exception as e:
                    self.logger.error(f"Failed to fetch {sym} individually: {e}")
                    failed_symbols.append(sym)

            if failed_symbols:
                self.logger.error(f"Failed to fetch {len(failed_symbols)} symbols even after individual retry: {failed_symbols}")

        return all_bars

    def minute_ticks(self, year: int, month: int, overwrite: bool = False, num_workers: int = 50, chunk_size: int = 30, sleep_time: float = 0.2):
        """
        Upload minute ticks using bulk fetch + concurrent processing.
        Fetches 30 symbols at a time for the specified month, then processes concurrently.

        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        :param num_workers: Number of concurrent processing workers (default: 50)
        :param chunk_size: Number of symbols to fetch at once (default: 30)
        :param sleep_time: Sleep time between API requests in seconds (default: 0.2)
        """
        # Load symbols for this month
        alpaca_symbols = self._load_symbols(year, month, sym_type='alpaca')

        # Update trading days for the specified month
        trading_days = self._load_trading_days(year, month)

        total_symbols = len(alpaca_symbols)
        total_days = len(trading_days)
        total_tasks = total_symbols * total_days

        # Shared statistics
        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()

        # Queue for passing data from parser to upload workers
        data_queue = queue.Queue(maxsize=200)

        self.logger.info(f"Starting {year}-{month:02d} minute ticks upload for {total_symbols} symbols × {total_days} days = {total_tasks} tasks")
        self.logger.info(f"Bulk fetching {chunk_size} symbols at a time | {num_workers} concurrent processors | sleep_time={sleep_time}s")

        # Start consumer threads
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=self._upload_minute_ticks_worker,
                args=(data_queue, stats, stats_lock),
                daemon=True
            )
            worker.start()
            workers.append(worker)

        # Producer: Bulk fetch and parse
        try:
            for i in range(0, total_symbols, chunk_size):
                chunk = alpaca_symbols[i:i + chunk_size]
                chunk_num = i // chunk_size + 1

                # Pre-filter symbols that need data (skip if all days exist and overwrite=False)
                if not overwrite:
                    symbols_to_fetch = []
                    for sym in chunk:
                        # Check if any day is missing for this symbol
                        needs_fetch = False
                        for day in trading_days:
                            if not self.validator.data_exists(sym, 'ticks', day=day):
                                needs_fetch = True
                                break
                        if needs_fetch:
                            symbols_to_fetch.append(sym)
                        else:
                            # All days exist, mark them all as canceled
                            for day in trading_days:
                                with stats_lock:
                                    stats['canceled'] += 1
                                    stats['completed'] += 1

                    if not symbols_to_fetch:
                        self.logger.info(f"Skipping chunk {chunk_num}: all data already exists")
                        continue

                    self.logger.info(f"Fetching chunk {chunk_num}/{(total_symbols + chunk_size - 1) // chunk_size}: {len(symbols_to_fetch)}/{len(chunk)} symbols (skipped {len(chunk) - len(symbols_to_fetch)})")
                    chunk = symbols_to_fetch
                else:
                    self.logger.info(f"Fetching chunk {chunk_num}/{(total_symbols + chunk_size - 1) // chunk_size}: {len(chunk)} symbols")

                # Bulk fetch for the month
                start = time.perf_counter()
                symbol_bars = self._fetch_minute_bulk(chunk, year, month, sleep_time=sleep_time)
                self.logger.debug(f"_fetch_minute_bulk: {time.perf_counter()-start:.2f}s")
                start = time.perf_counter()
                # Parse and organize by (symbol, day) - OPTIMIZED with vectorized operations
                for sym in chunk:
                    bars = symbol_bars.get(sym, [])

                    if not bars:
                        # No data for this symbol
                        for day in trading_days:
                            with stats_lock:
                                stats['skipped'] += 1
                                stats['completed'] += 1
                        continue

                    try:
                        # Convert all bars to DataFrame at once using vectorized operations
                        timestamps = [bar[TickField.TIMESTAMP.value] for bar in bars]
                        opens = [bar[TickField.OPEN.value] for bar in bars]
                        highs = [bar[TickField.HIGH.value] for bar in bars]
                        lows = [bar[TickField.LOW.value] for bar in bars]
                        closes = [bar[TickField.CLOSE.value] for bar in bars]
                        volumes = [bar[TickField.VOLUME.value] for bar in bars]
                        num_trades_list = [bar[TickField.NUM_TRADES.value] for bar in bars]
                        vwaps = [bar[TickField.VWAP.value] for bar in bars]

                        # Create DataFrame and process with vectorized operations
                        all_bars_df = pl.DataFrame({
                            'timestamp_utc': timestamps,
                            'open': opens,
                            'high': highs,
                            'low': lows,
                            'close': closes,
                            'volume': volumes,
                            'num_trades': num_trades_list,
                            'vwap': vwaps
                        }, strict=False).with_columns([
                            # Parse timestamp: UTC -> ET, remove timezone
                            # Use strptime with explicit format and timezone to handle 'Z' marker
                            pl.col('timestamp_utc')
                                .str.strptime(pl.Datetime('us', 'UTC'), format='%Y-%m-%dT%H:%M:%SZ')
                                .dt.convert_time_zone('America/New_York')
                                .dt.replace_time_zone(None)
                                .alias('timestamp'),
                            # Extract trade date for filtering by day (fast string slice)
                            pl.col('timestamp_utc')
                                .str.slice(0, 10)  # Just extract 'YYYY-MM-DD' from '2024-01-03T14:30:00Z'
                                .alias('trade_date'),
                            # Cast types (vectorized)
                            pl.col('open').cast(pl.Float64),
                            pl.col('high').cast(pl.Float64),
                            pl.col('low').cast(pl.Float64),
                            pl.col('close').cast(pl.Float64),
                            pl.col('volume').cast(pl.Int64),
                            pl.col('num_trades').cast(pl.Int64),
                            pl.col('vwap').cast(pl.Float64)
                        ]).drop('timestamp_utc')

                        # Process each trading day by filtering
                        for day in trading_days:
                            day_df = all_bars_df.filter(pl.col('trade_date') == day)

                            if len(day_df) > 0:
                                # Select final columns for upload
                                minute_df = day_df.select([
                                    'timestamp', 'open', 'high', 'low', 'close',
                                    'volume', 'num_trades', 'vwap'
                                ])
                            else:
                                # Empty DataFrame for days with no data
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
                        self.logger.error(f"Error processing bars for {sym}: {e}", exc_info=True)
                        # Mark all days as failed for this symbol
                        for day in trading_days:
                            data_queue.put((sym, day, pl.DataFrame()))
                            with stats_lock:
                                stats['failed'] += 1
                                stats['completed'] += 1
                self.logger.debug(f"loop: {time.perf_counter()-start:.2f}s")
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
    def _process_symbol_fundamental(self, sym: str, year: int, month: int, dei_fields: List[str], gaap_fields: List[str], overwrite: bool = False) -> dict:
        """
        Process fundamental data for a single symbol.
        Returns dict with status for progress tracking.

        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        if not overwrite and self.validator.data_exists(sym, 'fundamental', year):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for {year}-{month:02d} already exists'}

        cik = None  # Initialize to avoid unbound error
        try:
            # Get CIK using SecurityMaster (handles historical symbols)
            # Use middle of the month as reference date
            reference_date = dt.date(year, month, 15).strftime('%Y-%m-%d')
            cik = self._get_cik(sym, reference_date)

            if cik is None:
                return {'symbol': sym, 'status': 'skipped', 'error': f'No CIK found for {sym} at {reference_date}'}

            # Fetch from SEC EDGAR API
            fnd = Fundamental(cik, sym)

            # Build fields_dict using new API
            fields_dict = {}

            # Collect DEI fields
            for field in dei_fields:
                try:
                    dps = fnd.get_dps(field, 'dei')
                    fields_dict[field] = fnd.get_value_tuple(dps)
                except KeyError:
                    # Field not available for this company
                    fields_dict[field] = []

            # Collect US-GAAP fields
            for field in gaap_fields:
                try:
                    dps = fnd.get_dps(field, 'us-gaap')
                    fields_dict[field] = fnd.get_value_tuple(dps)
                except KeyError:
                    # Field not available for this company
                    fields_dict[field] = []

            # Define date range for the month
            start_date_obj = dt.date(year, month, 1)
            if month == 12:
                end_date_obj = dt.date(year, 12, 31)
            else:
                end_date_obj = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

            start_day = start_date_obj.strftime('%Y-%m-%d')
            end_day = end_date_obj.strftime('%Y-%m-%d')

            # Collect fields into DataFrame
            combined_df = fnd.collect_fields(start_day, end_day, fields_dict)

            # Setup S3 message
            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            s3_key = f"data/raw/fundamental/{sym}/{year}/{month:02d}/fundamental.parquet"
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
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.warning(f'Failed to fetch data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except ValueError as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Invalid data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Unexpected error for {sym}{cik_str}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def fundamental(self, year: int, month: int, max_workers: int = 20, overwrite: bool = False):
        """
        Upload fundamental data for all symbols using threading.

        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :param max_workers: Number of concurrent threads (default: 20)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        # Load symbols for this month
        sec_symbols = self._load_symbols(year, month, sym_type='sec')

        # Fields
        dei_fields = self.config.dei_fields
        gaap_fields = self.config.us_gaap_fields

        total = len(sec_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0

        self.logger.info(f"Starting {year}-{month:02d} fundamental upload for {total} symbols with {max_workers} workers (overwrite={overwrite})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_symbol_fundamental, sym, year, month, dei_fields, gaap_fields, overwrite): sym
                for sym in sec_symbols
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
            metadata: Optional[Dict[str, str]] = None
        ) -> None:
        """Upload file object to S3 with proper configuration"""

        # Define transfer config
        cfg = self.config.transfer
        transfer_config = TransferConfig(
            multipart_threshold=int(cfg.get('multipart_threshold', 10485760)),
            max_concurrency=int(cfg.get('max_concurrency', 5)),
            multipart_chunksize=int(cfg.get('multipart_chunksize', 10485760)),
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
    
    def run(
            self,
            start_year: int,
            start_month: int,
            end_year: int,
            end_month: int,
            max_workers: int=50,
            overwrite: bool=False,
            chunk_size: int=30,
            sleep_time: float=0.02
        ) -> None:
        """
        Run the complete workflow, fetch and upload fundamental, daily ticks and minute ticks data within the period

        :param start_year: Starting year
        :param start_month: Starting month (1-12)
        :param end_year: Ending year
        :param end_month: Ending month (1-12)
        :param max_workers: Number of concurrent workers
        :param overwrite: If True, overwrite existing data
        :param chunk_size: Number of symbols to fetch at once for minute data
        :param sleep_time: Sleep time between API requests
        """
        year = start_year
        month = start_month

        while (year < end_year) or (year == end_year and month <= end_month):
            self.logger.info(f"Processing {year}-{month:02d}")

            # Upload fundamental data
            self.fundamental(year, month, max_workers, overwrite)

            # Upload daily ticks
            self.upload_daily_ticks(year, month, overwrite)

            # Upload minute ticks (only for years >= 2017)
            if year >= 2017:
                self.minute_ticks(year, month, overwrite, max_workers, chunk_size, sleep_time)

            # Move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

    def close(self):
        """Close WRDS database connections"""
        if hasattr(self, 'crsp_ticks') and self.crsp_ticks is not None:
            if hasattr(self.crsp_ticks, 'conn') and self.crsp_ticks.conn is not None:
                self.crsp_ticks.conn.close()
                self.logger.info("WRDS connection closed")


if __name__ == "__main__":
    app = UploadApp()
    try:
        # Example: Run from January 2010 to December 2025
        app.run(start_year=2010, start_month=1, end_year=2025, end_month=12, overwrite=True)
        # app.daily_ticks(year=2010, month=1, overwrite=True)
    finally:
        app.close()