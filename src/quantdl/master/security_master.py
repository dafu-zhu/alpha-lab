import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import polars as pl
import pyarrow.parquet as pq
import requests
import yaml
from dotenv import load_dotenv

from quantdl.storage.utils import RateLimiter
from quantdl.universe.current import fetch_all_stocks
from quantdl.utils.logger import setup_logger

load_dotenv()

# Source parquet bundled with the package
_SOURCE_MASTER_PATH = Path(__file__).resolve().parent.parent / "data" / "security_master.parquet"

# Default local path for security master parquet
# Prefer working copy under LOCAL_STORAGE_PATH, fall back to bundled source
_storage_base = os.getenv("LOCAL_STORAGE_PATH", "")
if _storage_base:
    LOCAL_MASTER_PATH = Path(_storage_base) / "data" / "meta" / "master" / "security_master.parquet"
else:
    LOCAL_MASTER_PATH = _SOURCE_MASTER_PATH

# Morningstar→GICS mapping file
GICS_MAPPING_PATH = Path("configs/morningstar_to_gics.yaml")


def _normalize_ticker(ticker: str) -> str:
    """Remove separators and uppercase: 'BRK.B' -> 'BRKB', 'BRK-B' -> 'BRKB'."""
    return ticker.replace('.', '').replace('-', '').upper()


# OpenFIGI rate limits (with API key: 25 req/6s, 100 jobs/req)
OPENFIGI_RATE_LIMIT_NO_KEY = 25 / 60  # ~0.42 req/sec (no key)
OPENFIGI_RATE_LIMIT_WITH_KEY = 25 / 6  # ~4.17 req/sec (with key)
OPENFIGI_BATCH_SIZE = 100  # Max tickers per request (with API key)
OPENFIGI_BATCH_SIZE_NO_KEY = 10  # Smaller batch without key to avoid 413
OPENFIGI_MAX_RETRIES = 3  # Max retries per batch on transient errors


class SymbolNormalizer:
    """
    Deterministic symbol normalizer based on current Nasdaq stock list with SecurityMaster validation.

    Strategy:
    1. Load current stock list from Nasdaq (via fetch_all_stocks)
    2. For any incoming symbol with date context, verify security_id matches
    3. Prevents false matches (e.g., delisted ABCD ≠ current ABC.D)
    4. If verified same security: return Nasdaq format
    5. If different security or delisted: return original format

    Edge case handling:
        - ABCD (2021-2023, delisted, security_id=1000)
        - ABC.D (2025+, active, security_id=2000)
        - Both normalize to "ABCD" but different security_id
        - Solution: Keep historical ABCD as-is, don't convert to ABC.D

    Note:
        Nasdaq only covers currectly active stocks. If a stock is delisted, keep it in CRSP format as is. The symbol is for naming the storage folder. This is because the delisted stocks won't be updated, therefore they won't need to match the Nasdaq list.

    Examples:
        to_nasdaq_format('BRKB', '2024-01-01') -> 'BRK.B' (same security)
        to_nasdaq_format('ABCD', '2022-01-01') -> 'ABCD' (delisted, different from ABC.D)
    """

    # CRSP data coverage end date
    CRSP_LATEST_DATE = '2024-12-31'

    def __init__(self, security_master: Optional['SecurityMaster'] = None):
        """
        Initialize with current Nasdaq stock list and optional SecurityMaster.

        :param security_master: SecurityMaster instance for validation (optional)
        """
        # Load current stock list (cached in data/meta/universe/stock_exchange.csv)
        self.current_stocks_df = fetch_all_stocks(with_filter=True, refresh=False)

        # Create normalized lookup: {crsp_format: nasdaq_format}
        # e.g., {'BRKB': 'BRK.B', 'AAPL': 'AAPL', 'GOOGL': 'GOOGL'}
        self.sym_map = {}
        for ticker in self.current_stocks_df['Ticker']:
            # Skip NaN or non-string values
            if not isinstance(ticker, str):
                continue
            crsp_key = _normalize_ticker(ticker)
            self.sym_map[crsp_key] = ticker

        self.security_master = security_master
        self.logger = setup_logger(
            name="master.SecurityNormalizer",
            log_dir=Path("data/logs/master"),
            level=logging.INFO
        )

    def to_nasdaq_format(self, symbol: str, day: Optional[str] = None) -> str:
        """
        Normalize symbol to Nasdaq format with security_id validation.

        :param symbol: Ticker symbol in any format (BRKB, BRK.B, BRK-B)
        :param day: Date context for validation (format: 'YYYY-MM-DD', optional)
        :return: Nasdaq format if same security, otherwise original

        Examples:
            to_nasdaq_format('BRKB', '2024-01-01') -> 'BRK.B' (verified same security)
            to_nasdaq_format('BRKB') -> 'BRK.B' (no validation, assume same)
            to_nasdaq_format('ABCD', '2022-01-01') -> 'ABCD' (different security, keep original)
        """
        if not symbol:
            return symbol

        crsp_key = _normalize_ticker(symbol)

        # Check if exists in current stock list
        if crsp_key not in self.sym_map:
            # Not in current list (delisted), return as-is
            return symbol.upper()

        nasdaq_format = self.sym_map[crsp_key]

        # If no date context or no SecurityMaster, return Nasdaq format (assume same security)
        if day is None or self.security_master is None:
            return nasdaq_format

        # Validate using SecurityMaster: check if same security
        try:
            # Get security_id for original symbol at given date
            original_sid = self.security_master.get_security_id(
                symbol=crsp_key,  # Use CRSP format for lookup
                day=day,
                auto_resolve=False  # Strict match only
            )

            # Get security_id for Nasdaq format at CRSP latest date
            nasdaq_sid = self.security_master.get_security_id(
                symbol=crsp_key,  # Use CRSP format for lookup
                day=self.CRSP_LATEST_DATE,
                auto_resolve=False
            )

            # If same security_id, safe to convert to Nasdaq format
            if original_sid == nasdaq_sid:
                return nasdaq_format
            else:
                # Different securities, keep original format
                return symbol.upper()

        except ValueError:
            self.logger.error(f"Symbol {symbol} not found in SecurityMaster at one of the dates, keep original")
            return symbol.upper()

    def batch_normalize(
        self,
        symbols: List[str],
        day: Optional[str] = None
    ) -> List[str]:
        """
        Normalize a batch of symbols with optional date validation.

        :param symbols: List of ticker symbols in any format
        :param day: Date context for validation (format: 'YYYY-MM-DD', optional)
        :return: List of normalized symbols (Nasdaq format if verified)
        """
        return [self.to_nasdaq_format(sym, day) for sym in symbols]

    @staticmethod
    def to_crsp_format(symbol: str) -> str:
        """
        Convert any format to CRSP format (remove separators).

        :param symbol: Ticker in any format (e.g., BRK.B, BRK-B, BRKB)
        :return: CRSP format (e.g., BRKB)
        """
        return _normalize_ticker(symbol)

    @staticmethod
    def to_sec_format(symbol: str) -> str:
        """
        Convert Nasdaq format to SEC format (period -> hyphen).

        :param symbol: Ticker in Nasdaq format (e.g., BRK.B)
        :return: SEC format (e.g., BRK-B)
        """
        return symbol.replace('.', '-').upper()


class SecurityMaster:
    """
    Map stock symbols, CIKs across time horizon.

    Loads from local parquet (built by scripts/build_security_master.py).
    Updates via SEC/OpenFIGI/yfinance (no WRDS dependency).
    """
    # CRSP data coverage end date
    CRSP_LATEST_DATE = '2024-12-31'

    def __init__(self, local_path: Optional[Path] = None):
        """
        Initialize SecurityMaster from local parquet.

        :param local_path: Path to local parquet file (default: data/meta/master/security_master.parquet)
        """
        self.logger = setup_logger(
            name="master.SecurityMaster",
            log_dir=Path("data/logs/master"),
            level=logging.INFO,
            console_output=True
        )

        # Cache for GICS mapping (loaded on-demand)
        self._gics_mapping: Optional[Dict] = None

        if local_path is None:
            local_path = LOCAL_MASTER_PATH

        self.local_path = local_path
        # prev_universe.json lives next to the security_master.parquet
        self._prev_universe_path = local_path.parent / "prev_universe.json"

        if not local_path.exists():
            raise FileNotFoundError(
                f"Security master not found at {local_path}. "
                "Run: python scripts/build_security_master.py or qdl --master"
            )

        self.master_tb = self._load_from_local(local_path)
        self.logger.info(f"Loaded SecurityMaster ({len(self.master_tb)} rows)")

    @staticmethod
    def _load_from_local(path: Path) -> pl.DataFrame:
        """Load master_tb from local parquet file."""
        df = pl.read_parquet(str(path))
        return SecurityMaster._ensure_schema(df)

    @staticmethod
    def _ensure_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure required columns exist and drop cusip if present."""
        for col in ("exchange", "sector", "industry", "subindustry"):
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
        if "cusip" in df.columns:
            df = df.drop("cusip")
        return df

    @staticmethod
    def _enrich_row_from_sec(
        row_dict: dict,
        sec_info: Optional[Dict[str, Optional[str]]],
    ) -> None:
        """Fill null exchange/cik/company fields from SEC data (mutates row_dict)."""
        if not sec_info:
            return
        if not row_dict.get('exchange'):
            row_dict['exchange'] = sec_info.get('exchange')
        if not row_dict.get('cik'):
            row_dict['cik'] = sec_info.get('cik')
        if not row_dict.get('company') or row_dict['company'] == '':
            row_dict['company'] = sec_info.get('company')

    def _fetch_sec_exchange_mapping(self) -> pl.DataFrame:
        """
        Fetch SEC company tickers with exchange info.

        Uses company_tickers_exchange.json which provides exchange field.
        Returns DataFrame with columns: [ticker, cik, company, exchange]
        """
        url = "https://www.sec.gov/files/company_tickers_exchange.json"
        headers = {'User-Agent': os.getenv('SEC_USER_AGENT', 'name@example.com')}

        self.logger.info("Fetching SEC company tickers with exchange info...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        # Columnar format: {"fields": ["cik", "name", "ticker", "exchange"], "data": [[...], ...]}
        fields = data['fields']
        cik_idx = fields.index('cik')
        name_idx = fields.index('name')
        ticker_idx = fields.index('ticker')
        exchange_idx = fields.index('exchange')

        records = []
        for row in data['data']:
            cik_raw = row[cik_idx]
            ticker_raw = str(row[ticker_idx])
            name = str(row[name_idx])
            exchange = str(row[exchange_idx]) if row[exchange_idx] else None

            ticker = _normalize_ticker(ticker_raw)
            cik = str(cik_raw).zfill(10)

            if ticker and cik != '0000000000':
                records.append({
                    'ticker': ticker,
                    'cik': cik,
                    'company': name,
                    'exchange': exchange,
                })

        df = pl.DataFrame(records)
        self.logger.info(f"Loaded {len(df)} tickers from SEC (with exchange)")
        return df

    def _fetch_yfinance_metadata(self, tickers: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Fetch sector/industry from yfinance for given tickers.

        :param tickers: List of ticker symbols (Nasdaq format preferred)
        :return: {ticker_crsp: {"sector": ..., "industry": ...}} (Morningstar names)
        """
        try:
            import yfinance as yf
        except ImportError:
            self.logger.warning("yfinance not installed, skipping classification lookup")
            return {}

        import io
        import sys

        results: Dict[str, Dict[str, str]] = {}
        total = len(tickers)
        self.logger.info(f"Fetching yfinance metadata for {total} tickers...")

        for i, ticker in enumerate(tickers):
            try:
                yf_ticker = ticker.replace('.', '-')
                # Suppress yfinance 404 stderr noise
                saved_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    info = yf.Ticker(yf_ticker).info
                finally:
                    sys.stderr = saved_stderr
                sector = info.get('sector')
                industry = info.get('industry')
                if sector or industry:
                    results[_normalize_ticker(ticker)] = {
                        'sector': sector or '',
                        'industry': industry or '',
                    }
                time.sleep(0.3)
            except Exception as e:
                self.logger.debug(f"yfinance failed for {ticker}: {e}")

            if (i + 1) % 25 == 0 or i + 1 == total:
                self.logger.info(f"yfinance progress: {i + 1}/{total}")

        self.logger.info(f"yfinance: got metadata for {len(results)}/{total} tickers")
        return results

    def _load_gics_mapping(self) -> Dict:
        """Load Morningstar→GICS mapping from YAML (cached)."""
        if self._gics_mapping is not None:
            return self._gics_mapping

        if not GICS_MAPPING_PATH.exists():
            self.logger.warning(f"GICS mapping not found at {GICS_MAPPING_PATH}")
            self._gics_mapping = {'sectors': {}, 'industries': {}}
            return self._gics_mapping

        with open(GICS_MAPPING_PATH, 'r', encoding='utf-8') as f:
            self._gics_mapping = yaml.safe_load(f)

        return self._gics_mapping

    def _map_to_gics(self, ms_sector: str, ms_industry: str) -> Dict[str, Optional[str]]:
        """
        Map Morningstar classification to GICS.

        :param ms_sector: Morningstar sector name (e.g., "Technology")
        :param ms_industry: Morningstar industry name (e.g., "Semiconductors")
        :return: {"sector": ..., "industry": ..., "subindustry": ...}
        """
        mapping = self._load_gics_mapping()

        # Try industry-level mapping first (most precise)
        if ms_industry and ms_industry in mapping.get('industries', {}):
            ind = mapping['industries'][ms_industry]
            return {
                'sector': ind.get('gics_sector'),
                'industry': ind.get('gics_industry_group'),
                'subindustry': ind.get('gics_sub_industry'),
            }

        # Fall back to sector-level mapping
        if ms_sector and ms_sector in mapping.get('sectors', {}):
            return {
                'sector': mapping['sectors'][ms_sector],
                'industry': None,
                'subindustry': None,
            }

        return {'sector': None, 'industry': None, 'subindustry': None}

    def get_securities_in_range(
        self, start_year: int, end_year: int
    ) -> List[Tuple[str, int]]:
        """
        Return all (symbol, security_id) pairs active in [start_year, end_year].

        For securities with multiple symbols (e.g., FB→META), returns the
        latest symbol — Alpaca returns full history for the current symbol.

        :param start_year: Start year (inclusive)
        :param end_year: End year (inclusive)
        :return: List of (symbol, security_id) tuples sorted by security_id
        """
        year_start = dt.date(start_year, 1, 1)
        year_end = dt.date(end_year, 12, 31)

        active = self.master_tb.filter(
            pl.col('start_date').le(year_end),
            pl.col('end_date').ge(year_start),
        )

        # Per security_id, pick row with latest end_date → most recent symbol
        result = (
            active
            .sort('end_date', descending=True)
            .unique(subset=['security_id'], keep='first')
            .select(['symbol', 'security_id'])
            .sort('security_id')
        )

        return list(result.iter_rows())

    def auto_resolve(self, symbol: str, day: str) -> int:
        """
        Resolve an unmatched symbol to the security active on *day* that most
        recently used (or will use) *symbol*.
        """
        date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()

        candidates = (
            self.master_tb.filter(pl.col('symbol').eq(symbol))
            .select('security_id')
            .unique()
            .filter(pl.col('security_id').is_not_null())
        )

        if candidates.is_empty():
            self.logger.debug(f"auto_resolve failed: symbol '{symbol}' never existed in security master")
            raise ValueError(f"Symbol '{symbol}' never existed in security master")

        # For each candidate, check if it was active on target date (under ANY symbol)
        active_securities = []
        null_candidates = 0
        for candidate_sid in candidates['security_id']:
            if candidate_sid is None:
                null_candidates += 1
                continue
            was_active = self.master_tb.filter(
                pl.col('security_id').eq(candidate_sid),
                pl.col('start_date').le(date_check),
                pl.col('end_date').ge(date_check),
            )
            if was_active.is_empty():
                continue
            symbol_usage = self.master_tb.filter(
                pl.col('security_id').eq(candidate_sid),
                pl.col('symbol').eq(symbol),
            ).select(['start_date', 'end_date']).head(1)

            active_securities.append({
                'sid': candidate_sid,
                'symbol_start': symbol_usage['start_date'][0],
                'symbol_end': symbol_usage['end_date'][0],
            })

        if null_candidates > 0:
            self.logger.warning(
                f"auto_resolve: filtered {null_candidates} null security_id values for symbol='{symbol}'"
            )

        if len(active_securities) == 0:
            self.logger.debug(
                f"auto_resolve failed: symbol '{symbol}' exists but associated security "
                f"was not active on {day}"
            )
            raise ValueError(
                f"Symbol '{symbol}' exists but the associated security was not active on {day}"
            )

        if len(active_securities) == 1:
            sid = active_securities[0]['sid']
        else:
            # Multiple candidates -- pick the one whose symbol usage is closest to query date
            def distance_to_date(sec: dict) -> int:
                if date_check < sec['symbol_start']:
                    return (sec['symbol_start'] - date_check).days
                if date_check > sec['symbol_end']:
                    return (date_check - sec['symbol_end']).days
                return 0

            best_match = min(active_securities, key=distance_to_date)
            sid = best_match['sid']
            self.logger.info(f"auto_resolve: Multiple candidates found, selected security_id={sid}")

        try:
            cik = self.sid_to_info(sid, day, info='cik')
            company = self.sid_to_info(sid, day, info='company')
            self.logger.info(
                f"auto_resolve triggered for symbol='{symbol}' ({company}) on date='{day}', sid={sid}, cik={cik}"
            )
        except Exception as e:
            self.logger.error(f"auto_resolve triggered for symbol='{symbol}', sid={sid}: {e}")

        return sid

    def get_security_id(self, symbol: str, day: str, auto_resolve: bool = True) -> int:
        """
        Find the security_id for a symbol at a specific point in time.

        :param auto_resolve: If no exact match, resolve to the nearest security
        """
        date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()

        match = self.master_tb.filter(
            pl.col('symbol').eq(symbol),
            pl.col('start_date').le(date_check),
            pl.col('end_date').ge(date_check),
        )

        if match.is_empty():
            if auto_resolve:
                return self.auto_resolve(symbol, day)
            raise ValueError(f"Symbol {symbol} not found in day {day}")

        result = match.head(1).select('security_id').item()
        if result is None:
            raise ValueError(
                f"Symbol '{symbol}' found in security master for {day}, but security_id is None. "
                "This indicates corrupted data in the security master table."
            )
        return result

    def get_symbol_history(self, sid: int) -> List[Tuple[str, str, str]]:
        """
        Return all (symbol, start_iso, end_iso) tuples for a given security_id.

        Example: [('META', '2022-06-09', '2024-12-31'), ('FB', '2012-05-18', '2022-06-08')]
        """
        hist = (
            self.master_tb
            .filter(pl.col('security_id').eq(sid))
            .group_by('symbol')
            .agg(pl.col('start_date').min(), pl.col('end_date').max())
            .select(['symbol', 'start_date', 'end_date'])
            .rows()
        )
        return [(sym, start.isoformat(), end.isoformat()) for sym, start, end in hist]

    def sid_to_info(self, sid: int, day: str, info: str) -> object:
        """Look up a single field for a security_id on a given date."""
        date_obj = dt.datetime.strptime(day, "%Y-%m-%d").date()
        return (
            self.master_tb.filter(
                pl.col('security_id').eq(sid),
                pl.col('start_date').le(date_obj),
                pl.col('end_date').ge(date_obj),
            ).select(info).head(1).item()
        )

    def save_local(self, path: Optional[Path] = None) -> None:
        """
        Persist master_tb as parquet to local filesystem.

        :param path: Target path (default: LOCAL_MASTER_PATH)
        """
        if path is None:
            path = LOCAL_MASTER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        out_df = self.master_tb
        table = out_df.to_arrow()
        metadata = {
            b'crsp_end_date': self.CRSP_LATEST_DATE.encode(),
            b'export_timestamp': dt.datetime.now(dt.timezone.utc).isoformat().encode(),
            b'version': b'1.0',
            b'row_count': str(len(out_df)).encode()
        }
        existing_meta = table.schema.metadata or {}
        combined_meta = {**existing_meta, **metadata}
        table = table.replace_schema_metadata(combined_meta)

        pq.write_table(table, str(path))
        self.logger.info(f"Saved SecurityMaster to {path} ({len(out_df)} rows)")

    def _fetch_openfigi_mapping(
        self,
        tickers: List[str],
        rate_limiter: Optional[RateLimiter] = None
    ) -> Dict[str, Optional[str]]:
        """
        Batch lookup ticker -> shareClassFIGI via OpenFIGI API.

        API: POST https://api.openfigi.com/v3/mapping
        With API key: 25 req/6s, 100 jobs per request
        Without key: 25 req/min, 10 jobs per request

        :param tickers: List of ticker symbols
        :param rate_limiter: Optional RateLimiter instance
        :return: Dict mapping ticker -> shareClassFIGI (None if not found)
        """
        url = "https://api.openfigi.com/v3/mapping"
        headers = {"Content-Type": "application/json"}

        # Use API key if available (higher rate limit and batch size)
        api_key = os.getenv("OPENFIGI_API_KEY")
        if api_key:
            headers["X-OPENFIGI-APIKEY"] = api_key
            max_rate = OPENFIGI_RATE_LIMIT_WITH_KEY
            batch_size = OPENFIGI_BATCH_SIZE
        else:
            max_rate = OPENFIGI_RATE_LIMIT_NO_KEY
            batch_size = OPENFIGI_BATCH_SIZE_NO_KEY

        # Create rate limiter if not provided
        if rate_limiter is None:
            rate_limiter = RateLimiter(max_rate=max_rate)

        results: Dict[str, Optional[str]] = {}
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        self.logger.info(
            f"Starting OpenFIGI lookup for {len(tickers)} tickers "
            f"({total_batches} batches of {batch_size}, API key: {'yes' if api_key else 'no'})"
        )

        # Process in batches
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            batch = tickers[start:start + batch_size]

            # Retry loop with exponential backoff
            for retry in range(OPENFIGI_MAX_RETRIES + 1):
                payload = [
                    {"idType": "TICKER", "idValue": t, "exchCode": "US"}
                    for t in batch
                ]

                try:
                    rate_limiter.acquire()
                    response = requests.post(url, json=payload, headers=headers, timeout=30)

                    if response.status_code == 429 or response.status_code >= 500:
                        if retry < OPENFIGI_MAX_RETRIES:
                            wait_time = 2 ** retry
                            self.logger.warning(
                                f"HTTP {response.status_code}, retry {retry + 1}/{OPENFIGI_MAX_RETRIES} in {wait_time}s"
                            )
                            time.sleep(wait_time)
                            continue
                        # Exhausted retries
                        for t in batch:
                            results[t] = None
                        break

                    response.raise_for_status()
                    data = response.json()

                    # Parse response
                    for j, item in enumerate(data):
                        ticker = batch[j]
                        if "data" in item and item["data"]:
                            results[ticker] = item["data"][0].get("shareClassFIGI")
                        else:
                            results[ticker] = None
                    break  # Success

                except requests.RequestException as e:
                    if retry < OPENFIGI_MAX_RETRIES:
                        wait_time = 2 ** retry
                        self.logger.warning(f"Request error: {e}, retry {retry + 1}/{OPENFIGI_MAX_RETRIES} in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        self.logger.warning(f"Batch failed after {OPENFIGI_MAX_RETRIES + 1} attempts: {e}")
                        for t in batch:
                            results[t] = None

                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}")
                    for t in batch:
                        results[t] = None
                    break

            # Progress logging every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
                pct = ((batch_idx + 1) / total_batches) * 100
                self.logger.info(f"OpenFIGI progress: {batch_idx + 1}/{total_batches} batches ({pct:.0f}%)")

        found = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"OpenFIGI lookup complete: {found}/{len(results)} tickers mapped")

        return results

    def _fetch_nasdaq_universe(self) -> Set[str]:
        """
        Fetch current active stocks from Nasdaq FTP.

        :return: Set of active ticker symbols (Nasdaq format)
        """
        try:
            df = fetch_all_stocks(with_filter=True, refresh=True, logger=self.logger)
            tickers = set(df['Ticker'].tolist())
            self.logger.info(f"Fetched {len(tickers)} active tickers from Nasdaq")
            return tickers
        except Exception as e:
            self.logger.error(f"Failed to fetch Nasdaq universe: {e}")
            return set()

    def _detect_rebrands(
        self,
        disappeared: Set[str],
        appeared: Set[str],
        figi_mapping: Dict[str, Optional[str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Detect rebrands by matching shareClassFIGI between disappeared and appeared tickers.

        :param disappeared: Tickers that were in prev but not in current
        :param appeared: Tickers that are in current but not in prev
        :param figi_mapping: Dict mapping ticker -> shareClassFIGI
        :return: List of (old_ticker, new_ticker, figi) tuples for detected rebrands
        """
        rebrands = []

        # Build reverse lookup: FIGI -> disappeared ticker
        figi_to_old: Dict[str, str] = {}
        for ticker in disappeared:
            figi = figi_mapping.get(ticker)
            if figi:
                figi_to_old[figi] = ticker

        # Check if any appeared ticker has same FIGI as a disappeared ticker
        for ticker in appeared:
            figi = figi_mapping.get(ticker)
            if figi and figi in figi_to_old:
                old_ticker = figi_to_old[figi]
                rebrands.append((old_ticker, ticker, figi))
                self.logger.info(f"Detected rebrand: {old_ticker} -> {ticker} (FIGI: {figi})")

        return rebrands

    def _load_prev_universe(self) -> Tuple[Set[str], Optional[str]]:
        """
        Load previous universe from local JSON file.

        :return: Tuple of (prev_universe set, prev_date string or None)
        """
        try:
            if self._prev_universe_path.exists():
                data = json.loads(self._prev_universe_path.read_text(encoding='utf-8'))
                prev_universe = set(data.get('tickers', []))
                prev_date = data.get('date')
                self.logger.info(f"Loaded prev_universe: {len(prev_universe)} tickers from {prev_date}")
                return prev_universe, prev_date
            else:
                self.logger.info("No prev_universe found, will bootstrap from current Nasdaq list")
                return set(), None
        except Exception as e:
            self.logger.warning(f"Failed to load prev_universe: {e}, bootstrapping")
            return set(), None

    def _save_prev_universe(
        self,
        universe: Set[str],
        date: str
    ) -> None:
        """
        Save current universe to local file for next run.

        :param universe: Set of ticker symbols
        :param date: Date string (YYYY-MM-DD)
        """
        self._prev_universe_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'tickers': sorted(list(universe)),
            'date': date
        }
        self._prev_universe_path.write_text(
            json.dumps(data, indent=2),
            encoding='utf-8'
        )
        self.logger.info(f"Saved prev_universe: {len(universe)} tickers for {date}")

    def update(
        self,
        grace_period_days: int = 14
    ) -> Dict[str, int]:
        """
        Update master_tb using SEC + Nasdaq + OpenFIGI + yfinance.

        Merged algorithm (replaces update_from_sec + update_no_wrds):
        1. Fetch SEC company_tickers_exchange.json → sec_df (ticker, cik, company, exchange)
        2. Fetch Nasdaq FTP universe → current_nasdaq
        3. Load prev_universe.json
        4. If no prev_universe → bootstrap: save current, extend existing, return
        5. Diff: still_active, disappeared, appeared
        6. OpenFIGI for disappeared + appeared → rebrand detection
        7. For truly new IPOs: yfinance → Morningstar → GICS
        8. Apply updates:
           - EXTEND: still_active → end_date=today, fill null exchange/cik/company from SEC
           - REBRAND: same FIGI → close old, create new with same security_id
           - NEW IPO: new FIGI → new security_id + SEC metadata + GICS
           - DELIST: grace period passed → freeze end_date
        9. Save prev_universe + parquet

        :param grace_period_days: Days before treating missing ticker as delisted
        :return: Dict with counts {'extended', 'rebranded', 'added', 'delisted', 'unchanged'}
        """
        today = dt.date.today()
        today_str = today.isoformat()

        stats = {
            'extended': 0,
            'rebranded': 0,
            'added': 0,
            'delisted': 0,
            'unchanged': 0
        }

        # 1. Fetch SEC data with exchange info
        try:
            sec_df = self._fetch_sec_exchange_mapping()
        except Exception as e:
            self.logger.error(f"Failed to fetch SEC data: {e}")
            sec_df = pl.DataFrame({
                'ticker': [], 'cik': [], 'company': [], 'exchange': []
            }, schema={'ticker': pl.Utf8, 'cik': pl.Utf8, 'company': pl.Utf8, 'exchange': pl.Utf8})

        # Build SEC lookup: {ticker_norm: {cik, company, exchange}}
        sec_lookup: Dict[str, Dict[str, Optional[str]]] = {}
        for row in sec_df.iter_rows(named=True):
            sec_lookup[row['ticker']] = {
                'cik': row['cik'],
                'company': row['company'],
                'exchange': row['exchange'],
            }

        # 2. Fetch current Nasdaq universe
        current_nasdaq = self._fetch_nasdaq_universe()
        if not current_nasdaq:
            self.logger.error("Failed to fetch Nasdaq universe, aborting update")
            return stats

        # 3. Load previous universe
        prev_universe, prev_date = self._load_prev_universe()

        # 4. Bootstrap: if no prev_universe, extend existing + enrich from SEC
        if not prev_universe:
            self.logger.info("Bootstrapping: using current Nasdaq list as prev_universe")
            self._save_prev_universe(current_nasdaq, today_str)
            return self._bootstrap_extend(current_nasdaq, today, sec_lookup)

        # 5. Compute changes
        current_normalized = {_normalize_ticker(t): t for t in current_nasdaq}
        prev_normalized = {_normalize_ticker(t): t for t in prev_universe}

        current_set = set(current_normalized.keys())
        prev_set = set(prev_normalized.keys())

        still_active = current_set & prev_set
        disappeared = prev_set - current_set
        appeared = current_set - prev_set

        self.logger.info(
            f"Universe changes: {len(still_active)} active, "
            f"{len(disappeared)} disappeared, {len(appeared)} appeared"
        )

        # 6. Fetch OpenFIGI mappings for disappeared + appeared tickers
        tickers_to_lookup = list(disappeared | appeared)
        figi_mapping: Dict[str, Optional[str]] = {}
        if tickers_to_lookup:
            original_tickers: List[str] = [
                prev_normalized.get(t) or current_normalized[t]
                for t in tickers_to_lookup
            ]
            figi_results = self._fetch_openfigi_mapping(original_tickers)
            for t in tickers_to_lookup:
                orig = prev_normalized.get(t) or current_normalized[t]
                figi_mapping[t] = figi_results.get(orig)

        # 7. Detect rebrands
        rebrands = self._detect_rebrands(disappeared, appeared, figi_mapping)
        rebrand_old = {r[0] for r in rebrands}
        rebrand_new = {r[1] for r in rebrands}

        # 8. For truly new IPOs, fetch yfinance metadata
        new_ipos = appeared - rebrand_new
        yf_metadata: Dict[str, Dict[str, str]] = {}
        if new_ipos:
            # Use Nasdaq-format tickers for yfinance
            yf_tickers = [current_normalized[t] for t in new_ipos]
            yf_metadata = self._fetch_yfinance_metadata(yf_tickers)

        # 9. Process updates
        #    Only extend end_date for "current" rows (end_date >= CRSP_LATEST_DATE).
        #    Historical rows with earlier end_dates are preserved as-is.
        crsp_end = dt.datetime.strptime(self.CRSP_LATEST_DATE, '%Y-%m-%d').date()
        updated_rows = []

        for row in self.master_tb.iter_rows(named=True):
            row_dict = dict(row)
            symbol_norm = _normalize_ticker(row['symbol'])

            if symbol_norm in still_active:
                # Convert CRSP symbol to Nasdaq format
                row_dict['symbol'] = current_normalized[symbol_norm]

                # EXTEND: only if this is a "current" row
                row_end = row_dict.get('end_date')
                if row_end is not None and row_end >= crsp_end:
                    row_dict['end_date'] = today
                    stats['extended'] += 1
                else:
                    stats['unchanged'] += 1
                self._enrich_row_from_sec(row_dict, sec_lookup.get(symbol_norm))

            elif symbol_norm in rebrand_old:
                # REBRAND (old ticker): freeze end_date
                for old, new, figi in rebrands:
                    if old == symbol_norm:
                        break
                stats['rebranded'] += 1

            elif symbol_norm in disappeared:
                # DELIST: check grace period
                if prev_date:
                    prev_dt = dt.datetime.strptime(prev_date, '%Y-%m-%d').date()
                    days_missing = (today - prev_dt).days
                    if days_missing < grace_period_days:
                        row_dict['end_date'] = today
                        stats['extended'] += 1
                    else:
                        stats['delisted'] += 1
                else:
                    stats['unchanged'] += 1
            else:
                stats['unchanged'] += 1

            # Drop cusip from row if present
            row_dict.pop('cusip', None)
            updated_rows.append(row_dict)

        # 10. Add rebrand new rows (same security_id as old)
        for old_norm, new_norm, figi in rebrands:
            old_row = self.master_tb.filter(
                pl.col('symbol').str.replace_all(r'[.\-]', '').str.to_uppercase() == old_norm
            ).head(1)

            if old_row.is_empty():
                self.logger.warning(f"Rebrand old ticker {old_norm} not found in master_tb")
                continue

            old_security_id = old_row['security_id'][0]
            new_ticker = current_normalized[new_norm]
            sec_info = sec_lookup.get(new_norm, {})

            # Get GICS for rebranded ticker from yfinance
            yf_rebrand = self._fetch_yfinance_metadata([new_ticker])
            gics = {'sector': None, 'industry': None, 'subindustry': None}
            yf_key = _normalize_ticker(new_ticker)
            if yf_key in yf_rebrand:
                gics = self._map_to_gics(
                    yf_rebrand[yf_key].get('sector', ''),
                    yf_rebrand[yf_key].get('industry', ''),
                )

            new_row = {
                'security_id': old_security_id,
                'permno': old_row['permno'][0] if 'permno' in old_row.columns else None,
                'symbol': new_ticker,
                'company': sec_info.get('company') or (old_row['company'][0] if 'company' in old_row.columns else ''),
                'cik': sec_info.get('cik') or (old_row['cik'][0] if 'cik' in old_row.columns else None),
                'share_class_figi': figi,
                'start_date': today,
                'end_date': today,
                'exchange': sec_info.get('exchange') or (old_row['exchange'][0] if 'exchange' in old_row.columns else None),
                'sector': gics.get('sector') or (old_row['sector'][0] if 'sector' in old_row.columns else None),
                'industry': gics.get('industry') or (old_row['industry'][0] if 'industry' in old_row.columns else None),
                'subindustry': gics.get('subindustry') or (old_row['subindustry'][0] if 'subindustry' in old_row.columns else None),
            }
            updated_rows.append(new_row)

        # 11. Add truly new IPOs
        max_sid: int = self.master_tb['security_id'].max() or 1000  # type: ignore[assignment]

        for ticker_norm in new_ipos:
            ticker = current_normalized[ticker_norm]
            figi = figi_mapping.get(ticker_norm)
            sec_info = sec_lookup.get(ticker_norm, {})

            # Map yfinance → GICS
            gics = {'sector': None, 'industry': None, 'subindustry': None}
            if ticker_norm in yf_metadata:
                gics = self._map_to_gics(
                    yf_metadata[ticker_norm].get('sector', ''),
                    yf_metadata[ticker_norm].get('industry', ''),
                )

            max_sid += 1
            new_row = {
                'security_id': max_sid,
                'permno': None,
                'symbol': ticker,
                'company': sec_info.get('company', ''),
                'cik': sec_info.get('cik'),
                'share_class_figi': figi,
                'start_date': today,
                'end_date': today,
                'exchange': sec_info.get('exchange'),
                'sector': gics.get('sector'),
                'industry': gics.get('industry'),
                'subindustry': gics.get('subindustry'),
            }
            updated_rows.append(new_row)
            stats['added'] += 1

        # 12. Rebuild master_tb
        if updated_rows:
            self.master_tb = pl.DataFrame(updated_rows).cast({
                'security_id': pl.Int64,
                'start_date': pl.Date,
                'end_date': pl.Date
            })

            # Ensure share_class_figi column exists
            if 'share_class_figi' not in self.master_tb.columns:
                self.master_tb = self.master_tb.with_columns(
                    pl.lit(None).cast(pl.Utf8).alias('share_class_figi')
                )

        # 13. Save updated prev_universe
        self._save_prev_universe(current_nasdaq, today_str)

        # 14. Log results (caller is responsible for save_local)
        changes_made = stats['extended'] + stats['rebranded'] + stats['added'] + stats['delisted']
        if changes_made > 0:
            self.logger.info(
                f"SecurityMaster updated: "
                f"{stats['extended']} extended, {stats['rebranded']} rebranded, "
                f"{stats['added']} new IPOs, {stats['delisted']} delisted, "
                f"{stats['unchanged']} unchanged"
            )

        return stats

    def _bootstrap_extend(
        self,
        current_nasdaq: Set[str],
        today: dt.date,
        sec_lookup: Dict[str, Dict[str, Optional[str]]],
    ) -> Dict[str, int]:
        """
        Bootstrap helper: extend end_dates, enrich from SEC, and fetch OpenFIGI
        for active tickers. Used when no prev_universe exists.

        Only extends end_date for "current" rows (end_date == CRSP_LATEST_DATE).
        Historical rows with earlier end_dates are preserved as-is.
        """
        stats = {'extended': 0, 'rebranded': 0, 'added': 0, 'delisted': 0, 'unchanged': 0}
        crsp_end = dt.datetime.strptime(self.CRSP_LATEST_DATE, '%Y-%m-%d').date()

        current_normalized = {_normalize_ticker(t): t for t in current_nasdaq}

        # Fetch OpenFIGI for all active tickers
        self.logger.info(f"Bootstrap: fetching OpenFIGI for {len(current_nasdaq)} active tickers...")
        figi_mapping = self._fetch_openfigi_mapping(list(current_nasdaq))
        figi_by_norm = {_normalize_ticker(t): figi for t, figi in figi_mapping.items()}

        updated_rows = []
        for row in self.master_tb.iter_rows(named=True):
            row_dict = dict(row)
            row_dict.pop('cusip', None)
            symbol_norm = _normalize_ticker(row['symbol'])

            # Convert CRSP symbol to Nasdaq format (BRKB -> BRK.B)
            if symbol_norm in current_normalized:
                row_dict['symbol'] = current_normalized[symbol_norm]

                # Only extend "current" rows (end_date == CRSP latest date)
                if row_dict.get('end_date') == crsp_end:
                    row_dict['end_date'] = today
                    stats['extended'] += 1
                else:
                    stats['unchanged'] += 1

                self._enrich_row_from_sec(row_dict, sec_lookup.get(symbol_norm))

                # Store FIGI (on all rows for this symbol)
                if not row_dict.get('share_class_figi'):
                    row_dict['share_class_figi'] = figi_by_norm.get(symbol_norm)
            else:
                stats['unchanged'] += 1

            updated_rows.append(row_dict)

        # Identify Nasdaq tickers not matched to any "current" row (new IPOs / symbol reuses)
        extended_norms = set()
        for row in self.master_tb.iter_rows(named=True):
            if row.get('end_date') == crsp_end:
                extended_norms.add(_normalize_ticker(row['symbol']))

        unmatched = set(current_normalized.keys()) - extended_norms
        if unmatched:
            self.logger.info(
                f"Bootstrap: {len(unmatched)} Nasdaq tickers not in CRSP data, adding as new IPOs"
            )

            # Fetch yfinance for GICS classification
            yf_tickers = [current_normalized[n] for n in unmatched]
            yf_metadata = self._fetch_yfinance_metadata(yf_tickers)

            max_sid: int = self.master_tb['security_id'].max() or 1000  # type: ignore[assignment]

            for ticker_norm in unmatched:
                ticker = current_normalized[ticker_norm]
                sec_info = sec_lookup.get(ticker_norm, {})

                # Map yfinance → GICS
                gics = {'sector': None, 'industry': None, 'subindustry': None}
                if ticker_norm in yf_metadata:
                    gics = self._map_to_gics(
                        yf_metadata[ticker_norm].get('sector', ''),
                        yf_metadata[ticker_norm].get('industry', ''),
                    )

                max_sid += 1
                new_row = {
                    'security_id': max_sid,
                    'permno': None,
                    'symbol': ticker,
                    'company': sec_info.get('company', ''),
                    'cik': sec_info.get('cik'),
                    'share_class_figi': figi_by_norm.get(ticker_norm),
                    'start_date': today,
                    'end_date': today,
                    'exchange': sec_info.get('exchange'),
                    'sector': gics.get('sector'),
                    'industry': gics.get('industry'),
                    'subindustry': gics.get('subindustry'),
                }
                updated_rows.append(new_row)
                stats['added'] += 1

        if updated_rows:
            self.master_tb = pl.DataFrame(updated_rows).cast({
                'security_id': pl.Int64,
                'start_date': pl.Date,
                'end_date': pl.Date
            })

        figi_count = sum(1 for v in figi_by_norm.values() if v is not None)
        self.logger.info(
            f"Bootstrap: {stats['extended']} extended, {stats['added']} new IPOs added, "
            f"{figi_count}/{len(current_nasdaq)} FIGIs populated"
        )

        return stats

    def close(self):
        """No-op (kept for interface compat)."""
        pass
