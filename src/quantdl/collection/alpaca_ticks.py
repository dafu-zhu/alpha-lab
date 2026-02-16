import datetime as dt
import logging
import os
import time
import zoneinfo
from dataclasses import asdict

import polars as pl
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from quantdl.collection.models import TickDataPoint, TickField
from quantdl.utils.logger import LoggerFactory

load_dotenv()

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
EASTERN = zoneinfo.ZoneInfo("America/New_York")


def _to_utc_iso(datetime_obj: dt.datetime) -> str:
    """Format a UTC datetime as ISO string with 'Z' suffix."""
    return datetime_obj.isoformat().replace("+00:00", "Z")


class Ticks:
    def __init__(self) -> None:
        self.logger = LoggerFactory(
            log_dir='data/logs/ticks',
            level=logging.INFO,
            daily_rotation=True,
            console_output=False
        ).get_logger(name='collection.ticks')

        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET"),
        }

    def recent_daily_ticks(
        self,
        symbols: list[str],
        end_day: str,
        window: int = 90,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch recent daily data for a list of symbols using Alpaca's multi-symbol API.

        Alpaca's limit parameter applies to TOTAL data points across all symbols,
        not per symbol. This method handles pagination with next_page_token.

        :param symbols: List of symbols to fetch
        :param end_day: End date in format 'YYYY-MM-DD'
        :param window: Number of calendar days to look back
        :return: Dictionary {symbol: DataFrame} with price and volume data
        """
        end_dt = dt.datetime.strptime(end_day, '%Y-%m-%d')
        start_dt = end_dt - dt.timedelta(days=window)

        yesterday = dt.datetime.today() - dt.timedelta(days=1)
        if end_dt > yesterday:
            end_dt = yesterday

        start_str = _to_utc_iso(
            dt.datetime.combine(start_dt.date(), dt.time(0, 0), tzinfo=dt.timezone.utc)
        )
        end_str = _to_utc_iso(
            dt.datetime.combine(end_dt.date(), dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
        )

        self.logger.info(
            f"Fetching {len(symbols)} symbols from {start_dt.date()} to {end_day} "
            f"(window: {window} days)"
        )

        symbols_str = ','.join(symbols)
        all_data: dict[str, list[dict]] = {}
        page_token = None
        page_count = 0
        page_bar = tqdm(desc="Fetching pages", unit="page")
        try:
            while True:
                page_count += 1
                page_bar.update(1)
                params = {
                    "symbols": symbols_str,
                    "timeframe": "1Day",
                    "start": start_str,
                    "end": end_str,
                    "limit": 10000,
                    "adjustment": "split",
                    "feed": "sip",
                    "sort": "asc",
                }
                if page_token:
                    params["page_token"] = page_token

                try:
                    response = requests.get(ALPACA_BARS_URL, headers=self.headers, params=params)
                    time.sleep(0.1)  # Rate limiting: 200/min = ~3/sec, use 10/sec to be safe

                    if response.status_code != 200:
                        self.logger.error(f"API Error [{response.status_code}]: {response.text}")
                        break

                    data = response.json()
                    bars = data.get('bars') or {}
                    for symbol, symbol_bars in bars.items():
                        all_data.setdefault(symbol, []).extend(symbol_bars)

                    page_token = data.get('next_page_token')
                    if not page_token:
                        self.logger.info(f"Completed fetching data in {page_count} page(s)")
                        break

                except Exception as e:
                    self.logger.error(f"Request failed on page {page_count}: {e}")
                    break
        finally:
            page_bar.close()

        result_dict: dict[str, pl.DataFrame] = {}
        for symbol, bars in tqdm(all_data.items(), desc="Processing symbols", unit="sym"):
            if not bars:
                continue
            try:
                parsed_ticks = self.parse_ticks(bars)
                ticks_data = [asdict(dp) for dp in parsed_ticks]

                df = (
                    pl.DataFrame(ticks_data)
                    .with_columns(
                        pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S'),
                        pl.col('open').cast(pl.Float64),
                        pl.col('high').cast(pl.Float64),
                        pl.col('low').cast(pl.Float64),
                        pl.col('close').cast(pl.Float64),
                        pl.col('volume').cast(pl.Int64),
                        pl.col('num_trades').cast(pl.Int64),
                        pl.col('vwap').cast(pl.Float64),
                    )
                    .select('timestamp', 'close', 'volume', 'num_trades', 'vwap')
                )
                result_dict[symbol] = df

            except Exception as e:
                self.logger.error(f"Failed to process data for {symbol}: {e}")

        self.logger.info(f"Successfully fetched {len(result_dict)}/{len(symbols)} symbols")
        return result_dict

    def fetch_daily_range_bulk(
        self,
        symbols: list[str],
        start_str: str,
        end_str: str,
        sleep_time: float = 0.2,
        adjusted: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Bulk fetch daily bars for multiple symbols over a date range.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param start_str: Start time in UTC ISO format with 'Z' suffix
        :param end_str: End time in UTC ISO format with 'Z' suffix
        :param sleep_time: Sleep time between paginated requests in seconds
        :param adjusted: If True, apply split adjustments
        :return: Dict mapping symbol -> list of bars
        """
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start_str,
            "end": end_str,
            "limit": 10000,
            "adjustment": "split" if adjusted else "raw",
            "feed": "sip",
            "sort": "asc",
        }

        session = requests.Session()
        session.headers.update(self.headers)
        try:
            return self._fetch_with_pagination(
                session=session,
                base_url=ALPACA_BARS_URL,
                params=params,
                symbols=symbols,
                sleep_time=sleep_time,
            )
        finally:
            session.close()

    def fetch_daily_month_bulk(
        self,
        symbols: list[str],
        year: int,
        month: int,
        sleep_time: float = 0.2,
        adjusted: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Bulk fetch daily bars for multiple symbols for the specified month.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param year: Year to fetch
        :param month: Month to fetch (1-12)
        :param sleep_time: Sleep time between paginated requests in seconds
        :param adjusted: If True, apply split adjustments
        :return: Dict mapping symbol -> list of bars
        """
        start_str, end_str = self._get_month_range(year, month)
        return self.fetch_daily_range_bulk(
            symbols=symbols,
            start_str=start_str,
            end_str=end_str,
            sleep_time=sleep_time,
            adjusted=adjusted,
        )

    def fetch_daily_year_bulk(
        self,
        symbols: list[str],
        year: int,
        sleep_time: float = 0.2,
        adjusted: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Bulk fetch daily bars for multiple symbols for the specified year.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param year: Year to fetch
        :param sleep_time: Sleep time between paginated requests in seconds
        :param adjusted: If True, apply split adjustments
        :return: Dict mapping symbol -> list of bars
        """
        start_str = _to_utc_iso(
            dt.datetime.combine(dt.date(year, 1, 1), dt.time(0, 0), tzinfo=dt.timezone.utc)
        )
        end_str = _to_utc_iso(
            dt.datetime.combine(dt.date(year, 12, 31), dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
        )
        return self.fetch_daily_range_bulk(
            symbols=symbols,
            start_str=start_str,
            end_str=end_str,
            sleep_time=sleep_time,
            adjusted=adjusted,
        )

    def _get_month_range(self, year: int, month: int) -> tuple[str, str]:
        """
        Calculate UTC time range for a given month based on ET market hours.
        Uses 4:00 AM ET start (pre-market) to 8:00 PM ET end (after-hours) to cover
        extended hours if needed, while ensuring correct date boundaries.

        :param year: Year
        :param month: Month (1-12)
        :return: Tuple of (start_str, end_str) in ISO format with 'Z' suffix
        """
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        start_et = dt.datetime.combine(start_date, dt.time(4, 0), tzinfo=EASTERN)
        end_et = dt.datetime.combine(end_date, dt.time(20, 0), tzinfo=EASTERN)

        return (
            _to_utc_iso(start_et.astimezone(dt.timezone.utc)),
            _to_utc_iso(end_et.astimezone(dt.timezone.utc)),
        )

    def _fetch_with_pagination(
        self,
        session: requests.Session,
        base_url: str,
        params: dict,
        symbols: list[str],
        sleep_time: float,
    ) -> dict[str, list[dict]]:
        """
        Fetch data with pagination handling from Alpaca API.

        :param session: Requests session with headers configured
        :param base_url: API endpoint URL
        :param params: Initial request parameters
        :param symbols: List of symbols being fetched
        :param sleep_time: Sleep time between requests
        :return: Dict mapping symbol -> list of bars
        """
        all_bars: dict[str, list[dict]] = {sym: [] for sym in symbols}
        page_count = 0

        while True:
            page_count += 1
            response = session.get(base_url, params=params)
            time.sleep(sleep_time)

            if response.status_code != 200:
                self.logger.error(
                    f"API error (page {page_count}): {response.status_code}, {response.text}"
                )
                break

            data = response.json()
            bars = data.get("bars", {})

            for sym in symbols:
                if sym in bars:
                    all_bars[sym].extend(bars[sym])

            next_token = data.get("next_page_token")
            if not next_token:
                self.logger.info(
                    f"Fetched {sum(len(v) for v in all_bars.values())} total bars "
                    f"for {len(symbols)} symbols ({page_count} pages)"
                )
                break

            params["page_token"] = next_token

        return all_bars

    @staticmethod
    def parse_ticks(ticks: list[dict]) -> list[TickDataPoint]:
        """
        Parse raw tick data from Alpaca API into TickDataPoint objects.
        Converts timestamps from UTC to Eastern Time (timezone-naive).

        :param ticks: List of dictionaries with OHLCV data
        :return: List of TickDataPoint objects
        """
        datapoints = []

        for tick in ticks:
            if not tick:
                raise ValueError(f"tick is empty before parsing, got {tick}")

            timestamp_utc = dt.datetime.fromisoformat(
                tick[TickField.TIMESTAMP.value].replace('Z', '+00:00')
            )
            timestamp_et = timestamp_utc.astimezone(EASTERN).replace(tzinfo=None)

            dp = TickDataPoint(
                timestamp=timestamp_et.isoformat(),
                open=tick[TickField.OPEN.value],
                high=tick[TickField.HIGH.value],
                low=tick[TickField.LOW.value],
                close=tick[TickField.CLOSE.value],
                volume=tick[TickField.VOLUME.value],
                num_trades=tick[TickField.NUM_TRADES.value],
                vwap=tick[TickField.VWAP.value],
            )
            datapoints.append(dp)

        return datapoints
