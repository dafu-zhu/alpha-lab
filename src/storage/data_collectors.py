"""
Data collection functionality for market data.

This module handles fetching market data from various sources:
- Daily ticks from CRSP or Alpaca
- Minute ticks from Alpaca API
"""

import datetime as dt
import time
import logging
from typing import List, Dict
import requests
import polars as pl

from collection.models import TickField


class DataCollectors:
    """
    Handles data collection from various market data sources.
    """

    def __init__(
        self,
        crsp_ticks,
        alpaca_ticks,
        alpaca_headers: dict,
        logger: logging.Logger
    ):
        """
        Initialize data collectors.

        :param crsp_ticks: CRSPDailyTicks instance
        :param alpaca_ticks: Alpaca Ticks instance
        :param alpaca_headers: Headers for Alpaca API requests
        :param logger: Logger instance
        """
        self.crsp_ticks = crsp_ticks
        self.alpaca_ticks = alpaca_ticks
        self.alpaca_headers = alpaca_headers
        self.logger = logger

    def collect_daily_ticks_year(self, sym: str, year: int) -> pl.DataFrame:
        """
        Fetch daily ticks for entire year from appropriate source and return as Polars DataFrame.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        all_months_data = []

        if year < 2025:
            # Use CRSP for years < 2025 (avoids survivorship bias)
            crsp_symbol = sym.replace('.', '').replace('-', '')

            # Fetch all 12 months
            for month in range(1, 13):
                try:
                    json_list = self.crsp_ticks.collect_daily_ticks(
                        symbol=crsp_symbol,
                        year=year,
                        month=month,
                        adjusted=True,
                        auto_resolve=True
                    )
                    all_months_data.extend(json_list)
                except ValueError as e:
                    if "not active on" in str(e):
                        # Symbol not active in this month, skip
                        continue
                    else:
                        raise
        else:
            # Use Alpaca for years >= 2025
            try:
                df = self.alpaca_ticks.get_daily_year(
                    symbol=sym,
                    year=year,
                    adjusted=True
                )

                # Ensure correct column types and round decimals
                if len(df) > 0:
                    df = df.with_columns([
                        pl.col('timestamp').cast(pl.Utf8),
                        pl.col('open').round(4),
                        pl.col('high').round(4),
                        pl.col('low').round(4),
                        pl.col('close').round(4),
                        pl.col('volume').cast(pl.Int64)
                    ]).sort('timestamp')

                return df
            except Exception as e:
                # Return empty DataFrame if fetch fails
                return pl.DataFrame()

        # Convert to Polars DataFrame (for CRSP path)
        if not all_months_data:
            return pl.DataFrame()

        df = pl.DataFrame(all_months_data)

        # Ensure correct column types and round decimals
        if len(df) > 0:
            df = df.with_columns([
                pl.col('timestamp').cast(pl.Utf8),
                pl.col('open').round(4),
                pl.col('high').round(4),
                pl.col('low').round(4),
                pl.col('close').round(4),
                pl.col('volume').cast(pl.Int64)
            ]).sort('timestamp')

        return df

    def fetch_single_symbol_minute(
        self,
        symbol: str,
        year: int,
        month: int,
        sleep_time: float = 0.1
    ) -> List[dict]:
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

        start_str = dt.datetime.combine(
            start_date, dt.time(0, 0), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(
            end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")

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
        session.headers.update(self.alpaca_headers)

        try:
            # Initial request
            response = session.get(base_url, params=params)
            time.sleep(sleep_time)  # Rate limiting

            if response.status_code != 200:
                self.logger.error(
                    f"Single fetch error for {symbol}: {response.status_code}, {response.text}"
                )
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
                    self.logger.error(
                        f"Pagination error on page {page_count} for {symbol}: {response.status_code}"
                    )
                    break

                data = response.json()
                symbol_bars = data.get("bars", {}).get(symbol, [])
                bars.extend(symbol_bars)

                page_count += 1

            self.logger.info(f"Fetched {len(bars)} bars for {symbol} ({page_count} pages)")

        except Exception as e:
            self.logger.error(f"Exception during single fetch for {symbol}: {e}")

        return bars

    def fetch_minute_bulk(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> dict:
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

        start_str = dt.datetime.combine(
            start_date, dt.time(0, 0), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(
            end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")

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
        session.headers.update(self.alpaca_headers)

        for retry in range(max_retries):
            try:
                # Initial request (using session for connection reuse)
                response = session.get(base_url, params=params)
                time.sleep(sleep_time)  # Rate limiting

                if response.status_code == 429:
                    # Rate limit error - use exponential backoff with longer waits
                    wait_time = min(60, (2 ** retry) * 5)  # 5s, 10s, 20s (capped at 60s)
                    self.logger.warning(
                        f"Rate limit hit for bulk fetch (retry {retry + 1}/{max_retries}), "
                        f"waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                    continue
                elif response.status_code != 200:
                    self.logger.error(
                        f"Bulk fetch error for {symbols}: {response.status_code}, {response.text}"
                    )
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
                        self.logger.error(
                            f"Rate limit hit during pagination on page {page_count}, "
                            f"waiting {wait_time}s"
                        )
                        time.sleep(wait_time)
                        response = session.get(base_url, params=params)
                        time.sleep(sleep_time)

                    if response.status_code != 200:
                        self.logger.error(
                            f"Pagination error on page {page_count} for {symbols}: "
                            f"{response.status_code}"
                        )
                        break

                    data = response.json()
                    bars = data.get("bars", {})

                    for sym in symbols:
                        if sym in bars:
                            all_bars[sym].extend(bars[sym])

                    page_count += 1

                self.logger.info(
                    f"Fetched {sum(len(v) for v in all_bars.values())} total bars for "
                    f"{len(symbols)} symbols ({page_count} pages)"
                )
                bulk_success = True
                break

            except Exception as e:
                self.logger.error(
                    f"Exception during bulk fetch for {symbols} "
                    f"(retry {retry + 1}/{max_retries}): {e}"
                )
                if retry < max_retries - 1:
                    wait_time = (2 ** retry) * 2
                    self.logger.error(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)

        # Close the session after retry loop
        session.close()

        # If bulk fetch failed after retries, fetch symbols one by one
        if not bulk_success:
            self.logger.info(
                f"Bulk fetch failed after {max_retries} retries, "
                f"fetching symbols individually"
            )
            failed_symbols = []

            for sym in symbols:
                try:
                    bars = self.fetch_single_symbol_minute(sym, year, month, sleep_time=sleep_time)
                    all_bars[sym] = bars
                    if not bars:
                        self.logger.info(f"No data returned for {sym}")
                except Exception as e:
                    self.logger.error(f"Failed to fetch {sym} individually: {e}")
                    failed_symbols.append(sym)

            if failed_symbols:
                self.logger.warning(
                    f"Failed to fetch {len(failed_symbols)} symbols even after "
                    f"individual retry: {failed_symbols}"
                )

        return all_bars
