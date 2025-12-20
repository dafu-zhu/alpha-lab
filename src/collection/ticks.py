import os
import requests
import time
import zoneinfo
import datetime as dt
from typing import Tuple, List
from dataclasses import asdict
from dotenv import load_dotenv
import polars as pl
from pathlib import Path

from collection.models import TickField, TickDataPoint

load_dotenv()

class Ticks:
    def __init__(
            self, 
            symbol: str, 
            key: str, 
            secret: str
        ) -> None:
        self.symbol = symbol
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret
        }

    @staticmethod
    def get_trade_day_range(trade_day: str | dt.date) -> Tuple[str]:
        """
        Get the full UTC time range for a trading day (9:30 AM - 4:00 PM ET)

        :param trade_day: Specify trade day
        :param type:
            str: format "YYYY-MM-DD"
            datetime.date:
        :return: start, end
        """
        # Parse str day into dt.date object
        if isinstance(trade_day, str):
            trade_day = dt.datetime.strptime(trade_day, "%Y-%m-%d").date()

        # Declare ET time zone
        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_et = dt.datetime.combine(trade_day, dt.time(9, 30), tzinfo=eastern)
        end_et = dt.datetime.combine(trade_day, dt.time(16, 0), tzinfo=eastern)

        # Transform into UTC time
        start_str = start_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = end_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

        return start_str, end_str

    @staticmethod
    def get_year_range(year: str | int) -> Tuple[str]:
        """
        Get the UTC time range for a full year (Jan 1 - Dec 31)

        :param year: Year as string or integer
        :return: start, end in UTC format
        """
        if isinstance(year, str):
            year = int(year)

        # Create start and end dates in UTC
        start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(year, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)

        start_str = start.isoformat().replace("+00:00", "Z")
        end_str = end.isoformat().replace("+00:00", "Z")

        return start_str, end_str

    def get_ticks(
        self,
        start: str,
        end: str,
        timeframe: str = "1Min"
    ) -> List[dict]:
        """
        Get tick data from Alpaca API for specified timeframe and date range.

        :param start: Start datetime in UTC (format: "2025-01-03T14:30:00Z")
        :param end: End datetime in UTC (format: "2025-01-03T21:00:00Z")
        :param timeframe: Alpaca timeframe (e.g., "1Min", "1Day", "1Hour")
        :return: List of dictionaries with OHLCV data
        """
        symbol = str(self.symbol).upper()
        url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe={timeframe}&start={start}&end={end}&limit=10000&adjustment=raw&feed=sip&sort=asc"

        response = requests.get(url, headers=self.headers)

        # Rate limit: 200/m or 10/s
        time.sleep(0.1)

        bars = response.json()['bars'][self.symbol]
        return bars

    def get_minute(self, trade_day: str | dt.date) -> List[dict]:
        """
        Get minute-level OHLCV data for a specific trading day.

        :param trade_day: Trading day (format: "YYYY-MM-DD" or date object)
        :return: List of dictionaries with OHLCV data
        """
        start, end = self.get_trade_day_range(trade_day)
        return self.get_ticks(start, end, timeframe="1Min")

    def get_daily(self, year: str | int) -> List[dict]:
        """
        Get daily OHLCV data for a full year.

        :param year: Year as string or integer (e.g., "2024" or 2024)
        :return: List of dictionaries with OHLCV data
        """
        start, end = self.get_year_range(year)
        return self.get_ticks(start, end, timeframe="1Day")

    def parse_ticks(self, ticks: List[dict]) -> List[TickDataPoint]:
        """
        Parse raw tick data from Alpaca API into TickDataPoint objects.
        Converts timestamps from UTC to Eastern Time (timezone-naive).

        :param ticks: List of dictionaries with OHLCV data
        :return: List of TickDataPoint objects
        """
        eastern = zoneinfo.ZoneInfo("America/New_York")
        datapoints = []

        for tick in ticks:
            # Parse UTC timestamp and convert to ET
            timestamp_utc = dt.datetime.fromisoformat(
                tick[TickField.TIMESTAMP.value].replace('Z', '+00:00')
            )
            # Convert to Eastern Time and remove timezone info for storage
            timestamp_et = timestamp_utc.astimezone(eastern).replace(tzinfo=None)

            dp = TickDataPoint(
                timestamp=timestamp_et.isoformat(),
                open=tick[TickField.OPEN.value],
                high=tick[TickField.HIGH.value],
                low=tick[TickField.LOW.value],
                close=tick[TickField.CLOSE.value],
                volume=tick[TickField.VOLUME.value],
                num_trades=tick[TickField.NUM_TRADES.value],
                vwap=tick[TickField.VWAP.value]
            )
            datapoints.append(dp)
        return datapoints

    def store_ticks(self, dps: List[TickDataPoint], storage_type: str = "minute"):
        """
        Store tick datapoints to Parquet file.

        Storage formats:
        - minute: data/ticks/minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet (timestamp as Datetime)
        - daily: data/ticks/daily/{symbol}/{YYYY}/ticks.parquet (timestamp as Date)

        :param dps: List of TickDataPoint objects
        :param storage_type: "minute" or "daily"
        """
        if not dps:
            return

        # Extract date from first timestamp (format: "2025-01-03T09:30:00")
        first_timestamp = dps[0].timestamp
        date_obj = dt.datetime.fromisoformat(first_timestamp)

        # Build directory path based on storage type
        year = date_obj.strftime('%Y')

        if storage_type == "minute":
            month = date_obj.strftime('%m')
            day = date_obj.strftime('%d')
            dir_path = Path(f"data/ticks/minute/{self.symbol}/{year}/{month}/{day}")
        elif storage_type == "daily":
            dir_path = Path(f"data/ticks/daily/{self.symbol}/{year}")
        else:
            raise ValueError(f"Invalid storage_type: {storage_type}. Must be 'minute' or 'daily'")
        dir_path.mkdir(parents=True, exist_ok=True)

        # Convert datapoints to dictionaries
        ticks_data = [asdict(dp) for dp in dps]

        # Create DataFrame with appropriate schema based on storage type
        if storage_type == "minute":
            # For minute data: keep full datetime
            df = pl.DataFrame(ticks_data).with_columns([
                pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S'),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('num_trades').cast(pl.Int64),
                pl.col('vwap').cast(pl.Float64)
            ])
        else:  # daily
            # For daily data: extract date only
            df = pl.DataFrame(ticks_data).with_columns([
                pl.col('timestamp').str.to_date(format='%Y-%m-%dT%H:%M:%S'),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('num_trades').cast(pl.Int64),
                pl.col('vwap').cast(pl.Float64)
            ])

        # Write to Parquet file
        file_path = dir_path / 'ticks.parquet'
        df.write_parquet(file_path, compression='zstd')

        print(f"Stored {len(dps)} {storage_type} ticks to {file_path}")




if __name__ == "__main__":
    SYMBOL = "AAPL"
    ALPACA_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")

    ticks = Ticks(SYMBOL, ALPACA_KEY, ALPACA_SECRET)

    # Example 1: Fetch and store minute data for a specific day
    print("=" * 50)
    print("Fetching minute data...")
    trade_day = "2025-01-03"
    minutes = ticks.get_minute(trade_day)
    print(f"Fetched {len(minutes)} minute bars")

    datapoints_minute = ticks.parse_ticks(minutes)
    print(f"Parsed {len(datapoints_minute)} datapoints")

    ticks.store_ticks(datapoints_minute, storage_type="minute")

    # Example 2: Fetch and store daily data for a full year
    print("\n" + "=" * 50)
    print("Fetching daily data...")
    year = "2024"
    daily_bars = ticks.get_daily(year)
    print(f"Fetched {len(daily_bars)} daily bars")

    datapoints_daily = ticks.parse_ticks(daily_bars)
    print(f"Parsed {len(datapoints_daily)} datapoints")

    ticks.store_ticks(datapoints_daily, storage_type="daily")