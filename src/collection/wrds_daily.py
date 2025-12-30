"""
WRDS CRSP Daily Data Collection
Fetches OHLCV data from CRSP and converts to Polars DataFrame format matching ticks.py
"""
import wrds
import pandas as pd
import polars as pl
from dotenv import load_dotenv
import os
from typing import Optional
import datetime as dt
from pathlib import Path
from master.security_master import SecurityMaster

load_dotenv()


class WRDSDailyTicks:
    def __init__(self, conn: wrds.Connection=None):
        """Initialize WRDS connection and load master calendar"""
        if conn is None:
            username = os.getenv('WRDS_USERNAME')
            password = os.getenv('WRDS_PASSWORD')

            self.conn = wrds.Connection(
                wrds_username=username,
                wrds_password=password
            )
        else:
            self.conn = conn

        # Setup calendar directory
        self.calendar_dir = Path("data/calendar")
        self.calendar_path = self.calendar_dir / "master.parquet"

        self.security_master = SecurityMaster(db=self.conn)

    def get_daily(self, symbol: str, day: str, adjusted: bool=True):
        """
        Smart auto-resolve: Handle ticker name changes

        Example: get_daily('META', '2021-12-31')
        - 'META' not active on 2021-12-31 (exact match fails)
        - Find security that ever used 'META' → security_id=1234
        - Check if security_id=1234 was active on 2021-12-31 (as 'FB') → YES
        - Fetch data using permno for that security
        """
        # Try exact match
        sid = self.security_master.get_security_id(symbol, day)

        if not sid:
            master_tb = self.security_master.master_table()
            date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()

            # Find all securities that ever used this symbol
            candidates = master_tb.filter(
                pl.col('symbol').eq(symbol)
            ).select('security_id').unique()

            if candidates.is_empty():
                raise ValueError(f"Symbol '{symbol}' never existed in security master")

            # For each candidate, check if it was active on target date (under ANY symbol)
            active_securities = []
            for candidate_sid in candidates['security_id']:
                was_active = master_tb.filter(
                    pl.col('security_id').eq(candidate_sid),
                    pl.col('start_date').le(date_check),
                    pl.col('end_date').ge(date_check)
                )
                if not was_active.is_empty():
                    # Find when this security used the queried symbol
                    symbol_usage = master_tb.filter(
                        pl.col('security_id').eq(candidate_sid),
                        pl.col('symbol').eq(symbol)
                    ).select(['start_date', 'end_date']).head(1)

                    active_securities.append({
                        'sid': candidate_sid,
                        'symbol_start': symbol_usage['start_date'][0],
                        'symbol_end': symbol_usage['end_date'][0]
                    })

            # Resolve ambiguity
            if len(active_securities) == 0:
                raise ValueError(
                    f"Symbol '{symbol}' exists but the associated security was not active on {day}"
                )
            elif len(active_securities) == 1:
                sid = active_securities[0]['sid']
            else:
                # Multiple securities used this symbol and were active on target date
                # Pick the one that used this symbol closest to the query date
                def distance_to_date(sec):
                    """Calculate temporal distance from query date to when symbol was used"""
                    if date_check < sec['symbol_start']:
                        return (sec['symbol_start'] - date_check).days
                    elif date_check > sec['symbol_end']:
                        return (date_check - sec['symbol_end']).days
                    else:
                        return 0

                # Pick security with minimum distance
                best_match = min(active_securities, key=distance_to_date)
                sid = best_match['sid']

        permno = self.security_master.sid_to_permno(sid)

        # Write query to access CRSP database
        # CRSP DSF fields:
        # - date: trading date
        # - openprc: opening price
        # - askhi: high ask price (proxy for high)
        # - bidlo: low bid price (proxy for low)
        # - prc: closing price (negative if bid/ask average, take absolute)
        # - vol: volume (in hundreds of shares, multiply by 100)
        # - cfacpr: cumulative price adjustment factor (for splits/dividends)
        # - cfacshr: cumulative share adjustment factor
        if adjusted:
            # Fetch with adjustment factors
            query = f"""
                SELECT
                    date,
                    openprc / cfacpr as open,
                    askhi / cfacpr as high,
                    bidlo / cfacpr as low,
                    abs(prc) / cfacpr as close,
                    vol * cfacshr as volume
                FROM crsp.dsf
                WHERE permno = {permno}
                    AND date = '{day}'
                    AND prc IS NOT NULL
            """
        else:
            # Fetch raw unadjusted prices
            query = f"""
                SELECT
                    date,
                    openprc as open,
                    askhi as high,
                    bidlo as low,
                    abs(prc) as close,
                    vol as volume
                FROM crsp.dsf
                WHERE permno = {permno}
                    AND date = '{day}'
                    AND prc IS NOT NULL
            """
        
        df_pandas = self.conn.raw_sql(query, date_cols=['date'])

        # Handle empty data case
        if df_pandas.empty:
            return {}
        
        row = df_pandas.iloc[0]

        result = {
            'timestamp': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }      

        return result

    def get_daily_range(self, symbol: str, start_date: str, end_date: str, adjusted: bool=True):
        pass

    def close(self):
        """Close WRDS connection"""
        self.conn.close()


if __name__ == "__main__":
    # Example usage
    wrds_ticks = WRDSDailyTicks()

    # Close connection
    wrds_ticks.close()