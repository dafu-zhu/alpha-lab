import wrds
import pandas as pd
import polars as pl
from dotenv import load_dotenv
import os
import time
import requests
import datetime as dt

load_dotenv()

class SecurityMaster:
    def __init__(self):
        username = os.getenv('WRDS_USERNAME')
        password = os.getenv('WRDS_PASSWORD')
        self.db = wrds.Connection(
            wrds_username=username,
            wrds_password=password
        )

        self.cik_cusip = None
        self.lookup = None
    
    def cik_cusip_mapping(self):
        """
        All historical mappings, updated until Dec 31, 2024
        """
        query = """
        SELECT DISTINCT
            a.kypermno, 
            a.ticker, 
            a.tsymbol,
            a.comnam, 
            a.ncusip, 
            b.cik,
            a.namedt, 
            a.nameenddt
        FROM 
            crsp.s6z_nam AS a
        LEFT JOIN
            wrdssec_common.wciklink_cusip AS b
            ON SUBSTR(a.ncusip, 1, 8) = SUBSTR(b.cusip, 1, 8)
            AND (b.cik IS NULL OR a.namedt <= b.cikdate2)
            AND (b.cik IS NULL OR a.nameenddt >= b.cikdate1)
        WHERE
            a.shrcd IN (10, 11)
        ORDER BY 
            a.kypermno, a.namedt
        """

        # Execute and load into a DataFrame
        map_df = self.db.raw_sql(query)
        map_df['namedt'] = pd.to_datetime(map_df['namedt'])
        map_df['nameenddt'] = pd.to_datetime(map_df['nameenddt'])

        # Forward-fill CIK for records with NULL CIK (due to stale CIK mapping data)
        map_df = map_df.sort_values(['kypermno', 'ncusip', 'namedt'])
        map_df['cik'] = map_df.groupby(['kypermno', 'ncusip'])['cik'].ffill()

        result = (
            map_df.groupby(
                ['kypermno', 'cik', 'ticker', 'tsymbol', 'comnam', 'ncusip'],
                dropna=False
            ).agg({
                'namedt': 'min',
                'nameenddt': 'max'
            })
            .reset_index()
            .sort_values(['kypermno', 'namedt'])
            .dropna(subset=['tsymbol'])
        )

        pl_map = pl.DataFrame(result).with_columns(
            pl.col('kypermno').cast(pl.Int32).alias('permno'),
            pl.col('tsymbol').alias('symbol'),
            pl.col('comnam').alias('company'),
            pl.col('ncusip').alias('cusip'),
            pl.col('namedt').cast(pl.Date).alias('start_date'),
            pl.col('nameenddt').cast(pl.Date).alias('end_date')
        ).select(['permno', 'symbol', 'company', 'cik', 'cusip', 'start_date', 'end_date'])

        return pl_map
    
    def security_map(self):
        """
        Maps security_id with unique permno, replacing permno
        """
        if self.cik_cusip is None:
            self.cik_cusip = self.cik_cusip_mapping()
        unique_permnos = self.cik_cusip.select('permno').unique(maintain_order=True)
        result = pl.DataFrame({
            'security_id': range(1000, 1000 + len(unique_permnos)),
            'permno': unique_permnos
        })
        return result
    
    def master_table(self):
        """
        Create comprehensive table with security_id as master key, includes historically used symbols, cik, cusip
        """
        if self.cik_cusip is None:
            self.cik_cusip = self.cik_cusip_mapping()

        security_map = self.security_map()
        
        full_history = self.cik_cusip.join(security_map, on='permno', how='left')
        result = full_history[['security_id', 'symbol', 'company', 'cik', 'cusip', 'start_date', 'end_date']]

        return result

    def get_security_id(self, symbol: str, day: str) -> int:
        """
        Finds the Internal ID for a specific Symbol at a specific point in time.
        """
        date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()
        
        if self.lookup is None:
            self.lookup = self.master_table()
            
        match = self.lookup.filter(
            pl.col('symbol').eq(symbol),
            pl.col('start_date').le(date_check),
            pl.col('end_date').ge(date_check)
        )
        
        if match.is_empty():
            return None
        
        result = match.head(1).select('security_id').item()

        return result
    

if __name__ == "__main__":
    sm = SecurityMaster()

    df = sm.cik_cusip_mapping()
    df = df.filter(pl.col('permno').eq(83443))
    print(df)

    # Scenario A: You ask for "AH" in 2012
    print(f"Who was AH in 2012? ID: {sm.get_security_id('AH', '2012-06-01')}")

    # Scenario B: You ask for "RCM" in 2012 (The Trap)
    print(f"Who was RCM in 2012? ID: {sm.get_security_id('RCM', '2012-06-01')}")
    # Output: ID: None (Correct! RCM didn't exist then)

    # Scenario C: You ask for "RCM" in 2020
    print(f"Who was RCM in 2020? ID: {sm.get_security_id('RCM', '2020-01-01')}")

    print(f"Who was MSFT in 2024? ID: {sm.get_security_id('MSFT', '2024-01-01')}")

    print(f"Who was BRKB in 2022? ID: {sm.get_security_id('BRKB', '2022-01-01')}")