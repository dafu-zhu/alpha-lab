"""
SEC EDGAR Stock Fetcher
===============================

Fetches all actively traded US common stocks from SEC EDGAR API.
"""
import io
import os
import requests
import pandas as pd
from datetime import datetime
from ftplib import FTP

def fetch_all_stocks() -> pd.DataFrame:
    """
    Connects to ftp.nasdaqtrader.com to fetch the raw ticker list.
    """
    ftp_host = "ftp.nasdaqtrader.com"
    ftp_dir = "SymbolDirectory"
    ftp_file = "nasdaqtraded.txt"

    try:
        print(f"Connecting to FTP: {ftp_host}...")
        
        # Establish FTP Connection
        ftp = FTP(ftp_host)
        ftp.login()
        ftp.cwd(ftp_dir)
        
        # Download file to memory (BytesIO)
        # This is faster than writing to disk and reading back
        print(f"Downloading {ftp_file}...")
        byte_buffer = io.BytesIO()
        
        # Retrieve a file
        ftp.retrbinary(f"RETR {ftp_file}", byte_buffer.write)
        ftp.quit()
        
        # Reset buffer pointer to the beginning so Pandas can read it
        byte_buffer.seek(0)
        
        # Read CSV directly from URL (Separator is '|')
        # Funny enough: without specifying dtype, pandas recognize 'NaN' as null, which is in fact 'Nano Labs Ltd'
        df = pd.read_csv(byte_buffer, sep='|', dtype={'Symbol': str}, keep_default_na=False, na_values=[''])
        
        # Remove the file footer (usually contains file creation timestamp)
        df = df[:-1]
        
        # FILTER: Exclude ETFs
        # The 'ETF' column is 'Y' for ETFs and 'N' for stocks
        if 'ETF' in df.columns:
            df = df[df['ETF'] == 'N']
            
        # FILTER: Exclude Test Issues
        if 'Test Issue' in df.columns:
            df = df[df['Test Issue'] == 'N']

        # Rename columns to match your system
        df = df.rename(columns={'Symbol': 'Ticker', 'Security Name': 'Name'})
        
        # Additional Cleanup (Preferreds, Warrants, Rights)
        df = df[~df['Ticker'].str.contains(r'[-\.](?:W|R|P|U)$', regex=True, na=False)]
        df = df[~df['Ticker'].str.contains(r'[$~\^\.]', regex=True, na=False)]

        # Remove duplicates
        df = df.drop_duplicates(subset=['Ticker'], keep='first')
        df = df.sort_values('Ticker').reset_index(drop=True)
        
        print(f"Final Universe: {len(df)} common stocks")
        
        # Store result
        dir_path = os.path.join('data', 'symbols')
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, 'stock_exchange.csv')
        
        # Save only relevant columns
        output_df = df[['Ticker', 'Name']]
        output_df.to_csv(file_path, index=False)
        
        return output_df

    except Exception as error:
        print(f"Error fetching Nasdaq data: {error}")
        return None
    

if __name__ == "__main__":
    result = fetch_all_stocks()
    print(result.tail())
