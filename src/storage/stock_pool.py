"""
SEC EDGAR Stock Fetcher
===============================

Fetches all actively traded US common stocks from SEC EDGAR API.
"""
import os
import requests
import pandas as pd
from datetime import datetime
import time

def fetch_sec_stocks():
    """
    Fetch all US stocks from SEC EDGAR
    Returns a pandas DataFrame
    """
    # Setup headers (required by SEC)
    headers = {
        'User-Agent': 'name@example.com',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.sec.gov'
    }
    
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    
    try:
        # SEC rate limit: 10 requests/second
        time.sleep(0.11)
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            companies = []

            if 'data' in data:
                # Format: {"fields": [...], "data": [[...]]}
                for item in data['data']:
                    companies.append({
                        'CIK': str(item[0]).zfill(10),
                        'Name': item[1],
                        'Ticker': item[2],
                        'Exchange': item[3]
                    })
            
            df = pd.DataFrame(companies)
            
        else:
            url_fallback = "https://www.sec.gov/files/company_tickers.json"
            time.sleep(0.11)
            
            response = requests.get(url_fallback, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                companies = []
                for key, value in data.items():
                    companies.append({
                        'CIK': str(value['cik_str']).zfill(10),
                        'Ticker': value['ticker'],
                        'Name': value['title']
                    })
                
                df = pd.DataFrame(companies)
            else:
                return None
                
    except Exception as error:
        print(f"Error: {error}")
        return None
    
    initial_count = len(df)
    
    # Must have ticker
    df = df[df['Ticker'].notna()]
    df = df[df['Ticker'] != '']

    # Remove preferred stocks, warrants, units, rights
    # Pattern: ends with .PR, -W, -WT, -U, -R
    df = df[~df['Ticker'].str.match(r'.*[-\.](PR|W[A-Z]*|U[A-Z]*|R[A-Z]*)$', na=False)]

    # Remove special characters (^, ~, .)
    df = df[~df['Ticker'].str.contains(r'[\^~\.]', na=False)]

    # Filter for major US exchanges (if Exchange column exists)
    if 'Exchange' in df.columns and df['Exchange'].notna().any():
        major_exchanges = ['Nasdaq', 'NYSE', 'NYSEAmerican', 'AMEX', 'NYSEArca']
        df = df[df['Exchange'].isin(major_exchanges)]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Ticker'], keep='first')
    
    # Sort by ticker
    df = df.sort_values('Ticker').reset_index(drop=True)
    
    print(f"Final count: {len(df)} common stocks, removed {initial_count-len(df)}")

    # Store result as csv
    dir_path = os.path.join('data', 'symbols')
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, 'stock_exchange.csv')
    df.to_csv(file_path)
    
    return df


if __name__ == "__main__":
    result = fetch_sec_stocks()
    print(result.tail())
