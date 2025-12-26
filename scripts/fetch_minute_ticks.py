"""
Script to fetch minute-level ticks data from AWS S3.

Usage:
    python scripts/fetch_minute_ticks.py
"""

import sys
from pathlib import Path
import io
import datetime as dt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from storage.s3_client import S3Client
import polars as pl


def fetch_minute_ticks(symbol: str, trade_day: str) -> pl.DataFrame:
    """
    Fetch minute-level ticks data for a symbol from S3.

    :param symbol: Stock symbol in SEC format (e.g., 'AAPL', 'BRK-B')
    :param trade_day: Trading day in format 'YYYY-MM-DD'
    :return: Polars DataFrame with minute ticks data
    """
    print(f"\n{'=' * 80}")
    print(f"Fetching minute ticks for {symbol} - {trade_day}")
    print(f"{'=' * 80}\n")

    try:
        # Initialize S3 client
        s3_client = S3Client().client
        bucket_name = 'us-equity-datalake'

        # Parse date for S3 key
        date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')

        # Construct S3 key
        s3_key = f"data/raw/ticks/minute/{symbol}/{year}/{month}/{day}/ticks.parquet"
        print(f"S3 Key: {s3_key}")

        # Download file from S3
        print("Downloading from S3...")
        buffer = io.BytesIO()
        s3_client.download_fileobj(
            Bucket=bucket_name,
            Key=s3_key,
            Fileobj=buffer
        )

        # Read parquet from buffer
        buffer.seek(0)
        df = pl.read_parquet(buffer)

        print(f"\n✓ Data successfully fetched!")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Expected: 390 minutes (9:30 AM - 4:00 PM ET)")

        # Show column info
        print(f"\nColumns: {df.columns}")

        # Show basic statistics
        if df.shape[0] > 0:
            print(f"\nPrice Statistics:")
            print(f"  High: ${df['high'].max():.2f}")
            print(f"  Low: ${df['low'].min():.2f}")
            print(f"  Open (first): ${df['open'][0]:.2f}")
            print(f"  Close (last): ${df['close'][-1]:.2f}")
            print(f"  Total Volume: {df['volume'].sum():,.0f}")
            print(f"  Avg Minute Volume: {df['volume'].mean():,.0f}")

            # Time range
            print(f"\nTime Range:")
            print(f"  First timestamp: {df['timestamp'][0]}")
            print(f"  Last timestamp: {df['timestamp'][-1]}")

        print(f"\nDataFrame Preview (first 10 rows):")
        print(df.head(10))

        return df

    except s3_client.exceptions.NoSuchKey:
        print(f"\n❌ Data not found for {symbol} on {trade_day}")
        print(f"   S3 Key: {s3_key}")
        return None
    except Exception as e:
        print(f"\n❌ Error fetching data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to fetch minute ticks for multiple symbols and days."""

    # Define symbols and dates to fetch
    symbols = ['AAPL', 'MSFT']
    trade_days = ['2024-12-23', '2024-12-24']  # Recent trading days

    print(f"\n{'#' * 80}")
    print(f"# Fetching minute ticks data")
    print(f"# Symbols: {', '.join(symbols)}")
    print(f"# Days: {', '.join(trade_days)}")
    print(f"# Source: AWS S3 (us-equity-datalake)")
    print(f"{'#' * 80}")

    results = {}

    # Fetch data for each symbol-day combination
    for symbol in symbols:
        for trade_day in trade_days:
            key = f"{symbol}_{trade_day}"
            df = fetch_minute_ticks(symbol, trade_day)
            results[key] = df

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    success_count = sum(1 for df in results.values() if df is not None)
    total_count = len(symbols) * len(trade_days)
    print(f"\nSuccessfully fetched: {success_count}/{total_count}")

    for key, df in results.items():
        if df is not None:
            print(f"  ✓ {key}: {df.shape[0]} minute bars")
        else:
            print(f"  ✗ {key}: Failed")

    print()


if __name__ == "__main__":
    main()
