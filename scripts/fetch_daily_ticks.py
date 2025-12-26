"""
Script to fetch daily ticks data from AWS S3.

Usage:
    python scripts/fetch_daily_ticks.py
"""

import sys
from pathlib import Path
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from storage.s3_client import S3Client
import polars as pl


def fetch_daily_ticks(symbol: str, year: int = 2024) -> pl.DataFrame:
    """
    Fetch daily ticks data for a symbol from S3.

    :param symbol: Stock symbol in SEC format (e.g., 'AAPL', 'BRK-B')
    :param year: Year to fetch data for (default: 2024)
    :return: Polars DataFrame with daily ticks data
    """
    print(f"\n{'=' * 80}")
    print(f"Fetching daily ticks for {symbol} - Year {year}")
    print(f"{'=' * 80}\n")

    try:
        # Initialize S3 client
        s3_client = S3Client().client
        bucket_name = 'us-equity-datalake'

        # Construct S3 key
        s3_key = f"data/raw/ticks/daily/{symbol}/{year}/ticks.parquet"
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

        # Show data quality metrics
        non_null_rows = df.filter(pl.col('open').is_not_null()).shape[0]
        print(f"Trading days with data: {non_null_rows}/{df.shape[0]}")

        # Show column info
        print(f"\nColumns: {df.columns}")

        # Show basic statistics
        if non_null_rows > 0:
            print(f"\nPrice Statistics:")
            print(f"  High: ${df['high'].max():.2f}")
            print(f"  Low: ${df['low'].min():.2f}")
            print(f"  Close (latest): ${df.filter(pl.col('close').is_not_null())['close'][-1]:.2f}")
            print(f"  Avg Volume: {df['volume'].mean():,.0f}")

        print(f"\nDataFrame Preview:")
        print(df)

        return df

    except s3_client.exceptions.NoSuchKey:
        print(f"\n❌ Data not found for {symbol} in year {year}")
        print(f"   S3 Key: {s3_key}")
        return None
    except Exception as e:
        print(f"\n❌ Error fetching data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to fetch daily ticks for multiple symbols."""

    # Define symbols to fetch (use SEC format with '-')
    symbols = ['AAPL', 'MSFT', 'TSLA']
    year = 2024

    print(f"\n{'#' * 80}")
    print(f"# Fetching daily ticks data for {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"# Year: {year}")
    print(f"# Source: AWS S3 (us-equity-datalake)")
    print(f"{'#' * 80}")

    results = {}

    # Fetch data for each symbol
    for symbol in symbols:
        df = fetch_daily_ticks(symbol, year)
        results[symbol] = df

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    success_count = sum(1 for df in results.values() if df is not None)
    print(f"\nSuccessfully fetched: {success_count}/{len(symbols)}")

    for symbol, df in results.items():
        if df is not None:
            non_null_rows = df.filter(pl.col('open').is_not_null()).shape[0]
            print(f"  ✓ {symbol}: {non_null_rows} trading days with data")
        else:
            print(f"  ✗ {symbol}: Failed")

    print()


if __name__ == "__main__":
    main()
