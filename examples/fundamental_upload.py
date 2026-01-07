"""
Example: Fundamental Data Collection and Upload

This example demonstrates:
1. Using DataCollectors.collect_fundamental_year() to fetch concept-based fundamental data
2. Using UploadApp to upload fundamental data to S3

The system uses approved_mapping.yaml for concept-based extraction of all 29 financial metrics.
"""

from pathlib import Path
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.data_collectors import DataCollectors
from src.collection.alpaca_ticks import Ticks
from src.collection.crsp_ticks import CRSPDailyTicks
from src.storage.upload_app import UploadApp
from utils.logger import setup_logger


def example_1_collect_fundamental():
    """
    Example 1: Use DataCollectors to fetch fundamental data for a single company.

    This method:
    - Fetches all concepts from approved_mapping.yaml
    - Returns quarterly data (no forward-filling)
    - Returns a Polars DataFrame with columns: [timestamp, rev, net_inc, ta, tl, ...]
    """
    print("=" * 80)
    print("Example 1: DataCollectors.collect_fundamental_year()")
    print("=" * 80)

    # Setup logger
    logger = setup_logger(
        name='example_fundamental',
        log_dir=Path('data/logs/examples'),
        level=logging.INFO,
        console_output=True
    )

    # Initialize DataCollectors
    alpaca_ticks = Ticks()
    crsp_ticks = CRSPDailyTicks()

    collector = DataCollectors(
        crsp_ticks=crsp_ticks,
        alpaca_ticks=alpaca_ticks,
        alpaca_headers={},
        logger=logger
    )

    # Example: Fetch fundamental data for Apple (AAPL)
    symbol = 'AAPL'
    cik = '0000320193'  # Apple's CIK
    year = 2024

    print(f"\nFetching fundamental data for {symbol} (CIK: {cik}) in {year}...")
    print("Using concept-based extraction with approved_mapping.yaml\n")

    # Fetch data
    df = collector.collect_fundamental_year(
        cik=cik,
        year=year,
        symbol=symbol
    )

    # Display results
    if len(df) > 0:
        print(f"✓ Successfully fetched {len(df)} quarterly filings")
        print(f"  Columns ({len(df.columns)}): {df.columns[:10]}...")  # Show first 10 columns
        print(f"\n  First 2 quarters:")
        print(df.head(2))
    else:
        print("✗ No data found")

    return df


def example_2_upload_fundamental():
    """
    Example 2: Use UploadApp to upload fundamental data to S3.

    This demonstrates:
    - Uploading fundamental data for a specific year
    - Batch CIK pre-fetching for performance
    - Concept-based extraction with approved_mapping.yaml
    - Concurrent processing with rate limiting
    """
    print("\n" + "=" * 80)
    print("Example 2: UploadApp.fundamental()")
    print("=" * 80)

    # Initialize UploadApp
    app = UploadApp()

    try:
        # Upload fundamental data for 2024
        # This will:
        # - Load all symbols for 2024
        # - Batch pre-fetch CIKs
        # - Fetch data using concept-based extraction
        # - Upload to S3: data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet

        year = 2024
        max_workers = 50  # Concurrent workers (rate limited to 9.5 req/sec)
        overwrite = False  # Set to True to overwrite existing data

        print(f"\nUploading fundamental data for {year}...")
        print(f"  Max workers: {max_workers} (rate limited to 9.5 req/sec)")
        print(f"  Overwrite: {overwrite}")
        print(f"  Using: approved_mapping.yaml (29 concepts)")
        print(f"  Storage: data/raw/fundamental/{{symbol}}/{year}/fundamental.parquet\n")

        # CAUTION: This will upload data for ALL symbols in the universe!
        # For testing, you may want to modify the universe in upload_app.py
        # or test with a smaller year/subset

        app.upload_fundamental(year=year, max_workers=max_workers, overwrite=overwrite)

        print(f"\n✓ Upload completed for {year}")

    finally:
        app.close()


def example_3_upload_single_symbol():
    """
    Example 3: Upload fundamental data for a single symbol (manual approach).

    Demonstrates how to upload data for just one company without processing the entire universe.
    """
    print("\n" + "=" * 80)
    print("Example 3: Upload Single Symbol")
    print("=" * 80)

    app = UploadApp()

    try:
        # Manually process a single symbol
        symbol = 'RKLB'  # Rocket Lab USA Inc.
        cik = '1819994'
        year = 2024
        overwrite = True

        print(f"\nUploading fundamental data for {symbol} (CIK: {cik}) in {year}...")

        result = app._process_symbol_fundamental(
            sym=symbol,
            year=year,
            overwrite=overwrite,
            cik=cik
        )

        # Display result
        print(f"\nResult:")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Status: {result['status']}")
        print(f"  Error: {result.get('error', 'None')}")

        if result['status'] == 'success':
            print(f"\n✓ Successfully uploaded to: data/raw/fundamental/{symbol}/{year}/fundamental.parquet")

    finally:
        app.close()


if __name__ == "__main__":
    # Run examples
    print("\n")
    print("=" * 80)
    print("FUNDAMENTAL DATA COLLECTION & UPLOAD EXAMPLES")
    print("=" * 80)
    print()

    # Example 1: Fetch fundamental data using DataCollectors
    df = example_1_collect_fundamental()

    # Example 3: Upload single symbol (safer for testing)
    example_3_upload_single_symbol()

    # Example 2: Upload all symbols for a year (COMMENTED OUT FOR SAFETY)
    # CAUTION: This will process ALL symbols in the universe!
    # Uncomment only when ready to do a full upload
    # example_2_upload_fundamental()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)