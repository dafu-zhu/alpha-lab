"""
Script to fetch and display fundamental data for three stocks.

Usage:
    python scripts/fetch_fundamentals.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from collection.fundamental import Fundamental
from storage.config_loader import UploadConfig
from utils.mapping import symbol_cik_mapping


def fetch_and_print_fundamental(symbol: str, cik: str, year: int = 2024):
    """
    Fetch and print fundamental data for a single stock.

    :param symbol: Stock symbol (e.g., 'AAPL')
    :param cik: CIK number for the company
    :param year: Year to fetch data for (default: 2024)
    """
    print(f"\n{'=' * 80}")
    print(f"Fetching fundamental data for {symbol} (CIK: {cik}) - Year {year}")
    print(f"{'=' * 80}\n")

    try:
        # Initialize Fundamental instance
        fnd = Fundamental(cik, symbol)

        # Load config to get field lists
        config = UploadConfig()
        dei_fields = config.dei_fields
        gaap_fields = config.us_gaap_fields

        # Fetch DEI (Document and Entity Information) fields
        print(f"Fetching DEI fields: {dei_fields[:5]}... (showing first 5)")
        dei_df = fnd.collect_fields(year=year, fields=dei_fields, location='dei')

        # Fetch US-GAAP fields
        print(f"Fetching US-GAAP fields: {gaap_fields[:5]}... (showing first 5)")
        gaap_df = fnd.collect_fields(year=year, fields=gaap_fields, location='us-gaap')

        # Merge on Date column
        combined_df = dei_df.join(
            gaap_df,
            on='Date',
            how='inner'
        )

        print(f"\nData successfully fetched!")
        print(f"Shape: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")

        # Check for completely null columns
        null_columns = []
        for col in combined_df.columns:
            if combined_df[col].is_null().all():
                null_columns.append(col)

        print(f"\nCompletely null columns: {len(null_columns)}/{combined_df.shape[1]}")
        if null_columns:
            print(f"Null columns: {null_columns[:10]}{'...' if len(null_columns) > 10 else ''}")

        # Non-null columns
        non_null_columns = [col for col in combined_df.columns if col not in null_columns]
        print(f"Non-null columns: {len(non_null_columns)}/{combined_df.shape[1]}")

        print(f"\nColumns: {combined_df.columns}")
        print(f"\n{combined_df}")

        return combined_df

    except Exception as e:
        print(f"\n❌ Error fetching data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to fetch fundamental data for three stocks."""

    # Load symbol to CIK mapping
    print("Loading symbol to CIK mapping...")
    cik_map = symbol_cik_mapping()

    # Define three stocks to fetch
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    year = 2024

    print(f"\n{'#' * 80}")
    print(f"# Fetching fundamental data for {len(symbols)} stocks: {', '.join(symbols)}")
    print(f"# Year: {year}")
    print(f"{'#' * 80}")

    results = {}

    # Fetch data for each symbol
    for symbol in symbols:
        try:
            cik = cik_map[symbol]
            df = fetch_and_print_fundamental(symbol, cik, year)
            results[symbol] = df
        except KeyError:
            print(f"\n❌ Symbol {symbol} not found in CIK mapping")
            results[symbol] = None
        except Exception as e:
            print(f"\n❌ Unexpected error for {symbol}: {e}")
            results[symbol] = None

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    success_count = sum(1 for df in results.values() if df is not None)
    print(f"\nSuccessfully fetched: {success_count}/{len(symbols)}")

    for symbol, df in results.items():
        if df is not None:
            print(f"  ✓ {symbol}: {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            print(f"  ✗ {symbol}: Failed")

    print()


if __name__ == "__main__":
    main()
