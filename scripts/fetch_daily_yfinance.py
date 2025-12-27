"""
Simple script to fetch daily ticks data from yfinance and transform to Polars DataFrame.

This demonstrates yfinance data fetching with the same schema as Alpaca minute/daily ticks.

Usage:
    python scripts/fetch_daily_yfinance.py
"""

import sys
from pathlib import Path
import yfinance as yf
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def fetch_daily_ticks_yfinance(symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
    """
    Fetch daily ticks data from yfinance and convert to Polars DataFrame.

    :param symbol: Stock symbol (e.g., 'AAPL', 'BRK-B')
    :param start_date: Start date in format 'YYYY-MM-DD'
    :param end_date: End date in format 'YYYY-MM-DD'
    :return: Polars DataFrame with daily ticks matching Alpaca schema
    """
    print(f"\n{'=' * 80}")
    print(f"Fetching daily ticks for {symbol} ({start_date} to {end_date})")
    print(f"{'=' * 80}\n")

    try:
        yf_symbol = symbol

        # Fetch data from yfinance
        print(f"Downloading from yfinance (symbol: {yf_symbol})...")
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if hist.empty:
            print(f"❌ No data returned for {symbol}")
            return None

        # Reset index to get Date as a column
        hist = hist.reset_index()

        # Create Polars DataFrame with schema matching daily ticks format
        # Note: yfinance doesn't provide num_trades and vwap, so we set them to None
        df = pl.DataFrame({
            'Date': hist['Date'].dt.date.tolist(),  # Convert to date objects
            'open': hist['Open'].astype(float).tolist(),
            'high': hist['High'].astype(float).tolist(),
            'low': hist['Low'].astype(float).tolist(),
            'close': hist['Close'].astype(float).tolist(),
            'volume': hist['Volume'].astype(int).tolist(),
            'num_trades': [None] * len(hist),  # yfinance doesn't provide this
            'vwap': [None] * len(hist)  # yfinance doesn't provide this
        }).with_columns([
            pl.col('Date').cast(pl.Date),
            pl.col('open').cast(pl.Float64),
            pl.col('high').cast(pl.Float64),
            pl.col('low').cast(pl.Float64),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Int64),
            pl.col('num_trades').cast(pl.Int64),
            pl.col('vwap').cast(pl.Float64)
        ])

        print(f"\n✓ Data successfully fetched!")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Show column info
        print(f"\nColumns: {df.columns}")
        print(f"Schema:\n{df.schema}")

        # Show basic statistics
        if df.shape[0] > 0:
            print(f"\nPrice Statistics:")
            print(f"  High: ${df['high'].max():.2f}")
            print(f"  Low: ${df['low'].min():.2f}")
            print(f"  Open (first): ${df['open'][0]:.2f}")
            print(f"  Close (last): ${df['close'][-1]:.2f}")
            print(f"  Total Volume: {df['volume'].sum():,.0f}")
            print(f"  Avg Daily Volume: {df['volume'].mean():,.0f}")

            # Date range
            print(f"\nDate Range:")
            print(f"  First date: {df['Date'][0]}")
            print(f"  Last date: {df['Date'][-1]}")
            print(f"  Total trading days: {df.shape[0]}")

        print(f"\nDataFrame Preview (first 10 rows):")
        print(df.head(10))

        print(f"\nDataFrame Preview (last 5 rows):")
        print(df.tail(5))

        return df

    except Exception as e:
        print(f"\n❌ Error fetching data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to demonstrate yfinance daily ticks fetching."""

    # Define test parameters
    symbols = ['AAPL', 'MSFT', 'BRK-B']  # Include BRK-B to test dash handling
    start_date = '2010-01-01'
    end_date = '2010-12-31'

    print(f"\n{'#' * 80}")
    print(f"# Fetching daily ticks data from yfinance")
    print(f"# Symbols: {', '.join(symbols)}")
    print(f"# Period: {start_date} to {end_date}")
    print(f"# Source: Yahoo Finance (yfinance)")
    print(f"{'#' * 80}")

    results = {}

    # Fetch data for each symbol
    for symbol in symbols:
        df = fetch_daily_ticks_yfinance(symbol, start_date, end_date)
        results[symbol] = df

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    success_count = sum(1 for df in results.values() if df is not None)
    total_count = len(symbols)
    print(f"\nSuccessfully fetched: {success_count}/{total_count}")

    for symbol, df in results.items():
        if df is not None:
            print(f"  ✓ {symbol}: {df.shape[0]} trading days")
        else:
            print(f"  ✗ {symbol}: Failed")

    print(f"\n{'=' * 80}")
    print("Note: yfinance provides adjusted prices by default (auto_adjust=True)")
    print("Fields 'num_trades' and 'vwap' are set to None (not available in yfinance)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()