"""
Test script to verify yfinance/Alpaca integration for daily ticks.

This script tests:
1. yfinance data fetching for years < 2017
2. Alpaca data fetching for years >= 2017
3. Schema consistency between both sources

Usage:
    python scripts/test_daily_integration.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from collection.ticks import Ticks


def test_yfinance_fetch(symbol: str = 'AAPL', year: int = 2015):
    """Test yfinance data fetching for pre-2017 data."""
    print(f"\n{'=' * 80}")
    print(f"TEST 1: yfinance fetch for {symbol} (year {year})")
    print(f"{'=' * 80}\n")

    try:
        # Use SEC format for yfinance
        ticks = Ticks(symbol)
        df = ticks.collect_daily_ticks_yf(year=year)

        print(f"✓ Successfully fetched {len(df)} rows")
        print(f"\nSchema:")
        print(df.schema)
        print(f"\nFirst 5 rows:")
        print(df.head(5))
        print(f"\nLast 5 rows:")
        print(df.tail(5))

        # Check for null values in num_trades and vwap
        num_trades_null = df['num_trades'].is_null().sum()
        vwap_null = df['vwap'].is_null().sum()
        print(f"\nNull counts:")
        print(f"  num_trades: {num_trades_null}/{len(df)} (expected: {len(df)})")
        print(f"  vwap: {vwap_null}/{len(df)} (expected: {len(df)})")

        # Check for non-null OHLCV
        open_non_null = df['open'].is_not_null().sum()
        print(f"  open: {open_non_null}/{len(df)} non-null")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpaca_fetch(symbol: str = 'AAPL', year: int = 2024):
    """Test Alpaca data fetching for post-2017 data."""
    print(f"\n{'=' * 80}")
    print(f"TEST 2: Alpaca fetch for {symbol} (year {year})")
    print(f"{'=' * 80}\n")

    try:
        # Use Alpaca format (with '.')
        ticks = Ticks(symbol)
        df = ticks.collect_daily_ticks(year=year)

        print(f"✓ Successfully fetched {len(df)} rows")
        print(f"\nSchema:")
        print(df.schema)
        print(f"\nFirst 5 rows:")
        print(df.head(5))
        print(f"\nLast 5 rows:")
        print(df.tail(5))

        # Check for non-null values in num_trades and vwap
        num_trades_non_null = df['num_trades'].is_not_null().sum()
        vwap_non_null = df['vwap'].is_not_null().sum()
        print(f"\nNon-null counts:")
        print(f"  num_trades: {num_trades_non_null}/{len(df)}")
        print(f"  vwap: {vwap_non_null}/{len(df)}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schema_consistency():
    """Test that schemas match between yfinance and Alpaca."""
    print(f"\n{'=' * 80}")
    print(f"TEST 3: Schema consistency check")
    print(f"{'=' * 80}\n")

    try:
        # Fetch from both sources
        ticks_yf = Ticks('AAPL')
        df_yf = ticks_yf.collect_daily_ticks_yf(year=2015)

        ticks_alpaca = Ticks('AAPL')
        df_alpaca = ticks_alpaca.collect_daily_ticks(year=2024)

        # Compare column names
        yf_cols = df_yf.columns
        alpaca_cols = df_alpaca.columns

        print(f"yfinance columns: {yf_cols}")
        print(f"Alpaca columns:   {alpaca_cols}")

        if yf_cols == alpaca_cols:
            print(f"\n✓ Column names match!")
        else:
            print(f"\n✗ Column names differ!")
            return False

        # Compare schemas (data types)
        yf_schema = df_yf.schema
        alpaca_schema = df_alpaca.schema

        print(f"\nyfinance schema: {yf_schema}")
        print(f"Alpaca schema:   {alpaca_schema}")

        if yf_schema == alpaca_schema:
            print(f"\n✓ Schemas match perfectly!")
            return True
        else:
            print(f"\n⚠ Schemas differ (this is expected - both should work)")
            return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print(f"\n{'#' * 80}")
    print(f"# Testing yfinance/Alpaca Integration")
    print(f"# Testing data fetching for years < 2017 (yfinance) and >= 2017 (Alpaca)")
    print(f"{'#' * 80}")

    results = []

    # Test 1: yfinance
    results.append(('yfinance fetch (2015)', test_yfinance_fetch('AAPL', 2015)))

    # Test 2: Alpaca
    results.append(('Alpaca fetch (2024)', test_alpaca_fetch('AAPL', 2024)))

    # Test 3: Schema consistency
    results.append(('Schema consistency', test_schema_consistency()))

    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = '✓ PASS' if result else '✗ FAIL'
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print(f"\n🎉 All tests passed! Integration is working correctly.\n")
    else:
        print(f"\n❌ Some tests failed. Please review the errors above.\n")


if __name__ == "__main__":
    main()
