#!/usr/bin/env python3
"""
Generate top 3000 stock universes for each month from 2010-01 to 2026-01.

For each month, uses YYYY-MM-01 to fetch the top 3000 most liquid stocks
and stores them in data/symbols/YYYY/MM/universe_top3000.txt

Usage:
    python scripts/generate_monthly_top3000.py
"""

import sys
from pathlib import Path
import datetime as dt
import logging

# Add src to path so we can import modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from stock_pool.universe_manager import UniverseManager
from utils.logger import setup_logger


def main():
    # Setup logging
    log_dir = Path("data/logs/universe_generation")
    logger = setup_logger(
        "universe_top3000_generator",
        log_dir,
        logging.INFO,
        console_output=True
    )

    logger.info("=" * 80)
    logger.info("Top 3000 Universe Generation Script")
    logger.info("=" * 80)
    logger.info("Generating monthly top 3000 universes from 2010-01 to 2026-01")
    logger.info("Using YYYY-MM-01 date for each month's universe")
    logger.info("")

    # Initialize UniverseManager
    um = UniverseManager()

    # Define date range: 2010-01-01 to 2026-01-01
    start_date = dt.date(2010, 1, 1)
    end_date = dt.date(2026, 1, 1)

    # Generate list of all months in the range
    current_date = start_date
    months = []
    while current_date <= end_date:
        months.append(current_date)
        # Move to next month
        if current_date.month == 12:
            current_date = dt.date(current_date.year + 1, 1, 1)
        else:
            current_date = dt.date(current_date.year, current_date.month + 1, 1)

    logger.info(f"Total months to process: {len(months)}")
    logger.info("")

    # Track statistics
    total_months = len(months)
    successful = 0
    failed = 0
    skipped = 0

    for idx, month_date in enumerate(months, 1):
        year = month_date.year
        month = month_date.month

        # Determine the fetch date (YYYY-MM-01)
        fetch_date_str = f"{year}-{month:02d}-01"

        # Determine output path
        output_dir = Path(f"data/symbols/{year}/{month:02d}")
        output_file = output_dir / "universe_top3000.txt"

        # Skip if file already exists
        if output_file.exists():
            logger.info(f"[{idx}/{total_months}] {year}-{month:02d}: SKIPPED (file exists)")
            skipped += 1
            continue

        logger.info(f"[{idx}/{total_months}] Processing {year}-{month:02d} using {fetch_date_str}...")

        try:
            # Step 1: Get stock universe
            if year < 2025:
                all_symbols = um.get_hist_symbols(fetch_date_str)
                logger.info(f"  Historical universe: {len(all_symbols)} symbols")
            else:
                all_symbols = um.get_current_symbols(refresh=False)
                logger.info(f"  Current universe: {len(all_symbols)} symbols")

            # Step 2: Get top 3000 based on liquidity
            # Use 'crsp' for historical data (< 2025), 'alpaca' for recent data
            source = 'crsp' if year < 2025 else 'alpaca'
            logger.info(f"  Calculating top 3000 using {source} data...")

            top_3000 = um.get_top_3000(fetch_date_str, all_symbols, source)

            if len(top_3000) == 0:
                logger.error(f"  Failed to get top 3000 for {fetch_date_str}")
                failed += 1
                continue

            logger.info(f"  Top 3000 calculated: {len(top_3000)} symbols")

            # Step 3: Write to file
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                for symbol in top_3000:
                    f.write(f"{symbol}\n")

            logger.info(f"  Written to {output_file}")
            logger.info(f"  SUCCESS")
            successful += 1

        except Exception as e:
            logger.error(f"  ERROR: {str(e)}", exc_info=True)
            failed += 1
            continue

        logger.info("")

    # Final summary
    logger.info("=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)
    logger.info(f"Total months:  {total_months}")
    logger.info(f"Successful:    {successful}")
    logger.info(f"Skipped:       {skipped}")
    logger.info(f"Failed:        {failed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
