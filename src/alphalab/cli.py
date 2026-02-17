"""Unified CLI for AlphaLab.

Commands:
    alab --master               Build security_master + calendar_master
    alab --all                  Master build + download all data
    alab --ticks                Download daily ticks only
    alab --fundamental          Download fundamentals only
    alab --top-3000             Download top 3000 universe only
    alab --features             Build feature wide tables only
"""

import argparse
import datetime as dt
import os
import shutil
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


def _get_local_storage_path() -> Path:
    """Get LOCAL_STORAGE_PATH or raise."""
    base = os.getenv("LOCAL_STORAGE_PATH", "")
    if not base:
        raise ValueError("LOCAL_STORAGE_PATH environment variable required")
    return Path(os.path.expanduser(base))


def _build_master(logger) -> None:
    """Build security_master + calendar_master under LOCAL_STORAGE_PATH."""
    from alphalab.master.security_master import SecurityMaster
    from alphalab.utils.logger import console_log

    storage = _get_local_storage_path()
    master_dir = storage / "data" / "meta" / "master"
    master_dir.mkdir(parents=True, exist_ok=True)

    # --- Security Master ---
    console_log(logger, "Security Master", section=True)
    source = Path(__file__).resolve().parent / "data" / "security_master.parquet"
    working = master_dir / "security_master.parquet"

    if not working.exists():
        if source.exists():
            shutil.copy2(str(source), str(working))
            logger.debug(f"Initialized working copy from source parquet")
        else:
            raise FileNotFoundError(
                f"No source parquet at {source}. "
                "Run: python scripts/build_security_master.py"
            )
    else:
        logger.debug(f"Using existing working copy at {working}")

    start_time = time.time()
    master = SecurityMaster(local_path=working)
    result = master.update()
    master.save_local(working)
    elapsed = time.time() - start_time
    logger.info(
        f"Successfully updated SecurityMaster in {elapsed:.1f}s "
        f"({len(master.master_tb)} rows, {result['extended']} extended, {result['added']} added)"
    )

    # --- Calendar Master ---
    console_log(logger, "Calendar Master", section=True)
    calendar_path = master_dir / "calendar_master.parquet"
    _build_calendar_master(calendar_path, logger)


def _build_calendar_master(output_path: Path, logger) -> None:
    """Fetch NYSE trading calendar from Alpaca API and save as parquet."""
    import polars as pl

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        logger.error("ALPACA_API_KEY and ALPACA_API_SECRET required for calendar")
        return

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    params = {
        "start": "2009-01-01T00:00:00Z",
        "end": "2029-12-31T00:00:00Z",
        "date_type": "TRADING",
    }

    start_time = time.time()
    logger.debug("Fetching NYSE trading calendar from Alpaca API...")
    resp = requests.get(
        "https://paper-api.alpaca.markets/v2/calendar",
        headers=headers,
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()

    dates = [
        dt.datetime.strptime(entry["date"], "%Y-%m-%d").date()
        for entry in data
    ]

    df = pl.DataFrame({"timestamp": dates})
    df.write_parquet(output_path)
    elapsed = time.time() - start_time
    logger.info(f"Successfully built calendar in {elapsed:.1f}s ({len(dates)} trading days)")


def _download(args, logger) -> None:
    """Run data download pipeline."""
    from alphalab.storage.app import UploadApp

    app = UploadApp(start_year=args.start)
    try:
        app.run(
            start_year=args.start,
            end_year=args.end,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            daily_chunk_size=args.daily_chunk_size,
            daily_sleep_time=args.daily_sleep_time,
            run_fundamental=args.fundamental or args.all,
            run_daily_ticks=args.ticks or args.all,
            run_top_3000=args.top_3000 or args.all,
            run_features=args.features or args.all,
            run_all=False,
        )
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(
        description="AlphaLab CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Commands
    parser.add_argument(
        "--master", action="store_true",
        help="Build security_master + calendar_master",
    )
    parser.add_argument("--all", action="store_true",
                        help="Master build + download all data")
    parser.add_argument("--ticks", action="store_true",
                        help="Download daily ticks")
    parser.add_argument("--fundamental", action="store_true",
                        help="Download fundamentals")
    parser.add_argument("--top-3000", action="store_true",
                        help="Download top 3000 universe")
    parser.add_argument("--features", action="store_true",
                        help="Build feature wide tables")

    # Options
    parser.add_argument("--start", type=int, default=2017,
                        help="Start year (default: 2017)")
    parser.add_argument("--end", type=int, default=2025,
                        help="End year (default: 2025)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--daily-chunk-size", type=int, default=200)
    parser.add_argument("--daily-sleep-time", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=50)

    args = parser.parse_args()

    import logging
    from alphalab.utils.logger import setup_logger
    logger = setup_logger(
        name="alab",
        log_dir=Path("data/logs"),
        level=logging.INFO,
        console_output=True,
    )

    today = dt.date.today()
    if args.end > today.year:
        parser.error(f"--end {args.end} cannot exceed current year {today.year}")

    has_download = (args.all or args.ticks or args.fundamental
                    or args.top_3000 or args.features)

    if not args.master and not has_download:
        parser.print_help()
        return

    # --all includes --master
    if args.master or args.all:
        _build_master(logger)

    if has_download:
        _download(args, logger)


if __name__ == "__main__":
    main()
