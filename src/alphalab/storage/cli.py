import argparse
import datetime as dt

from alphalab.storage.app import UploadApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data lake upload workflows")
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--run-fundamental", action="store_true")
    parser.add_argument("--run-daily-ticks", action="store_true")
    parser.add_argument("--run-top-3000", action="store_true")
    parser.add_argument("--run-features", action="store_true")
    parser.add_argument("--run-all", action="store_true")

    parser.add_argument("--alpaca-start-year", type=int, default=2025)
    parser.add_argument("--daily-chunk-size", type=int, default=200)
    parser.add_argument("--daily-sleep-time", type=float, default=0.2)

    parser.add_argument("--max-workers", type=int, default=50)

    args = parser.parse_args()

    # Validate end_year not in future
    today = dt.date.today()
    if args.end_year > today.year:
        parser.error(f"--end-year {args.end_year} cannot exceed current year {today.year}")

    app = UploadApp(
        start_year=args.alpaca_start_year
    )
    try:
        app.run(
            start_year=args.start_year,
            end_year=args.end_year,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            daily_chunk_size=args.daily_chunk_size,
            daily_sleep_time=args.daily_sleep_time,
            run_fundamental=args.run_fundamental,
            run_daily_ticks=args.run_daily_ticks,
            run_top_3000=args.run_top_3000,
            run_features=args.run_features,
            run_all=args.run_all
        )
    finally:
        app.close()
