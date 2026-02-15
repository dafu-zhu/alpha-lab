"""
Upload application orchestrator.

Coordinates data upload workflows using specialized handlers.
"""

import os
import datetime as dt
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from quantdl.utils.logger import setup_logger, console_log
from quantdl.utils.calendar import TradingCalendar
from quantdl.storage.clients import S3Client
from quantdl.storage.pipeline import DataCollectors, DataPublishers, Validator
from quantdl.storage.utils import UploadConfig, CIKResolver, RateLimiter
from quantdl.collection.alpaca_ticks import Ticks
from quantdl.master.security_master import SecurityMaster
from quantdl.universe.manager import UniverseManager

load_dotenv()


class UploadApp:
    """
    Orchestrates data upload workflows.

    Delegates actual upload logic to specialized handlers while managing
    shared resources like S3 client, WRDS connection, and rate limiters.
    """

    def __init__(self, start_year: int = 2017):
        # Core infrastructure
        self.config = UploadConfig()
        self.client = S3Client().client
        self.logger = setup_logger(
            name="uploadapp",
            log_dir=Path("data/logs/upload"),
            level=logging.DEBUG,
            console_output=True
        )

        self.validator = Validator(self.client)
        self.calendar = TradingCalendar()

        # Data fetchers
        self.alpaca_ticks = Ticks()

        # SecurityMaster: local → S3 (no WRDS)
        self.security_master = SecurityMaster(
            s3_client=self.client,
            bucket_name='us-equity-datalake'
        )

        # Universe and CIK resolution
        self.universe_manager = UniverseManager(
            security_master=self.security_master
        )

        self.sec_rate_limiter = RateLimiter(max_rate=9.5)
        self.cik_resolver = CIKResolver(
            security_master=self.security_master,
            logger=self.logger
        )

        # Alpaca credentials
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET")
        }

        # Data collectors and publishers
        self.data_collectors = DataCollectors(
            crsp_ticks=None,
            alpaca_ticks=self.alpaca_ticks,
            alpaca_headers=self.headers,
            logger=self.logger,
            sec_rate_limiter=self.sec_rate_limiter,
        )

        self.data_publishers = DataPublishers(
            s3_client=self.client,
            upload_config=self.config,
            logger=self.logger,
            data_collectors=self.data_collectors,
            security_master=self.security_master,
        )

        self._start_year = start_year

    # ===========================
    # Handler factory methods
    # ===========================

    def _get_daily_ticks_handler(self):
        from quantdl.storage.handlers.ticks import DailyTicksHandler
        return DailyTicksHandler(
            s3_client=self.client,
            bucket_name=self.config.bucket,
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            security_master=self.security_master,
            universe_manager=self.universe_manager,
            validator=self.validator,
            logger=self.logger,
        )

    def _get_minute_ticks_handler(self):
        from quantdl.storage.handlers.ticks import MinuteTicksHandler
        return MinuteTicksHandler(
            s3_client=self.client,
            bucket_name=self.config.bucket,
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            security_master=self.security_master,
            universe_manager=self.universe_manager,
            validator=self.validator,
            calendar=self.calendar,
            logger=self.logger
        )

    def _get_fundamental_handler(self):
        from quantdl.storage.handlers.fundamental import FundamentalHandler
        return FundamentalHandler(
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            cik_resolver=self.cik_resolver,
            universe_manager=self.universe_manager,
            validator=self.validator,
            sec_rate_limiter=self.sec_rate_limiter,
            logger=self.logger
        )

    def _get_ttm_handler(self):
        from quantdl.storage.handlers.fundamental import TTMHandler
        return TTMHandler(
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            cik_resolver=self.cik_resolver,
            universe_manager=self.universe_manager,
            validator=self.validator,
            sec_rate_limiter=self.sec_rate_limiter,
            logger=self.logger
        )

    def _get_derived_handler(self):
        from quantdl.storage.handlers.fundamental import DerivedHandler
        return DerivedHandler(
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            cik_resolver=self.cik_resolver,
            universe_manager=self.universe_manager,
            validator=self.validator,
            logger=self.logger
        )

    def _get_top3000_handler(self):
        from quantdl.storage.handlers.top3000 import Top3000Handler
        return Top3000Handler(
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            universe_manager=self.universe_manager,
            validator=self.validator,
            calendar=self.calendar,
            logger=self.logger,
        )

    def _get_sentiment_handler(self):
        from quantdl.storage.handlers.sentiment import SentimentHandler
        return SentimentHandler(
            s3_client=self.client,
            bucket=self.config.bucket,
            cik_resolver=self.cik_resolver,
            universe_manager=self.universe_manager,
            logger=self.logger
        )

    # ===========================
    # Public upload methods
    # ===========================

    def upload_daily_ticks(
        self,
        year: int,
        overwrite: bool = False,
        use_monthly_partitions: bool = True,
        by_year: bool = False,
        chunk_size: int = 200,
        sleep_time: float = 0.2,
        current_year: Optional[int] = None
    ):
        """Upload daily ticks for a year."""
        handler = self._get_daily_ticks_handler()
        return handler.upload_year(
            year=year,
            overwrite=overwrite,
            chunk_size=chunk_size,
            sleep_time=sleep_time,
            current_year=current_year
        )

    def upload_minute_ticks(
        self,
        year: int,
        month: int,
        overwrite: bool = False,
        resume: bool = False,
        num_workers: int = 50,
        chunk_size: int = 500,
        sleep_time: float = 0.0
    ):
        """Upload minute ticks for a single month."""
        self.upload_minute_ticks_year(
            year=year,
            months=[month],
            overwrite=overwrite,
            resume=resume,
            num_workers=num_workers,
            chunk_size=chunk_size,
            sleep_time=sleep_time
        )

    def upload_minute_ticks_year(
        self,
        year: int,
        months: list[int] = None,
        overwrite: bool = False,
        resume: bool = False,
        num_workers: int = 50,
        chunk_size: int = 500,
        sleep_time: float = 0.0
    ):
        """Upload minute ticks for multiple months."""
        handler = self._get_minute_ticks_handler()
        return handler.upload_year(
            year=year,
            months=months,
            overwrite=overwrite,
            resume=resume,
            num_workers=num_workers,
            chunk_size=chunk_size,
            sleep_time=sleep_time
        )

    def upload_fundamental(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ):
        """Upload raw fundamental data."""
        handler = self._get_fundamental_handler()
        return handler.upload(start_date, end_date, max_workers, overwrite)

    def upload_ttm_fundamental(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ):
        """Upload TTM fundamental data."""
        handler = self._get_ttm_handler()
        return handler.upload(start_date, end_date, max_workers, overwrite)

    def upload_derived_fundamental(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ):
        """Upload derived fundamental data."""
        handler = self._get_derived_handler()
        return handler.upload(start_date, end_date, max_workers, overwrite)

    def upload_sentiment(
        self,
        start_date: str,
        end_date: str,
        overwrite: bool = False
    ):
        """Upload sentiment data using FinBERT."""
        handler = self._get_sentiment_handler()
        return handler.upload(start_date, end_date, overwrite)

    def upload_top_3000_monthly(
        self,
        year: int,
        overwrite: bool = False,
        auto_resolve: bool = True
    ):
        """Upload top 3000 symbols for each month."""
        handler = self._get_top3000_handler()
        return handler.upload_year(year, overwrite, auto_resolve)

    # ===========================
    # Main run method
    # ===========================

    def run(
        self,
        start_year: int,
        end_year: int,
        max_workers: int = 50,
        overwrite: bool = False,
        resume: bool = False,
        chunk_size: int = 500,
        sleep_time: float = 0.0,
        daily_chunk_size: int = 200,
        daily_sleep_time: float = 0.2,
        minute_ticks_start_year: int = 2017,
        run_fundamental: bool = False,
        run_derived_fundamental: bool = False,
        run_ttm_fundamental: bool = False,
        run_daily_ticks: bool = False,
        run_minute_ticks: bool = False,
        run_top_3000: bool = False,
        run_sentiment: bool = False,
        run_all: bool = False
    ) -> None:
        """Run complete upload workflow."""
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        import time as _time
        _start_time = _time.time()

        if run_all:
            run_fundamental = True
            run_derived_fundamental = True
            run_ttm_fundamental = True
            run_daily_ticks = True
            # Note: minute ticks excluded from --run-all by default (use --run-minute-ticks explicitly)
            run_top_3000 = True
            run_sentiment = True

        console_log(self.logger, f"Upload: {start_year}-{end_year}", section=True)

        # Daily ticks
        if run_daily_ticks:
            console_log(self.logger, f"Daily Ticks ({start_year}-{end_year})", section=True)
            self._run_daily_ticks(start_year, end_year, overwrite, daily_chunk_size, daily_sleep_time)

        # Minute ticks and top 3000 by year
        for year in range(start_year, end_year + 1):
            if run_minute_ticks:
                today = dt.date.today()
                if year > today.year:
                    self.logger.debug(f"Skipping minute ticks for {year}: future")
                    continue

                if year >= minute_ticks_start_year:
                    console_log(self.logger, f"Minute Ticks ({year})", section=True)
                    self.upload_minute_ticks_year(
                        year, overwrite=overwrite, resume=resume,
                        num_workers=max_workers, chunk_size=chunk_size, sleep_time=sleep_time
                    )

            if run_top_3000:
                self.upload_top_3000_monthly(year, overwrite=overwrite)

        # Fundamentals (after ticks)
        if run_fundamental:
            console_log(self.logger, f"Raw Fundamental: {start_date} to {end_date}", section=True)
            self.upload_fundamental(start_date, end_date, max_workers, overwrite)

        if run_ttm_fundamental:
            console_log(self.logger, f"TTM Fundamental: {start_date} to {end_date}", section=True)
            self.upload_ttm_fundamental(start_date, end_date, max_workers, overwrite)

        if run_derived_fundamental:
            console_log(self.logger, f"Derived Fundamental: {start_date} to {end_date}", section=True)
            self.upload_derived_fundamental(start_date, end_date, max_workers, overwrite)

        if run_sentiment:
            sentiment_start = max(start_date, "2017-01-01")
            console_log(self.logger, f"Sentiment: {sentiment_start} to {end_date}", section=True)
            self.upload_sentiment(sentiment_start, end_date, overwrite)

        elapsed = _time.time() - _start_time
        minutes, seconds = divmod(int(elapsed), 60)
        console_log(self.logger, f"Done: Upload complete ({minutes}m {seconds:02d}s)", section=True)

    def _run_daily_ticks(
        self,
        start_year: int,
        end_year: int,
        overwrite: bool,
        chunk_size: int,
        sleep_time: float
    ):
        """Run daily ticks upload using Alpaca."""
        handler = self._get_daily_ticks_handler()

        today = dt.date.today()
        effective_end = min(end_year, today.year)
        for year in range(start_year, effective_end + 1):
            self.logger.debug(f"Uploading Alpaca daily ticks for {year}")
            handler.upload_year(
                year=year,
                overwrite=overwrite,
                chunk_size=chunk_size,
                sleep_time=sleep_time
            )

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'security_master') and self.security_master:
            self.security_master.close()
