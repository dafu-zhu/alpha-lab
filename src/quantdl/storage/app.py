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
from quantdl.storage.clients import StorageClient
from quantdl.storage.pipeline import DataCollectors, DataPublishers, Validator
from quantdl.storage.utils import CIKResolver, RateLimiter
from quantdl.collection.alpaca_ticks import Ticks
from quantdl.master.security_master import SecurityMaster
from quantdl.universe.manager import UniverseManager

load_dotenv()


class UploadApp:
    """
    Orchestrates data upload workflows.

    Delegates actual upload logic to specialized handlers while managing
    shared resources like storage client and rate limiters.
    """

    def __init__(self, start_year: int = 2017):
        # Core infrastructure
        local_path = os.getenv('LOCAL_STORAGE_PATH')
        if not local_path:
            raise ValueError(
                "LOCAL_STORAGE_PATH environment variable required"
            )
        self.client = StorageClient(local_path)
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

        # SecurityMaster: local only
        self.security_master = SecurityMaster()

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
            storage_client=self.client,
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
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            security_master=self.security_master,
            validator=self.validator,
            logger=self.logger,
        )

    def _get_fundamental_handler(self):
        from quantdl.storage.handlers.fundamental import FundamentalHandler
        return FundamentalHandler(
            data_publishers=self.data_publishers,
            data_collectors=self.data_collectors,
            cik_resolver=self.cik_resolver,
            security_master=self.security_master,
            universe_manager=self.universe_manager,
            validator=self.validator,
            sec_rate_limiter=self.sec_rate_limiter,
            logger=self.logger
        )

    def _get_features_handler(self):
        from quantdl.storage.handlers.features import FeaturesHandler
        from quantdl.features.builder import FeatureBuilder
        feature_builder = FeatureBuilder(
            data_path=str(self.client.base_path),
            security_master=self.security_master,
            logger=self.logger,
        )
        return FeaturesHandler(
            feature_builder=feature_builder,
            security_master=self.security_master,
            universe_manager=self.universe_manager,
            calendar=self.calendar,
            logger=self.logger,
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

    # ===========================
    # Public upload methods
    # ===========================

    def upload_daily_ticks(
        self,
        start_year: int,
        end_year: int,
        overwrite: bool = False,
        chunk_size: int = 50,
        sleep_time: float = 0.2,
    ):
        """Upload daily ticks for all securities active in [start_year, end_year]."""
        handler = self._get_daily_ticks_handler()
        return handler.upload_range(
            start_year=start_year,
            end_year=end_year,
            overwrite=overwrite,
            chunk_size=chunk_size,
            sleep_time=sleep_time,
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
        daily_chunk_size: int = 200,
        daily_sleep_time: float = 0.2,
        run_fundamental: bool = False,
        run_daily_ticks: bool = False,
        run_top_3000: bool = False,
        run_features: bool = False,
        run_all: bool = False
    ) -> None:
        """Run complete upload workflow."""
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        import time as _time
        _start_time = _time.time()

        if run_all:
            run_fundamental = True
            run_daily_ticks = True
            run_top_3000 = True
            run_features = True

        console_log(self.logger, f"Upload: {start_year}-{end_year}", section=True)

        # Daily ticks
        if run_daily_ticks:
            console_log(self.logger, f"Daily Ticks ({start_year}-{end_year})", section=True)
            self._run_daily_ticks(start_year, end_year, overwrite, daily_chunk_size, daily_sleep_time)

        # Top 3000 by year
        if run_top_3000:
            for year in range(start_year, end_year + 1):
                self.upload_top_3000_monthly(year, overwrite=overwrite)

        # Fundamentals (after ticks)
        if run_fundamental:
            console_log(self.logger, f"Raw Fundamental: {start_date} to {end_date}", section=True)
            self.upload_fundamental(start_date, end_date, max_workers, overwrite)

        # Features (after raw data uploads)
        if run_features:
            console_log(self.logger, f"Features: {start_year}-{end_year}", section=True)
            handler = self._get_features_handler()
            handler.build(start_year, end_year, overwrite=overwrite)

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
        """Run daily ticks upload using Alpaca (single range, SecurityMaster-driven)."""
        handler = self._get_daily_ticks_handler()

        today = dt.date.today()
        effective_end = min(end_year, today.year)
        handler.upload_range(
            start_year=start_year,
            end_year=effective_end,
            overwrite=overwrite,
            chunk_size=chunk_size,
            sleep_time=sleep_time,
        )

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'security_master') and self.security_master:
            self.security_master.close()
