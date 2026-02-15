"""Feature build handler — builds Arrow IPC wide tables from raw data."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantdl.features.builder import FeatureBuilder
    from quantdl.master.security_master import SecurityMaster
    from quantdl.universe.manager import UniverseManager
    from quantdl.utils.calendar import TradingCalendar


class FeaturesHandler:
    """Builds feature wide tables from raw ticks and fundamental data."""

    def __init__(
        self,
        feature_builder: FeatureBuilder,
        security_master: SecurityMaster,
        universe_manager: UniverseManager,
        calendar: TradingCalendar,
        logger: logging.Logger,
    ) -> None:
        self.feature_builder = feature_builder
        self.security_master = security_master
        self.universe_manager = universe_manager
        self.calendar = calendar
        self.logger = logger

    def build(
        self,
        start_year: int,
        end_year: int,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Build all features for the given year range.

        Returns:
            Dict with field_name → output path mappings.
        """
        # Collect trading days
        start = date(start_year, 1, 1)
        end = date(end_year, 12, 31)
        trading_days = self.calendar.get_trading_days(start, end)

        # Collect security IDs across all years
        all_sids: set[str] = set()
        for year in range(start_year, end_year + 1):
            symbols = self.universe_manager.load_symbols_for_year(year, sym_type="alpaca")
            for sym in symbols:
                try:
                    sid = self.security_master.get_security_id(sym, f"{year}-12-31")
                    all_sids.add(str(sid))
                except ValueError:
                    pass

        security_ids = sorted(all_sids)
        self.logger.info(
            f"Building features: {len(trading_days)} trading days, "
            f"{len(security_ids)} securities, {start_year}-{end_year}"
        )

        return self.feature_builder.build_all(
            trading_days=trading_days,
            security_ids=security_ids,
            overwrite=overwrite,
        )
