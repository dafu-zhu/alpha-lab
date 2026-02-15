"""Build group mask wide tables from security master metadata."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl


class GroupFeatureBuilder:
    """Builds wide tables for group fields (sector, industry, subindustry, exchange)."""

    def __init__(self, security_master, logger: logging.Logger | None = None) -> None:
        self._security_master = security_master
        self._logger = logger or logging.getLogger(__name__)

    def _null_wide(
        self, trading_days: list[date], security_ids: list[str]
    ) -> pl.DataFrame:
        """Return a wide table of nulls (timestamp + one null Utf8 column per sid)."""
        calendar = pl.DataFrame({"timestamp": trading_days})
        return calendar.with_columns(
            [pl.lit(None).cast(pl.Utf8).alias(sid) for sid in security_ids]
        )

    def build(
        self,
        field_name: str,
        trading_days: list[date],
        security_ids: list[str],
    ) -> pl.DataFrame:
        """Build a wide table with constant group value per security across all days."""
        master_tb = getattr(self._security_master, "master_tb", None)
        if master_tb is None or "security_id" not in master_tb.columns:
            self._logger.debug("security_id not in security master, returning nulls")
            return self._null_wide(trading_days, security_ids)

        if field_name not in master_tb.columns:
            self._logger.debug(f"Column '{field_name}' not in security master, returning nulls")
            return self._null_wide(trading_days, security_ids)

        # Build lookup: security_id -> last known group value
        lookup = (
            master_tb.select(
                pl.col("security_id").cast(pl.Utf8),
                pl.col(field_name).cast(pl.Utf8),
            )
            .group_by("security_id")
            .agg(pl.col(field_name).last())
        )
        sid_to_value = dict(lookup.iter_rows())

        calendar = pl.DataFrame({"timestamp": trading_days})
        return calendar.with_columns(
            [pl.lit(sid_to_value.get(sid)).cast(pl.Utf8).alias(sid) for sid in security_ids]
        )
