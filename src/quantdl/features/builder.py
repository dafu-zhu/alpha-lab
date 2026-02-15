"""Feature builder orchestrator — builds all features in dependency order."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from quantdl.features.registry import (
    ALL_FIELDS,
    FieldCategory,
    FieldSource,
    get_build_order,
)
from quantdl.features.builders.ticks import TicksFeatureBuilder
from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
from quantdl.features.builders.groups import GroupFeatureBuilder


class FeatureBuilder:
    """Orchestrates building all feature wide tables in dependency-safe order."""

    def __init__(
        self,
        data_path: str,
        security_master,
        logger: logging.Logger | None = None,
    ) -> None:
        self._data_path = Path(data_path)
        self._logger = logger or logging.getLogger(__name__)
        self._ticks = TicksFeatureBuilder(data_path, self._logger)
        self._fundamental = FundamentalFeatureBuilder(data_path, self._logger)
        self._groups = GroupFeatureBuilder(security_master, self._logger)
        self._built: dict[str, pl.DataFrame] = {}

    def build_all(
        self,
        trading_days: list[date],
        security_ids: list[str],
        overwrite: bool = False,
    ) -> dict[str, str]:
        """Build all features, writing Arrow IPC files.

        Returns:
            Dict mapping field_name → output file path.
        """
        out_dir = self._data_path / "data" / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        order = get_build_order()
        written: dict[str, str] = {}

        for field_name in order:
            out_path = out_dir / f"{field_name}.arrow"

            if not overwrite and out_path.exists():
                # Load into cache for downstream deps
                self._built[field_name] = pl.read_ipc(str(out_path))
                written[field_name] = str(out_path)
                self._logger.debug(f"Skipped (exists): {field_name}")
                continue

            df = self._build_single(field_name, trading_days, security_ids)
            self._built[field_name] = df
            df.write_ipc(str(out_path))
            written[field_name] = str(out_path)
            self._logger.debug(f"Built: {field_name} → {out_path}")

        return written

    def _build_single(
        self,
        field_name: str,
        trading_days: list[date],
        security_ids: list[str],
    ) -> pl.DataFrame:
        """Build a single feature field."""
        spec = ALL_FIELDS[field_name]

        # Direct ticks column
        if spec.source == FieldSource.TICKS and spec.ticks_column:
            return self._ticks.build_direct(
                field_name, spec.ticks_column, trading_days, security_ids
            )

        # Computed price/volume
        if spec.category == FieldCategory.PRICE_VOLUME and spec.source == FieldSource.COMPUTED:
            return self._ticks.build_computed(
                field_name, self._built, trading_days, security_ids
            )

        # Raw fundamental
        if spec.source == FieldSource.FUNDAMENTAL_RAW and spec.concept:
            return self._fundamental.build_raw(
                field_name, spec.concept, trading_days, security_ids
            )

        # Derived (arithmetic)
        if spec.category == FieldCategory.DERIVED:
            return self._fundamental.build_derived(
                field_name, self._built, trading_days, security_ids
            )

        # Group masks
        if spec.category == FieldCategory.GROUP:
            return self._groups.build(field_name, trading_days, security_ids)

        raise ValueError(f"Don't know how to build field: {field_name}")
