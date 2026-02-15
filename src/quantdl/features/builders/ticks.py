"""Build wide tables from raw daily ticks data."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import polars as pl


class TicksFeatureBuilder:
    """Reads raw ticks parquet files and produces wide (time × security_id) tables."""

    def __init__(self, data_path: str, logger: logging.Logger | None = None) -> None:
        self._data_path = Path(data_path)
        self._logger = logger or logging.getLogger(__name__)

    def _read_column(
        self, security_id: str, column: str
    ) -> tuple[str, pl.DataFrame | None]:
        """Read a single column from one security's ticks file."""
        path = self._data_path / "data" / "raw" / "ticks" / "daily" / security_id / "ticks.parquet"
        if not path.exists():
            return security_id, None
        try:
            df = pl.read_parquet(str(path), columns=["timestamp", column])
            if df.schema["timestamp"] == pl.String:
                df = df.with_columns(pl.col("timestamp").str.to_date())
            df = df.rename({column: security_id})
            return security_id, df
        except Exception as exc:
            self._logger.debug(f"Failed reading {path}: {exc}")
            return security_id, None

    def build_direct(
        self,
        field_name: str,
        ticks_column: str,
        trading_days: list[date],
        security_ids: list[str],
    ) -> pl.DataFrame:
        """Build wide table for a direct ticks column (close, open, etc.)."""
        calendar = pl.DataFrame({"timestamp": trading_days})
        results: dict[str, pl.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {
                pool.submit(self._read_column, sid, ticks_column): sid
                for sid in security_ids
            }
            for future in as_completed(futures):
                sid, df = future.result()
                if df is not None and len(df) > 0:
                    results[sid] = df

        wide = calendar.clone()
        for sid, df in results.items():
            wide = wide.join(df, on="timestamp", how="left")

        return wide.sort("timestamp")

    @staticmethod
    def _value_cols(df: pl.DataFrame, security_ids: list[str]) -> list[str]:
        """Return columns present in both the DataFrame and security_ids list."""
        sid_set = set(security_ids)
        return [c for c in df.columns if c != "timestamp" and c in sid_set]

    def build_computed(
        self,
        field_name: str,
        built: dict[str, pl.DataFrame],
        trading_days: list[date],
        security_ids: list[str],
    ) -> pl.DataFrame:
        """Build computed ticks-derived fields (returns, adv20, cap, split)."""
        if field_name == "returns":
            close = built["close"]
            cols = self._value_cols(close, security_ids)
            return close.select(
                "timestamp",
                *[(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in cols],
            )

        if field_name == "adv20":
            volume = built["volume"]
            cols = self._value_cols(volume, security_ids)
            return volume.select(
                "timestamp",
                *[pl.col(c).rolling_mean(window_size=20).alias(c) for c in cols],
            )

        if field_name == "cap":
            close = built["close"]
            sharesout = built["sharesout"]
            common_sids = sorted(
                set(close.columns) & set(sharesout.columns) - {"timestamp"}
            )
            if not common_sids:
                return pl.DataFrame({"timestamp": trading_days})
            merged = close.select("timestamp", *common_sids).join(
                sharesout.select("timestamp", *[pl.col(c).alias(f"{c}_so") for c in common_sids]),
                on="timestamp",
                how="left",
            )
            return merged.select(
                "timestamp",
                *[(pl.col(c) * pl.col(f"{c}_so")).alias(c) for c in common_sids],
            )

        if field_name == "split":
            # Stub: return 1.0 for all
            calendar = pl.DataFrame({"timestamp": trading_days})
            return calendar.with_columns([pl.lit(1.0).alias(sid) for sid in security_ids])

        raise ValueError(f"Unknown computed ticks field: {field_name}")
