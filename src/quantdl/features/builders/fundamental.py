"""Build wide tables from raw fundamental data with forward-fill."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import polars as pl


class FundamentalFeatureBuilder:
    """Reads raw fundamental parquet files and produces wide tables with forward-fill."""

    def __init__(self, data_path: str, logger: logging.Logger | None = None) -> None:
        self._data_path = Path(data_path)
        self._logger = logger or logging.getLogger(__name__)

    def _read_concept(
        self, security_id: str, concept: str
    ) -> tuple[str, pl.DataFrame | None]:
        """Read a single concept for one security, returning (as_of_date, value)."""
        path = (
            self._data_path / "data" / "raw" / "fundamental" / security_id / "fundamental.parquet"
        )
        if not path.exists():
            return security_id, None
        try:
            df = pl.read_parquet(str(path))
            if df.schema["as_of_date"] == pl.String:
                df = df.with_columns(pl.col("as_of_date").str.to_date())
            filtered = df.filter(pl.col("concept") == concept)
            if len(filtered) == 0:
                return security_id, None
            # Deduplicate: keep last value per date
            deduped = (
                filtered.sort("as_of_date")
                .group_by("as_of_date")
                .agg(pl.col("value").last())
            )
            result = deduped.select(
                pl.col("as_of_date").alias("Date"),
                pl.col("value").cast(pl.Float64).alias(security_id),
            )
            return security_id, result
        except Exception as exc:
            self._logger.debug(f"Failed reading {path} for {concept}: {exc}")
            return security_id, None

    def build_raw(
        self,
        field_name: str,
        concept: str,
        trading_days: list[date],
        security_ids: list[str],
    ) -> pl.DataFrame:
        """Build wide table for a raw fundamental concept with forward-fill."""
        calendar = pl.DataFrame({"Date": trading_days})
        results: dict[str, pl.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {
                pool.submit(self._read_concept, sid, concept): sid
                for sid in security_ids
            }
            for future in as_completed(futures):
                sid, df = future.result()
                if df is not None and len(df) > 0:
                    results[sid] = df

        wide = calendar.clone()
        for sid, df in results.items():
            wide = wide.join(df, on="Date", how="left")

        # Sort columns for deterministic output, then forward-fill
        sorted_cols = sorted(c for c in wide.columns if c != "Date")
        wide = wide.select("Date", *sorted_cols)
        if sorted_cols:
            wide = wide.with_columns([pl.col(c).forward_fill() for c in sorted_cols])

        return wide.sort("Date")

    # ── Arithmetic helpers ──

    @staticmethod
    def _common_sids(security_ids: list[str], *dfs: pl.DataFrame) -> list[str]:
        """Return security_ids present in all DataFrames."""
        common = set(security_ids)
        for df in dfs:
            common &= set(df.columns)
        common.discard("Date")
        return sorted(common)

    @staticmethod
    def _binop(
        a: pl.DataFrame,
        b: pl.DataFrame,
        sids: list[str],
        op: str,
    ) -> pl.DataFrame:
        """Apply a binary operation (+, -, *) across matching security columns."""
        merged = a.select("Date", *sids).join(
            b.select("Date", *[pl.col(c).alias(f"{c}_b") for c in sids]),
            on="Date",
            how="left",
        )
        ops = {
            "+": lambda c: (pl.col(c) + pl.col(f"{c}_b")).alias(c),
            "-": lambda c: (pl.col(c) - pl.col(f"{c}_b")).alias(c),
            "*": lambda c: (pl.col(c) * pl.col(f"{c}_b")).alias(c),
        }
        return merged.select("Date", *[ops[op](c) for c in sids])

    @staticmethod
    def _safe_div(
        num_df: pl.DataFrame, den_df: pl.DataFrame, sids: list[str]
    ) -> pl.DataFrame:
        """Divide num_df by den_df, returning null where denominator is zero."""
        merged = num_df.select(
            "Date", *[pl.col(c).alias(f"{c}_num") for c in sids]
        ).join(
            den_df.select("Date", *[pl.col(c).alias(f"{c}_den") for c in sids]),
            on="Date",
            how="left",
        )
        return merged.select(
            "Date",
            *[
                pl.when(pl.col(f"{c}_den") != 0)
                .then(pl.col(f"{c}_num") / pl.col(f"{c}_den"))
                .otherwise(None)
                .alias(c)
                for c in sids
            ],
        )

    def _empty_wide(self, trading_days: list[date]) -> pl.DataFrame:
        """Return a timestamp-only DataFrame (used when no common sids exist)."""
        return pl.DataFrame({"Date": trading_days})

    # ── Derived field builder ──

    def build_derived(
        self,
        field_name: str,
        built: dict[str, pl.DataFrame],
        trading_days: list[date],
        security_ids: list[str],
    ) -> pl.DataFrame:
        """Build derived fundamental fields from already-built dependencies."""

        # Simple two-operand formulas: (field, op, left_key, right_key)
        _TWO_OP = {
            "debt":               ("+", "debt_lt", "debt_st"),
            "ebitda":             ("+", "operating_income", "depre_amort"),
            "working_capital":    ("-", "assets_curr", "liabilities_curr"),
            "eps":                ("/", "income", "sharesout"),
            "bookvalue_ps":       ("/", "equity", "sharesout"),
            "current_ratio":      ("/", "assets_curr", "liabilities_curr"),
            "return_equity":      ("/", "income", "equity"),
            "return_assets":      ("/", "income", "assets"),
            "sales_ps":           ("/", "sales", "sharesout"),
            "inventory_turnover": ("/", "cogs", "inventory"),
        }

        if field_name in _TWO_OP:
            op, left_key, right_key = _TWO_OP[field_name]
            sids = self._common_sids(security_ids, built[left_key], built[right_key])
            if not sids:
                return self._empty_wide(trading_days)
            if op == "/":
                return self._safe_div(built[left_key], built[right_key], sids)
            return self._binop(built[left_key], built[right_key], sids, op)

        # a + b - c pattern
        if field_name == "invested_capital":
            sids = self._common_sids(security_ids, built["equity"], built["debt_lt"], built["cash"])
            if not sids:
                return self._empty_wide(trading_days)
            return self._binop(
                self._binop(built["equity"], built["debt_lt"], sids, "+"),
                built["cash"], sids, "-",
            )

        if field_name == "enterprise_value":
            sids = self._common_sids(security_ids, built["cap"], built["debt_lt"], built["cash"])
            if not sids:
                return self._empty_wide(trading_days)
            return self._binop(
                self._binop(built["cap"], built["debt_lt"], sids, "+"),
                built["cash"], sids, "-",
            )

        # a + b + c pattern
        if field_name == "operating_expense":
            sids = self._common_sids(
                security_ids, built["cogs"], built["sga_expense"], built["depre_amort"]
            )
            if not sids:
                return self._empty_wide(trading_days)
            return self._binop(
                self._binop(built["cogs"], built["sga_expense"], sids, "+"),
                built["depre_amort"], sids, "+",
            )

        # QoQ growth approximation
        if field_name == "sales_growth":
            sales = built["sales"]
            cols = [c for c in sales.columns if c != "Date" and c in set(security_ids)]
            if not cols:
                return self._empty_wide(trading_days)
            return sales.select(
                "Date",
                *[(pl.col(c) / pl.col(c).shift(63) - 1).alias(c) for c in cols],
            )

        raise ValueError(f"Unknown derived field: {field_name}")
