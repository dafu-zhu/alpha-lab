"""Shared fixtures for feature tests."""

from datetime import date

import polars as pl
import pytest


@pytest.fixture
def trading_days():
    """10 trading days in Jan 2024."""
    return [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
        date(2024, 1, 11),
        date(2024, 1, 12),
        date(2024, 1, 16),
    ]


@pytest.fixture
def security_ids():
    """Two test security IDs."""
    return ["SEC001", "SEC002"]


@pytest.fixture
def raw_data_dir(tmp_path, trading_days, security_ids):
    """Create mock raw ticks + fundamental data under tmp_path."""
    # Ticks
    for sid in security_ids:
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / sid
        ticks_dir.mkdir(parents=True)
        df = pl.DataFrame({
            "timestamp": trading_days,
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [102.0 + i for i in range(10)],
            "volume": [1_000_000 + i * 100_000 for i in range(10)],
            "vwap": [101.0 + i for i in range(10)],
        })
        df.write_parquet(str(ticks_dir / "ticks.parquet"))

    # Fundamental: sparse quarterly data
    for idx, sid in enumerate(security_ids):
        fnd_dir = tmp_path / "data" / "raw" / "fundamental" / sid
        fnd_dir.mkdir(parents=True)
        base = 1e9 if idx == 0 else 5e8
        rows = []
        for concept in ["assets", "liabilities", "income", "sharesout", "equity",
                        "sales", "cogs", "operating_income", "depre_amort",
                        "debt_lt", "debt_st", "cash", "assets_curr", "liabilities_curr",
                        "sga_expense", "inventory", "capex"]:
            rows.append({
                "as_of_date": date(2024, 1, 5),
                "concept": concept,
                "value": base,
                "accn": "0001-24-000001",
                "form": "10-K",
            })
        df = pl.DataFrame(rows)
        df.write_parquet(str(fnd_dir / "fundamental.parquet"))

    return tmp_path
