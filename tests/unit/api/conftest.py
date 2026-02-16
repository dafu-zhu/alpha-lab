"""Shared test fixtures for AlphaLab API."""

from collections.abc import Generator
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary test data directory with parquet files.

    Files are created at paths matching the storage directory structure.
    E.g., data/meta/master/security_master.parquet
    """
    # Create security master test data
    security_master_dir = tmp_path / "data" / "meta" / "master"
    security_master_dir.mkdir(parents=True)
    security_master = pl.DataFrame({
        "security_id": ["SEC001", "SEC002", "SEC003", "SEC004"],
        "symbol": ["AAPL", "MSFT", "GOOGL", "META"],
        "company": ["Apple Inc", "Microsoft Corp", "Alphabet Inc", "Meta Platforms"],
        "cik": ["0000320193", "0000789019", "0001652044", "0001326801"],
        "start_date": [
            date(2000, 1, 1),
            date(2000, 1, 1),
            date(2004, 8, 19),
            date(2012, 5, 18),
        ],
        "end_date": [None, None, None, None],
    })
    security_master.write_parquet(security_master_dir / "security_master.parquet")

    # Create calendar master test data (trading days)
    # Include weekdays from test data range (skip weekends)
    trading_days = [
        # Jan 2024 (for daily tests)
        date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5),
        date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10),
        # Q1-Q2 2024 (for fundamentals/metrics tests) - sample days
        date(2024, 3, 29), date(2024, 4, 1),
        date(2024, 6, 28),
    ]
    calendar_master = pl.DataFrame({"date": trading_days})
    calendar_master.write_parquet(security_master_dir / "calendar_master.parquet")

    # Create daily ticks test data for AAPL (single file per security_id)
    daily_ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "SEC001"
    daily_ticks_dir.mkdir(parents=True)
    daily_ticks = pl.DataFrame({
        "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
        "open": [185.0 + i * 0.5 for i in range(10)],
        "high": [186.0 + i * 0.5 for i in range(10)],
        "low": [184.0 + i * 0.5 for i in range(10)],
        "close": [185.5 + i * 0.5 for i in range(10)],
        "volume": [1000000 + i * 10000 for i in range(10)],
    })
    daily_ticks.write_parquet(daily_ticks_dir / "ticks.parquet")

    # Create fundamentals test data (keyed by security_id)
    fundamentals_dir = tmp_path / "data" / "raw" / "fundamental" / "SEC001"
    fundamentals_dir.mkdir(parents=True)
    fundamentals = pl.DataFrame({
        "symbol": ["AAPL"] * 4,
        "as_of_date": [
            date(2024, 3, 31),
            date(2024, 3, 31),
            date(2024, 6, 30),
            date(2024, 6, 30),
        ],
        "accn": ["0001-24-001", "0001-24-001", "0001-24-002", "0001-24-002"],
        "form": ["10-Q", "10-Q", "10-Q", "10-Q"],
        "concept": ["Revenue", "NetIncome", "Revenue", "NetIncome"],
        "value": [90000000000.0, 20000000000.0, 95000000000.0, 22000000000.0],
        "start": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 4, 1), date(2024, 4, 1)],
        "end": [date(2024, 3, 31), date(2024, 3, 31), date(2024, 6, 30), date(2024, 6, 30)],
        "frame": ["Q1", "Q1", "Q2", "Q2"],
        "is_instant": [False, False, False, False],
    })
    fundamentals.write_parquet(fundamentals_dir / "fundamental.parquet")

    # Create universe test data (top3000)
    universe_dir = tmp_path / "data" / "meta" / "universe"
    universe_dir.mkdir(parents=True)
    universe = pl.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "META"],
    })
    universe.write_parquet(universe_dir / "top3000.parquet")

    # Return tmp_path as the base, so paths like "data/meta/master/..." work
    yield tmp_path


@pytest.fixture
def client(test_data_dir: Path) -> Any:
    """Create client with local test data."""
    from alphalab.api import AlphaLabClient
    return AlphaLabClient(data_path=str(test_data_dir))
