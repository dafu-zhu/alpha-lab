"""Tests for SecurityMaster."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quantdl.api.data.security_master import SecurityMaster
from quantdl.api.storage.backend import StorageBackend
from quantdl.api.storage.cache import DiskCache


@pytest.fixture
def storage(test_data_dir: Path) -> StorageBackend:
    """Create storage backend with local test data."""
    return StorageBackend(data_path=test_data_dir)


@pytest.fixture
def security_master(storage: StorageBackend, temp_cache_dir: str) -> SecurityMaster:
    """Create SecurityMaster with cache."""
    cache = DiskCache(cache_dir=temp_cache_dir)
    return SecurityMaster(storage, cache)


class TestResolve:
    """Symbol resolution tests."""

    def test_resolve_by_symbol(self, security_master: SecurityMaster) -> None:
        """Test resolving by symbol."""
        info = security_master.resolve("AAPL")
        assert info is not None
        assert info.symbol == "AAPL"
        assert info.security_id == "SEC001"
        assert info.company == "Apple Inc"

    def test_resolve_by_security_id(self, security_master: SecurityMaster) -> None:
        """Test resolving by security_id."""
        info = security_master.resolve("SEC002")
        assert info is not None
        assert info.symbol == "MSFT"

    def test_resolve_by_cik(self, security_master: SecurityMaster) -> None:
        """Test resolving by CIK."""
        info = security_master.resolve("0000789019")
        assert info is not None
        assert info.symbol == "MSFT"

    def test_resolve_missing(self, security_master: SecurityMaster) -> None:
        """Test resolving nonexistent symbol."""
        info = security_master.resolve("INVALID")
        assert info is None

    def test_resolve_point_in_time(self, security_master: SecurityMaster) -> None:
        """Test point-in-time resolution."""
        # GOOGL started 2004-08-19
        info_before = security_master.resolve("GOOGL", as_of=date(2004, 1, 1))
        assert info_before is None

        info_after = security_master.resolve("GOOGL", as_of=date(2005, 1, 1))
        assert info_after is not None
        assert info_after.symbol == "GOOGL"


class TestGetBySecurityId:
    """Tests for get_by_security_id method."""

    def test_get_by_security_id(self, security_master: SecurityMaster) -> None:
        """Test getting security by security_id."""
        info = security_master.get_by_security_id("SEC001")
        assert info is not None
        assert info.security_id == "SEC001"
        assert info.symbol == "AAPL"

    def test_get_by_security_id_not_found(self, security_master: SecurityMaster) -> None:
        """Test getting nonexistent security_id."""
        info = security_master.get_by_security_id("NONEXISTENT")
        assert info is None


class TestBatchResolve:
    """Batch resolution tests."""

    def test_resolve_batch(self, security_master: SecurityMaster) -> None:
        """Test batch symbol resolution."""
        result = security_master.resolve_batch(["AAPL", "MSFT", "INVALID"])

        assert "AAPL" in result
        assert "MSFT" in result
        assert "INVALID" in result

        assert result["AAPL"] is not None
        assert result["MSFT"] is not None
        assert result["INVALID"] is None


class TestSearch:
    """Search functionality tests."""

    def test_search_by_symbol(self, security_master: SecurityMaster) -> None:
        """Test searching by partial symbol."""
        results = security_master.search("AA")
        assert len(results) >= 1
        assert any(r.symbol == "AAPL" for r in results)

    def test_search_by_company(self, security_master: SecurityMaster) -> None:
        """Test searching by company name."""
        results = security_master.search("Microsoft")
        assert len(results) >= 1
        assert any(r.symbol == "MSFT" for r in results)

    def test_search_limit(self, security_master: SecurityMaster) -> None:
        """Test search result limit."""
        results = security_master.search("", limit=2)
        assert len(results) <= 2


class TestSymbolChanges:
    """Tests for symbol changes over time."""

    def test_symbol_change_history(
        self, test_data_dir: Path, temp_cache_dir: str
    ) -> None:
        """Test security with symbol change."""
        # Create security master with symbol change
        security_master_df = pl.DataFrame({
            "security_id": ["SEC100", "SEC100"],
            "symbol": ["OLD_SYM", "NEW_SYM"],
            "company": ["Test Company", "Test Company"],
            "cik": ["0001234567", "0001234567"],
            "cusip": ["123456789", "123456789"],
            "start_date": [date(2020, 1, 1), date(2023, 1, 1)],
            "end_date": [date(2022, 12, 31), None],
        })

        # Write to test directory
        sm_path = test_data_dir / "data" / "master" / "security_master.parquet"
        security_master_df.write_parquet(sm_path)

        # Create fresh security master
        storage = StorageBackend(data_path=test_data_dir)
        sm = SecurityMaster(storage, DiskCache(cache_dir=temp_cache_dir + "/new"))

        # Resolve at different points in time
        old_info = sm.resolve("OLD_SYM", as_of=date(2021, 6, 1))
        assert old_info is not None
        assert old_info.symbol == "OLD_SYM"

        new_info = sm.resolve("NEW_SYM", as_of=date(2024, 1, 1))
        assert new_info is not None
        assert new_info.symbol == "NEW_SYM"

        # Same security_id
        assert old_info.security_id == new_info.security_id
