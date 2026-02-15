"""Tests for CalendarMaster."""

from datetime import date
from pathlib import Path

import pytest

from quantdl.api.data.calendar_master import CalendarMaster
from quantdl.api.backend import StorageBackend


@pytest.fixture
def storage(test_data_dir: Path) -> StorageBackend:
    """Create storage backend with local test data."""
    return StorageBackend(data_path=test_data_dir)


@pytest.fixture
def calendar_master(storage: StorageBackend) -> CalendarMaster:
    """Create CalendarMaster."""
    return CalendarMaster(storage)


class TestIsTradingDay:
    """Tests for is_trading_day method."""

    def test_is_trading_day_true(self, calendar_master: CalendarMaster) -> None:
        """Test is_trading_day returns True for trading day."""
        # Jan 2, 2024 is in the test calendar
        result = calendar_master.is_trading_day(date(2024, 1, 2))
        assert result is True

    def test_is_trading_day_false(self, calendar_master: CalendarMaster) -> None:
        """Test is_trading_day returns False for non-trading day."""
        # Jan 1, 2024 (New Year's Day) is not in the test calendar
        result = calendar_master.is_trading_day(date(2024, 1, 1))
        assert result is False

    def test_is_trading_day_weekend(self, calendar_master: CalendarMaster) -> None:
        """Test is_trading_day returns False for weekend."""
        # Jan 6, 2024 is a Saturday
        result = calendar_master.is_trading_day(date(2024, 1, 6))
        assert result is False


class TestGetTradingDays:
    """Tests for get_trading_days method."""

    def test_get_trading_days_range(self, calendar_master: CalendarMaster) -> None:
        """Test getting trading days in range."""
        result = calendar_master.get_trading_days(date(2024, 1, 1), date(2024, 1, 10))
        assert isinstance(result, list)
        assert len(result) > 0
        # Should be sorted
        assert result == sorted(result)
        # All dates should be within range
        for d in result:
            assert date(2024, 1, 1) <= d <= date(2024, 1, 10)

    def test_get_trading_days_empty_range(self, calendar_master: CalendarMaster) -> None:
        """Test getting trading days for range with no trading days."""
        # Very old date range not in test data
        result = calendar_master.get_trading_days(date(1990, 1, 1), date(1990, 1, 10))
        assert result == []

    def test_get_trading_days_single_day(self, calendar_master: CalendarMaster) -> None:
        """Test getting trading days for single day."""
        result = calendar_master.get_trading_days(date(2024, 1, 2), date(2024, 1, 2))
        assert len(result) == 1
        assert result[0] == date(2024, 1, 2)


class TestCaching:
    """Tests for in-memory caching behavior."""

    def test_calendar_caching(self, storage: StorageBackend) -> None:
        """Test that calendar data is cached in memory."""
        cm = CalendarMaster(storage)

        # First call loads data
        cm.is_trading_day(date(2024, 1, 2))

        # Internal cache should be populated
        assert cm._df is not None
        assert cm._trading_days is not None

        # Subsequent calls should use cached data
        result = cm.is_trading_day(date(2024, 1, 3))
        assert result is True
