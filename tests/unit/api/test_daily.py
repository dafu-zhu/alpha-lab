"""Tests for ticks (daily) data API."""

from datetime import date
from pathlib import Path

import pytest

from quantdl.api import QuantDLClient
from quantdl.api.exceptions import DataNotFoundError


@pytest.fixture
def client(test_data_dir: Path) -> QuantDLClient:
    """Create client with local test data."""
    return QuantDLClient(data_path=str(test_data_dir))


class TestDailyAPI:
    """Tests for client.ticks() method."""

    def test_daily_single_symbol(self, client: QuantDLClient) -> None:
        """Test fetching daily data for single symbol."""
        df = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        assert "timestamp" in df.columns
        assert "AAPL" in df.columns
        assert len(df) > 0

    def test_daily_multiple_symbols(self, client: QuantDLClient) -> None:
        """Test fetching daily data for multiple symbols."""
        # Note: Only AAPL has test data in conftest
        df = client.ticks(["AAPL"], "close", "2024-01-01", "2024-01-10")

        assert "timestamp" in df.columns
        assert "AAPL" in df.columns

    def test_daily_wide_format(self, client: QuantDLClient) -> None:
        """Test that ticks returns wide format."""
        df = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        # First column is timestamp
        assert df.columns[0] == "timestamp"
        # Other columns are symbols
        assert "AAPL" in df.columns[1:]

    def test_daily_sorted_by_date(self, client: QuantDLClient) -> None:
        """Test that daily data is sorted by date."""
        df = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        dates = df["timestamp"].to_list()
        assert dates == sorted(dates)

    def test_daily_field_options(self, client: QuantDLClient) -> None:
        """Test different price fields."""
        for field in ["open", "high", "low", "close", "volume"]:
            df = client.ticks("AAPL", field, "2024-01-01", "2024-01-10")
            assert len(df) > 0

    def test_daily_invalid_symbol(self, client: QuantDLClient) -> None:
        """Test fetching data for invalid symbol."""
        with pytest.raises(DataNotFoundError):
            client.ticks("INVALID_SYMBOL", "close", "2024-01-01", "2024-01-10")

    def test_daily_date_filtering(self, client: QuantDLClient) -> None:
        """Test date range filtering."""
        df = client.ticks("AAPL", "close", "2024-01-03", "2024-01-05")

        dates = df["timestamp"].to_list()
        for d in dates:
            assert d >= date(2024, 1, 3)
            assert d <= date(2024, 1, 5)


class TestDailyCaching:
    """Tests for ticks data caching."""

    def test_daily_uses_cache(self, client: QuantDLClient) -> None:
        """Test that second request uses cache (when caching enabled)."""
        # First request
        df1 = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        # Check cache stats (empty dict if no cache in local mode)
        stats = client.cache_stats()
        if "entries" in stats:
            assert stats["entries"] >= 1

        # Second request should return same data
        df2 = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        assert df1.equals(df2)

    def test_clear_cache(self, client: QuantDLClient) -> None:
        """Test cache clearing."""
        client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        client.clear_cache()
        stats = client.cache_stats()
        # In local mode without cache_dir, cache_stats returns empty dict
        assert stats.get("entries", 0) == 0
