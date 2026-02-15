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

        assert "Date" in df.columns
        assert "AAPL" in df.columns
        assert len(df) > 0

    def test_daily_multiple_symbols(self, client: QuantDLClient) -> None:
        """Test fetching daily data for multiple symbols."""
        # Note: Only AAPL has test data in conftest
        df = client.ticks(["AAPL"], "close", "2024-01-01", "2024-01-10")

        assert "Date" in df.columns
        assert "AAPL" in df.columns

    def test_daily_wide_format(self, client: QuantDLClient) -> None:
        """Test that ticks returns wide format."""
        df = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        # First column is timestamp
        assert df.columns[0] == "Date"
        # Other columns are symbols
        assert "AAPL" in df.columns[1:]

    def test_daily_sorted_by_date(self, client: QuantDLClient) -> None:
        """Test that daily data is sorted by date."""
        df = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")

        dates = df["Date"].to_list()
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

        dates = df["Date"].to_list()
        for d in dates:
            assert d >= date(2024, 1, 3)
            assert d <= date(2024, 1, 5)

    def test_daily_repeated_request_returns_same_data(self, client: QuantDLClient) -> None:
        """Test that repeated requests return identical data."""
        df1 = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")
        df2 = client.ticks("AAPL", "close", "2024-01-01", "2024-01-10")
        assert df1.equals(df2)
