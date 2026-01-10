"""
Unit tests for utils.mapping module
Tests symbol-CIK mapping and calendar alignment functionality
"""
import pytest
import datetime as dt
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import polars as pl
from quantdl.utils.mapping import symbol_cik_mapping, align_calendar


class TestSymbolCikMapping:
    """Test symbol_cik_mapping function"""

    @patch('quantdl.utils.mapping.requests.get')
    def test_successful_mapping_retrieval(self, mock_get):
        """Test successful retrieval of symbol-CIK mapping"""
        # Mock SEC response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."},
            "1": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft Corp"},
            "2": {"ticker": "GOOGL", "cik_str": 1652044, "title": "Alphabet Inc."}
        }
        mock_get.return_value = mock_response

        result = symbol_cik_mapping()

        # Verify request was made with proper headers
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://www.sec.gov/files/company_tickers.json"
        assert 'User-Agent' in call_args[1]['headers']

        # Verify mapping
        assert result == {
            "AAPL": 320193,
            "MSFT": 789019,
            "GOOGL": 1652044
        }

    @patch('quantdl.utils.mapping.requests.get')
    def test_failed_request(self, mock_get):
        """Test handling of failed HTTP request"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = symbol_cik_mapping()

        # Should return empty dict on failure
        assert result == {}

    @patch('quantdl.utils.mapping.requests.get')
    def test_missing_ticker_field(self, mock_get):
        """Test handling of entries with missing ticker field"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193},
            "1": {"cik_str": 789019, "title": "No Ticker Corp"},  # Missing ticker
            "2": {"ticker": "GOOGL", "cik_str": 1652044}
        }
        mock_get.return_value = mock_response

        result = symbol_cik_mapping()

        # Should skip entry with missing ticker
        assert result == {
            "AAPL": 320193,
            "GOOGL": 1652044
        }

    @patch('quantdl.utils.mapping.requests.get')
    def test_empty_response(self, mock_get):
        """Test handling of empty response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        result = symbol_cik_mapping()

        assert result == {}


class TestAlignCalendar:
    """Test align_calendar function"""

    @pytest.fixture
    def mock_calendar(self, tmp_path):
        """Create mock trading calendar"""
        dates = [
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 3),
            dt.date(2024, 1, 4),
            dt.date(2024, 1, 5),
        ]
        df = pl.DataFrame({"timestamp": dates})
        calendar_path = tmp_path / "calendar.parquet"
        df.write_parquet(calendar_path)
        return calendar_path

    def test_align_with_datetime_format(self, mock_calendar):
        """Test alignment with datetime format timestamps"""
        data = [
            {
                "timestamp": "2024-01-02T00:00:00",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000,
                "symbol": "AAPL"
            },
            {
                "timestamp": "2024-01-03T00:00:00",
                "open": 103.0,
                "high": 106.0,
                "low": 102.0,
                "close": 105.0,
                "volume": 1100000,
                "symbol": "AAPL"
            }
        ]

        result = align_calendar(
            data=data,
            start_date=dt.date(2024, 1, 1),
            end_date=dt.date(2024, 1, 5),
            calendar_path=mock_calendar
        )

        assert len(result) == 4  # All trading days
        assert result[0]["timestamp"] == "2024-01-02"
        assert result[0]["close"] == 103.0
        assert result[1]["timestamp"] == "2024-01-03"
        assert result[1]["close"] == 105.0

    def test_align_with_date_format(self, mock_calendar):
        """Test alignment with date format timestamps"""
        data = [
            {
                "timestamp": "2024-01-02",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000
            }
        ]

        result = align_calendar(
            data=data,
            start_date=dt.date(2024, 1, 2),
            end_date=dt.date(2024, 1, 2),
            calendar_path=mock_calendar
        )

        assert len(result) == 1
        assert result[0]["timestamp"] == "2024-01-02"
        assert result[0]["open"] == 100.0

    def test_align_drops_optional_columns(self, mock_calendar):
        """Test that optional columns are dropped"""
        data = [
            {
                "timestamp": "2024-01-02",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000,
                "num_trades": 5000,  # Optional column
                "vwap": 102.5        # Optional column
            }
        ]

        result = align_calendar(
            data=data,
            start_date=dt.date(2024, 1, 2),
            end_date=dt.date(2024, 1, 2),
            calendar_path=mock_calendar
        )

        assert len(result) == 1
        # Verify optional columns are not present
        assert "num_trades" not in result[0]
        assert "vwap" not in result[0]
        # Verify required columns are present
        assert "open" in result[0]
        assert "close" in result[0]

    def test_align_without_optional_columns(self, mock_calendar):
        """Test alignment when optional columns are not present"""
        data = [
            {
                "timestamp": "2024-01-02",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000
            }
        ]

        result = align_calendar(
            data=data,
            start_date=dt.date(2024, 1, 2),
            end_date=dt.date(2024, 1, 2),
            calendar_path=mock_calendar
        )

        assert len(result) == 1
        assert result[0]["open"] == 100.0

    def test_align_fills_missing_dates(self, mock_calendar):
        """Test that missing trading days are filled with nulls"""
        data = [
            {
                "timestamp": "2024-01-02",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000
            }
            # Missing 2024-01-03, 2024-01-04
        ]

        result = align_calendar(
            data=data,
            start_date=dt.date(2024, 1, 2),
            end_date=dt.date(2024, 1, 5),
            calendar_path=mock_calendar
        )

        # Should have all trading days, with nulls for missing ones
        assert len(result) == 4
        assert result[0]["timestamp"] == "2024-01-02"
        assert result[0]["close"] == 103.0
        assert result[1]["timestamp"] == "2024-01-03"
        assert result[1]["close"] is None  # Missing data

    def test_align_type_casting(self, mock_calendar):
        """Test that data types are properly cast"""
        data = [
            {
                "timestamp": "2024-01-02",
                "open": "100.0",  # String
                "high": "105.0",
                "low": "99.0",
                "close": "103.0",
                "volume": "1000000"  # String
            }
        ]

        result = align_calendar(
            data=data,
            start_date=dt.date(2024, 1, 2),
            end_date=dt.date(2024, 1, 2),
            calendar_path=mock_calendar
        )

        assert len(result) == 1
        # Verify types are cast correctly
        assert isinstance(result[0]["open"], float)
        assert isinstance(result[0]["volume"], int)

    def test_align_empty_data(self, mock_calendar):
        """Test alignment with empty data"""
        data = []

        result = align_calendar(
            data=data,
            start_date=dt.date(2024, 1, 2),
            end_date=dt.date(2024, 1, 5),
            calendar_path=mock_calendar
        )

        # Should still return calendar days with null data
        assert len(result) == 4
        assert result[0]["timestamp"] == "2024-01-02"
        assert result[0]["close"] is None
