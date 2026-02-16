"""
Unit tests for universe.historical module
Tests historical universe retrieval functionality
"""
import pytest
from unittest.mock import Mock, patch
import polars as pl
from datetime import date


class TestGetHistUniverseLocal:
    """Test get_hist_universe_local function"""

    def test_with_security_master_instance(self):
        """Test using security_master kwarg."""
        from alphalab.universe.historical import get_hist_universe_local

        mock_sm = Mock()
        mock_sm.master_tb = pl.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "company": ["Apple", "Microsoft"],
            "start_date": [date(2020, 1, 1), date(2020, 1, 1)],
            "end_date": [date(2025, 12, 31), date(2025, 12, 31)],
        })

        result = get_hist_universe_local(2024, security_master=mock_sm)
        assert len(result) == 2
        assert "Ticker" in result.columns
        assert "Name" in result.columns

    def test_filters_by_year(self):
        """Test that securities are filtered by year."""
        from alphalab.universe.historical import get_hist_universe_local

        mock_sm = Mock()
        mock_sm.master_tb = pl.DataFrame({
            "symbol": ["AAPL", "MSFT", "OLDCO"],
            "company": ["Apple", "Microsoft", "Old Company"],
            "start_date": [date(2020, 1, 1), date(2020, 1, 1), date(2010, 1, 1)],
            "end_date": [date(2025, 12, 31), date(2025, 12, 31), date(2015, 12, 31)],
        })

        result = get_hist_universe_local(2024, security_master=mock_sm)
        assert len(result) == 2  # OLDCO excluded (ended 2015)

        result_2012 = get_hist_universe_local(2012, security_master=mock_sm)
        assert len(result_2012) == 1  # Only OLDCO active in 2012... wait, AAPL/MSFT started 2020
        # Actually OLDCO is active 2010-2015, so 2012 should return OLDCO
        tickers = result_2012["Ticker"].to_list()
        assert "OLDCO" in tickers

    def test_deduplicates_tickers(self):
        """Test that duplicate tickers are removed."""
        from alphalab.universe.historical import get_hist_universe_local

        mock_sm = Mock()
        mock_sm.master_tb = pl.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "company": ["Apple Inc", "Apple Inc"],
            "start_date": [date(2020, 1, 1), date(2022, 1, 1)],
            "end_date": [date(2021, 12, 31), date(2025, 12, 31)],
        })

        result = get_hist_universe_local(2024, security_master=mock_sm)
        assert len(result) == 1

    def test_no_master_file_raises(self, tmp_path, monkeypatch):
        """Test error when no security master exists."""
        from alphalab.universe import historical
        monkeypatch.setattr(historical, "LOCAL_MASTER_PATH", tmp_path / "nonexistent.parquet")

        with pytest.raises(FileNotFoundError, match="No security master found"):
            historical.get_hist_universe_local(2024)
