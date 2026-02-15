"""
Unit tests for universe.manager module
Tests universe manager functionality for symbol management
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import pandas as pd
from pathlib import Path
from quantdl.universe.manager import UniverseManager


class TestUniverseManager:
    """Test UniverseManager class"""

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    def test_initialization_default(self, mock_logger, mock_sm_class, mock_ticks_class):
        """Test UniverseManager initialization with defaults"""
        mock_sm = Mock()
        mock_sm_class.return_value = mock_sm

        manager = UniverseManager()

        # Verify SecurityMaster was created
        mock_sm_class.assert_called_once()
        assert manager.security_master == mock_sm

        # Verify Alpaca fetcher was created
        mock_ticks_class.assert_called_once()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_initialization_with_security_master(self, mock_logger, mock_ticks_class):
        """Test initialization with provided SecurityMaster"""
        mock_sm = Mock()

        manager = UniverseManager(security_master=mock_sm)

        # Verify provided security master was used (not a new one)
        assert manager.security_master == mock_sm

        # Verify Alpaca fetcher was created
        mock_ticks_class.assert_called_once()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_no_cache(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test get_current_symbols without cache"""
        # Mock fetch_all_stocks
        mock_df = pd.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.get_current_symbols(refresh=False)

        # Verify fetch was called
        mock_fetch.assert_called_once_with(refresh=False, logger=manager.logger)

        # Verify result
        assert symbols == ['AAPL', 'MSFT', 'GOOGL']

        # Verify cache was set
        assert manager._current_symbols_cache == ['AAPL', 'MSFT', 'GOOGL']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_with_cache(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test get_current_symbols with cache"""
        manager = UniverseManager()

        # Set cache
        manager._current_symbols_cache = ['CACHED1', 'CACHED2']

        symbols = manager.get_current_symbols(refresh=False)

        # Verify fetch was NOT called
        mock_fetch.assert_not_called()

        # Verify cached result was returned
        assert symbols == ['CACHED1', 'CACHED2']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_refresh(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test get_current_symbols with refresh=True"""
        # Mock fetch_all_stocks
        mock_df = pd.DataFrame({'Ticker': ['NEW1', 'NEW2']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()

        # Set cache with old data
        manager._current_symbols_cache = ['OLD1', 'OLD2']

        symbols = manager.get_current_symbols(refresh=True)

        # Verify fetch was called with refresh=True
        mock_fetch.assert_called_once_with(refresh=True, logger=manager.logger)

        # Verify new data was returned and cached
        assert symbols == ['NEW1', 'NEW2']
        assert manager._current_symbols_cache == ['NEW1', 'NEW2']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_empty_result(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test get_current_symbols with empty result"""
        # Mock empty DataFrame
        mock_fetch.return_value = pd.DataFrame()

        manager = UniverseManager()

        with pytest.raises(ValueError, match="Failed to fetch symbols"):
            manager.get_current_symbols()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_none_result(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test get_current_symbols with None result"""
        mock_fetch.return_value = None

        manager = UniverseManager()

        with pytest.raises(ValueError, match="Failed to fetch symbols"):
            manager.get_current_symbols()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_alpaca(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with year >= 2025 (Alpaca format)"""
        # Mock fetch_all_stocks
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="alpaca")

        # Verify fetch was called
        mock_fetch.assert_called_once()

        # Verify Alpaca format (same as Nasdaq)
        assert symbols == ['AAPL', 'BRK.B', 'GOOGL']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_sec(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with year >= 2025 (SEC format)"""
        # Mock fetch_all_stocks
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="sec")

        # Verify SEC format (dots replaced with hyphens)
        assert symbols == ['AAPL', 'BRK-B', 'GOOGL']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_local')
    def test_load_symbols_for_year_historical_alpaca(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with historical year (Alpaca format)"""
        # Mock historical universe
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'MSFT']})
        mock_get_hist.return_value = mock_df

        mock_sm = Mock()
        mock_sm_class.return_value = mock_sm

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="alpaca")

        # Verify get_hist_universe_local was called with security_master
        mock_get_hist.assert_called_once_with(2024, security_master=mock_sm)

        # Verify Alpaca format
        assert symbols == ['AAPL', 'BRK.B', 'MSFT']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_local')
    def test_load_symbols_for_year_historical_sec(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with historical year (SEC format)"""
        # Mock historical universe
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'MSFT']})
        mock_get_hist.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="sec")

        # Verify SEC format (dots replaced with hyphens)
        assert symbols == ['AAPL', 'BRK-B', 'MSFT']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    def test_load_symbols_for_year_invalid_type(self, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with invalid sym_type"""
        mock_logger.return_value = Mock()
        manager = UniverseManager()

        symbols = manager.load_symbols_for_year(year=2024, sym_type="invalid")
        assert symbols == []
        manager.logger.error.assert_called_once()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    def test_store_dir_creation(self, mock_logger, mock_sm_class, mock_ticks_class):
        """Test that store directory is created"""
        manager = UniverseManager()

        assert manager.store_dir == Path("data/meta/universe")

    def test_get_top_3000_alpaca(self):
        """Alpaca path returns ranked symbols."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({
            "close": [5.0],
            "volume": [500.0]
        })
        manager.alpaca_fetcher.recent_daily_ticks.return_value = {"AAA": df}

        result = manager.get_top_3000("2024-06-30", ["AAA"])

        assert result == ["AAA"]
        manager.alpaca_fetcher.recent_daily_ticks.assert_called_once_with(
            ["AAA"], end_day="2024-06-30"
        )

    def test_get_top_3000_no_liquidity(self):
        """No symbols pass liquidity filter."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.alpaca_fetcher = Mock()

        manager.alpaca_fetcher.recent_daily_ticks.return_value = {}

        result = manager.get_top_3000("2024-06-30", ["AAA"])

        assert result == []
        manager.logger.error.assert_called_once()

    def test_get_top_3000_filters_threshold(self):
        """Filters out symbols below average dollar volume threshold."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.alpaca_fetcher = Mock()

        high_df = pl.DataFrame({"close": [50.0], "volume": [100.0]})
        low_df = pl.DataFrame({"close": [1.0], "volume": [10.0]})
        manager.alpaca_fetcher.recent_daily_ticks.return_value = {
            "HIGH": high_df,
            "LOW": low_df
        }

        result = manager.get_top_3000("2024-06-30", ["HIGH", "LOW"])

        assert result == ["HIGH"]


class TestUniverseManagerEdgeCases:
    """Test edge cases for UniverseManager"""

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_cache_invalidation_on_refresh(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test that cache is updated when refresh=True"""
        manager = UniverseManager()

        # First call with old data
        mock_df1 = pd.DataFrame({'Ticker': ['OLD1', 'OLD2']})
        mock_fetch.return_value = mock_df1
        symbols1 = manager.get_current_symbols(refresh=False)

        assert symbols1 == ['OLD1', 'OLD2']

        # Second call with refresh and new data
        mock_df2 = pd.DataFrame({'Ticker': ['NEW1', 'NEW2', 'NEW3']})
        mock_fetch.return_value = mock_df2
        symbols2 = manager.get_current_symbols(refresh=True)

        assert symbols2 == ['NEW1', 'NEW2', 'NEW3']
        assert manager._current_symbols_cache == ['NEW1', 'NEW2', 'NEW3']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_empty(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Empty current universe returns empty list."""
        mock_fetch.return_value = pl.DataFrame({'Ticker': []})
        mock_logger.return_value = Mock()

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="alpaca")

        assert symbols == []

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_local')
    def test_load_symbols_for_year_historical_empty(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Empty historical universe returns empty list."""
        mock_get_hist.return_value = pl.DataFrame({'Ticker': []})
        mock_logger.return_value = Mock()

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="alpaca")

        assert symbols == []

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_local')
    def test_load_symbols_for_year_exception(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Exceptions while loading symbols return empty list."""
        mock_get_hist.side_effect = Exception("boom")

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="alpaca")

        assert symbols == []

    def test_close_does_not_raise(self):
        """Test that close() completes without error."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # close() is now a no-op pass; should not raise
        manager.close()

    def test_get_top_3000_with_auto_resolve_true(self):
        """Test get_top_3000 with auto_resolve=True (unused but accepted)."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({
            "close": [10.0],
            "volume": [200.0]
        })
        manager.alpaca_fetcher.recent_daily_ticks.return_value = {"AAA": df}

        result = manager.get_top_3000("2024-06-30", ["AAA"], auto_resolve=True)

        # Verify alpaca_fetcher was called (auto_resolve is not passed through)
        manager.alpaca_fetcher.recent_daily_ticks.assert_called_once_with(
            ["AAA"],
            end_day="2024-06-30"
        )
        assert result == ["AAA"]

    def test_get_top_3000_with_auto_resolve_false(self):
        """Test get_top_3000 with auto_resolve=False (unused but accepted)."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({
            "close": [10.0],
            "volume": [200.0]
        })
        manager.alpaca_fetcher.recent_daily_ticks.return_value = {"AAA": df}

        result = manager.get_top_3000("2024-06-30", ["AAA"], auto_resolve=False)

        # Verify alpaca_fetcher was called (auto_resolve is not passed through)
        manager.alpaca_fetcher.recent_daily_ticks.assert_called_once_with(
            ["AAA"],
            end_day="2024-06-30"
        )
        assert result == ["AAA"]

    def test_get_top_3000_ranks_by_liquidity(self):
        """Test that get_top_3000 correctly ranks symbols by liquidity."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.alpaca_fetcher = Mock()

        # Symbol with high liquidity
        high_df = pl.DataFrame({
            "close": [100.0, 100.0],
            "volume": [1000.0, 1000.0]
        })
        # Symbol with medium liquidity
        mid_df = pl.DataFrame({
            "close": [50.0, 50.0],
            "volume": [500.0, 500.0]
        })
        # Symbol with low liquidity (but above threshold)
        low_df = pl.DataFrame({
            "close": [10.0, 10.0],
            "volume": [200.0, 200.0]
        })

        manager.alpaca_fetcher.recent_daily_ticks.return_value = {
            "MID": mid_df,
            "HIGH": high_df,
            "LOW": low_df
        }

        result = manager.get_top_3000("2024-06-30", ["MID", "HIGH", "LOW"])

        # Verify symbols are ranked by average dollar volume
        assert result == ["HIGH", "MID", "LOW"]

    def test_get_top_3000_empty_dataframe_excluded(self):
        """Test that symbols with empty dataframes are excluded."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.alpaca_fetcher = Mock()

        good_df = pl.DataFrame({
            "close": [10.0],
            "volume": [200.0]
        })
        empty_df = pl.DataFrame({
            "close": [],
            "volume": []
        })

        manager.alpaca_fetcher.recent_daily_ticks.return_value = {
            "GOOD": good_df,
            "EMPTY": empty_df
        }

        result = manager.get_top_3000("2024-06-30", ["GOOD", "EMPTY"])

        # Only GOOD should be in the result
        assert result == ["GOOD"]

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_initialization_s3_only_security_master(self, mock_logger, mock_ticks_class):
        """Test initialization with S3-only SecurityMaster"""
        mock_sm = Mock()

        manager = UniverseManager(security_master=mock_sm)

        # Verify provided security master was used
        assert manager.security_master == mock_sm

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_local')
    def test_load_symbols_historical_cache_hit(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year returns cached result for historical years"""
        manager = UniverseManager()

        # Pre-populate cache with tuple key (year, sym_type)
        manager._historical_cache[(2023, "alpaca")] = ["CACHED1", "CACHED2", "CACHED3"]

        # Call load_symbols_for_year
        symbols = manager.load_symbols_for_year(year=2023, sym_type="alpaca")

        # Verify cache was used and historical query was NOT made
        assert symbols == ["CACHED1", "CACHED2", "CACHED3"]
        mock_get_hist.assert_not_called()
        manager.logger.debug.assert_any_call(
            "Loaded 3 symbols for 2023 from cache (format=alpaca)"
        )

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.SecurityMaster')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_local')
    def test_load_symbols_historical_uses_security_master(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year passes security_master to get_hist_universe_local"""
        mock_df = pl.DataFrame({'Ticker': ['AAPL']})
        mock_get_hist.return_value = mock_df

        mock_sm = Mock()
        mock_sm_class.return_value = mock_sm

        manager = UniverseManager()
        manager.load_symbols_for_year(year=2020, sym_type="alpaca")

        # Verify security_master was passed to get_hist_universe_local
        mock_get_hist.assert_called_once_with(2020, security_master=mock_sm)
