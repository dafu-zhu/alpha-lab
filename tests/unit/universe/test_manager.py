"""
Unit tests for universe.manager module
Tests universe manager functionality for symbol management
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import pandas as pd
from pathlib import Path
from alphalab.universe.manager import UniverseManager


class TestUniverseManager:
    """Test UniverseManager class"""

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.setup_logger')
    def test_initialization_with_security_master(self, mock_logger, mock_ticks_class):
        """Test initialization with provided SecurityMaster"""
        mock_sm = Mock()

        manager = UniverseManager(security_master=mock_sm)

        # Verify provided security master was used (not a new one)
        assert manager.security_master == mock_sm

        # Verify Alpaca fetcher was created
        mock_ticks_class.assert_called_once()

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_empty_result(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test get_current_symbols with empty result"""
        # Mock empty DataFrame
        mock_fetch.return_value = pd.DataFrame()

        manager = UniverseManager()

        with pytest.raises(ValueError, match="Failed to fetch symbols"):
            manager.get_current_symbols()

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_none_result(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test get_current_symbols with None result"""
        mock_fetch.return_value = None

        manager = UniverseManager()

        with pytest.raises(ValueError, match="Failed to fetch symbols"):
            manager.get_current_symbols()

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_sec(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with year >= 2025 (SEC format)"""
        # Mock fetch_all_stocks
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="sec")

        # Verify SEC format (dots replaced with hyphens)
        assert symbols == ['AAPL', 'BRK-B', 'GOOGL']

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.get_hist_universe_local')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.get_hist_universe_local')
    def test_load_symbols_for_year_historical_sec(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with historical year (SEC format)"""
        # Mock historical universe
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'MSFT']})
        mock_get_hist.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="sec")

        # Verify SEC format (dots replaced with hyphens)
        assert symbols == ['AAPL', 'BRK-B', 'MSFT']

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    def test_load_symbols_for_year_invalid_type(self, mock_logger, mock_sm_class, mock_ticks_class):
        """Test load_symbols_for_year with invalid sym_type"""
        mock_logger.return_value = Mock()
        manager = UniverseManager()

        symbols = manager.load_symbols_for_year(year=2024, sym_type="invalid")
        assert symbols == []
        manager.logger.error.assert_called_once()

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_empty(self, mock_fetch, mock_logger, mock_sm_class, mock_ticks_class):
        """Empty current universe returns empty list."""
        mock_fetch.return_value = pl.DataFrame({'Ticker': []})
        mock_logger.return_value = Mock()

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="alpaca")

        assert symbols == []

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.get_hist_universe_local')
    def test_load_symbols_for_year_historical_empty(self, mock_get_hist, mock_logger, mock_sm_class, mock_ticks_class):
        """Empty historical universe returns empty list."""
        mock_get_hist.return_value = pl.DataFrame({'Ticker': []})
        mock_logger.return_value = Mock()

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="alpaca")

        assert symbols == []

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.get_hist_universe_local')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.setup_logger')
    def test_initialization_s3_only_security_master(self, mock_logger, mock_ticks_class):
        """Test initialization with S3-only SecurityMaster"""
        mock_sm = Mock()

        manager = UniverseManager(security_master=mock_sm)

        # Verify provided security master was used
        assert manager.security_master == mock_sm

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.get_hist_universe_local')
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

    @patch('alphalab.universe.manager.Ticks')
    @patch('alphalab.universe.manager.SecurityMaster')
    @patch('alphalab.universe.manager.setup_logger')
    @patch('alphalab.universe.manager.get_hist_universe_local')
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


class TestUniverseManagerLocalPaths:
    """Tests for local storage path handling in UniverseManager."""

    def test_get_local_storage_path_success(self):
        """Test _get_local_storage_path returns path from env."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': '/test/path'}):
            result = manager._get_local_storage_path()
            assert result == Path('/test/path')

    def test_get_local_storage_path_with_tilde(self):
        """Test _get_local_storage_path expands ~."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        import os

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': '~/data'}):
            result = manager._get_local_storage_path()
            assert str(result).startswith(os.path.expanduser('~'))

    def test_get_local_storage_path_missing_raises(self):
        """Test _get_local_storage_path raises when env var missing."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        with patch.dict('os.environ', {}, clear=True):
            # Remove LOCAL_STORAGE_PATH if it exists
            import os
            if 'LOCAL_STORAGE_PATH' in os.environ:
                del os.environ['LOCAL_STORAGE_PATH']
            with pytest.raises(ValueError, match="LOCAL_STORAGE_PATH"):
                manager._get_local_storage_path()


class TestUniverseManagerVerifyTicks:
    """Tests for _verify_ticks_exist method."""

    def test_verify_ticks_exist_returns_false_no_dir(self, tmp_path):
        """Test _verify_ticks_exist returns False when ticks dir doesn't exist."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._verify_ticks_exist()
            assert result is False

    def test_verify_ticks_exist_returns_false_empty_dir(self, tmp_path):
        """Test _verify_ticks_exist returns False when ticks dir is empty."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # Create empty ticks directory
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily"
        ticks_dir.mkdir(parents=True)

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._verify_ticks_exist()
            assert result is False

    def test_verify_ticks_exist_returns_false_no_parquet(self, tmp_path):
        """Test _verify_ticks_exist returns False when no parquet files."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # Create security dirs without parquet files
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily"
        for i in range(5):
            (ticks_dir / str(i)).mkdir(parents=True)

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._verify_ticks_exist()
            assert result is False  # Only 0 files found, need >= 3

    def test_verify_ticks_exist_returns_true(self, tmp_path):
        """Test _verify_ticks_exist returns True when parquet files exist."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # Create security dirs with parquet files
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily"
        for i in range(5):
            sec_dir = ticks_dir / str(i)
            sec_dir.mkdir(parents=True)
            (sec_dir / "ticks.parquet").touch()

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._verify_ticks_exist()
            assert result is True

    def test_verify_ticks_exist_handles_env_error(self):
        """Test _verify_ticks_exist returns False on environment error."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # Missing env var causes ValueError
        with patch.dict('os.environ', {}, clear=True):
            import os
            if 'LOCAL_STORAGE_PATH' in os.environ:
                del os.environ['LOCAL_STORAGE_PATH']
            result = manager._verify_ticks_exist()
            assert result is False


class TestUniverseManagerADVCalculation:
    """Tests for ADV calculation methods."""

    def test_calculate_adv_single_file_not_exists(self, tmp_path):
        """Test _calculate_adv_single returns 0 when file doesn't exist."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._calculate_adv_single((123, "AAPL", ["2024-01-01"]))
            assert result == (123, "AAPL", 0.0)

    def test_calculate_adv_single_with_data(self, tmp_path):
        """Test _calculate_adv_single calculates ADV correctly."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # Create parquet file with test data
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "123"
        ticks_dir.mkdir(parents=True)

        df = pl.DataFrame({
            "timestamp": ["2024-01-02", "2024-01-03"],
            "close": [100.0, 110.0],
            "volume": [1000, 2000],
        })
        df.write_parquet(ticks_dir / "ticks.parquet")

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._calculate_adv_single(
                (123, "AAPL", ["2024-01-02", "2024-01-03"])
            )
            assert result[0] == 123
            assert result[1] == "AAPL"
            # ADV = mean(close * volume) = mean(100*1000, 110*2000) = mean(100000, 220000) = 160000
            assert result[2] == 160000.0

    def test_calculate_adv_single_no_matching_dates(self, tmp_path):
        """Test _calculate_adv_single returns 0 when no dates match."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # Create parquet file with test data
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "123"
        ticks_dir.mkdir(parents=True)

        df = pl.DataFrame({
            "timestamp": ["2024-01-02"],
            "close": [100.0],
            "volume": [1000],
        })
        df.write_parquet(ticks_dir / "ticks.parquet")

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._calculate_adv_single(
                (123, "AAPL", ["2024-12-01"])  # Date not in file
            )
            assert result == (123, "AAPL", 0.0)

    def test_calculate_adv_single_handles_exception(self, tmp_path):
        """Test _calculate_adv_single handles exceptions gracefully."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()

        # Create invalid parquet file
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "123"
        ticks_dir.mkdir(parents=True)
        (ticks_dir / "ticks.parquet").write_text("invalid")

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            result = manager._calculate_adv_single(
                (123, "AAPL", ["2024-01-02"])
            )
            assert result == (123, "AAPL", 0.0)


class TestUniverseManagerTop3000LocalPath:
    """Tests for get_top_3000 with local ticks storage."""

    def test_get_top_3000_uses_local_when_available(self, tmp_path):
        """Test get_top_3000 uses local ticks when available."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.security_master = Mock()
        manager.alpaca_fetcher = Mock()

        # Setup security_master to return security IDs
        manager.security_master.get_security_id.side_effect = [1, 2, 3]

        # Create ticks files
        for sid in [1, 2, 3]:
            ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / str(sid)
            ticks_dir.mkdir(parents=True)
            df = pl.DataFrame({
                "timestamp": ["2024-06-28", "2024-06-29"],
                "close": [100.0 * sid, 100.0 * sid],
                "volume": [1000 * sid, 1000 * sid],
            })
            df.write_parquet(ticks_dir / "ticks.parquet")

        with patch.dict('os.environ', {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            with patch('alphalab.utils.calendar.TradingCalendar') as mock_cal:
                import datetime as dt
                mock_cal_instance = Mock()
                mock_cal_instance.get_trading_days.return_value = [
                    dt.date(2024, 6, 28), dt.date(2024, 6, 29)
                ]
                mock_cal.return_value = mock_cal_instance

                result = manager.get_top_3000("2024-06-30", ["AAA", "BBB", "CCC"])

                # Should have ranked results based on ADV
                assert len(result) == 3
                # Alpaca fetcher should NOT be called when local is available
                manager.alpaca_fetcher.recent_daily_ticks.assert_not_called()
