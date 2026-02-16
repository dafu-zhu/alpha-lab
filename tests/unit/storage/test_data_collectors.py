"""
Unit tests for storage.data_collectors module
Tests data collection functionality with new DataCollector architecture
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict
import threading
import polars as pl
import datetime as dt
import logging


class TestTicksDataCollector:
    """Test TicksDataCollector class (new architecture)"""

    def test_initialization(self):
        """Test TicksDataCollector initialization with dependency injection"""
        from alphalab.storage.pipeline import TicksDataCollector

        mock_alpaca = Mock()
        mock_headers = {'Authorization': 'Bearer token'}
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers=mock_headers,
            logger=mock_logger
        )

        assert collector.alpaca_ticks == mock_alpaca
        assert collector.alpaca_headers == mock_headers
        assert collector.logger == mock_logger

    def test_collect_daily_ticks_year_uses_alpaca(self):
        """Test collecting daily ticks always uses Alpaca"""
        from alphalab.storage.pipeline import TicksDataCollector
        from alphalab.collection.models import TickDataPoint

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.return_value = {
            "AAPL": [{
                "t": "2024-01-02T05:00:00Z",
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 1000000,
                "n": 5000,
                "vw": 100.5
            }]
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2024-01-02T00:00:00",
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
                num_trades=5000,
                vwap=100.5
            )
        ]
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year('AAPL', 2024)

        mock_alpaca.fetch_daily_year_bulk.assert_called_once_with(
            symbols=['AAPL'],
            year=2024,
            adjusted=True
        )
        assert len(result) == 1
        assert result["close"][0] == 101.0

    def test_collect_daily_ticks_year_alpaca(self):
        """Test collecting daily ticks for year 2025 (uses Alpaca)"""
        from alphalab.storage.pipeline import TicksDataCollector
        from alphalab.collection.models import TickDataPoint

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.return_value = {
            "AAPL": [{
                "t": "2025-01-01T05:00:00Z",
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 1000000,
                "n": 5000,
                "vw": 100.5
            }]
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-01-01T00:00:00",
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
                num_trades=5000,
                vwap=100.5
            )
        ]
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year('AAPL', 2025)

        mock_alpaca.fetch_daily_year_bulk.assert_called_once_with(
            symbols=['AAPL'],
            year=2025,
            adjusted=True
        )
        assert len(result) == 1
        assert result["close"][0] == 101.0

    def test_collect_daily_ticks_year_alpaca_failure(self):
        """Alpaca path returns empty on exceptions."""
        from alphalab.storage.pipeline import TicksDataCollector

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.side_effect = Exception("boom")
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year("AAPL", 2025)

        assert result.is_empty()

    def test_collect_daily_ticks_year_bulk_alpaca(self):
        """Bulk Alpaca year fetch returns normalized DataFrames."""
        from alphalab.storage.pipeline import TicksDataCollector
        from alphalab.collection.models import TickDataPoint

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.return_value = {
            "AAPL": [{
                "t": "2025-01-02T05:00:00Z",
                "o": 150.0,
                "h": 155.0,
                "l": 149.0,
                "c": 154.0,
                "v": 1000,
                "n": 5,
                "vw": 153.0
            }],
            "MSFT": []
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-01-02T00:00:00",
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000,
                num_trades=5,
                vwap=153.0
            )
        ]
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year_bulk(["AAPL", "MSFT"], 2025)

        mock_alpaca.fetch_daily_year_bulk.assert_called_once_with(["AAPL", "MSFT"], 2025, adjusted=True)
        assert result["AAPL"]["close"][0] == 154.0
        assert result["MSFT"].is_empty()

    def test_collect_daily_ticks_month_filters_correctly(self):
        """Test that collect_daily_ticks_month calls month-specific Alpaca API"""
        from alphalab.storage.pipeline import TicksDataCollector
        from alphalab.collection.models import TickDataPoint

        mock_alpaca = Mock()

        # Mock Alpaca's get_daily() to return June data only
        june_bars = [
            {
                "t": "2025-06-30T04:00:00Z",
                "o": 103.0,
                "h": 108.0,
                "l": 102.0,
                "c": 107.0,
                "v": 1300000,
                "n": 1000,
                "vw": 105.5
            }
        ]
        mock_alpaca.fetch_daily_month_bulk.return_value = {"AAPL": june_bars}
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-06-30T00:00:00",
                open=103.0,
                high=108.0,
                low=102.0,
                close=107.0,
                volume=1300000,
                num_trades=1000,
                vwap=105.5
            )
        ]

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        # Test June (month=6) for year 2025 (uses Alpaca)
        result = collector.collect_daily_ticks_month("AAPL", 2025, 6)

        # Verify bulk month fetch
        mock_alpaca.fetch_daily_month_bulk.assert_called_once_with(
            symbols=["AAPL"],
            year=2025,
            month=6,
            adjusted=True
        )
        mock_alpaca.parse_ticks.assert_called_once_with(june_bars)

        assert len(result) == 1
        assert result["timestamp"][0] == "2025-06-30"
        assert result["close"][0] == 107.0

    def test_collect_daily_ticks_month_uses_year_df(self):
        """Test that collect_daily_ticks_month filters from provided year_df."""
        from alphalab.storage.pipeline import TicksDataCollector

        mock_alpaca = Mock()

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        year_df = pl.DataFrame({
            "timestamp": ["2024-06-30", "2024-07-01"],
            "open": [190.0, 191.0],
            "high": [195.0, 196.0],
            "low": [189.0, 190.0],
            "close": [193.0, 194.0],
            "volume": [50000000, 51000000]
        })

        result = collector.collect_daily_ticks_month("AAPL", 2024, 6, year_df=year_df)

        assert len(result) == 1
        assert result["timestamp"][0] == "2024-06-30"
        mock_alpaca.fetch_daily_month_bulk.assert_not_called()

    def test_collect_daily_ticks_month_year_df_empty(self):
        """Year DF empty returns empty."""
        from alphalab.storage.pipeline import TicksDataCollector

        collector = TicksDataCollector(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        result = collector.collect_daily_ticks_month("AAPL", 2024, 6, year_df=pl.DataFrame())

        assert result.is_empty()

    def test_collect_daily_ticks_month_year_df_filtered_empty(self):
        """Year DF with other months returns empty."""
        from alphalab.storage.pipeline import TicksDataCollector

        collector = TicksDataCollector(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        year_df = pl.DataFrame({
            "timestamp": ["2024-07-01"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })

        result = collector.collect_daily_ticks_month("AAPL", 2024, 6, year_df=year_df)

        assert result.is_empty()

    def test_collect_daily_ticks_month_empty_when_no_data(self):
        """Test that collect_daily_ticks_month returns empty when month has no data (Alpaca)"""
        from alphalab.storage.pipeline import TicksDataCollector

        mock_alpaca = Mock()

        # Mock Alpaca's get_daily() to return empty data for requested month
        mock_alpaca.fetch_daily_month_bulk.return_value = {"AAPL": []}

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        # Test June (month=6) for year 2025 which has no data
        result = collector.collect_daily_ticks_month("AAPL", 2025, 6)

        # Verify bulk month fetch
        mock_alpaca.fetch_daily_month_bulk.assert_called_once_with(
            symbols=["AAPL"],
            year=2025,
            month=6,
            adjusted=True
        )

        assert result.is_empty()

    def test_collect_daily_ticks_month_bulk_alpaca(self):
        """Bulk month fetch returns normalized DataFrames via Alpaca."""
        from alphalab.storage.pipeline import TicksDataCollector
        from alphalab.collection.models import TickDataPoint

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_month_bulk.return_value = {
            "AAPL": [{
                "t": "2025-06-30T05:00:00Z",
                "o": 200.0,
                "h": 210.0,
                "l": 195.0,
                "c": 205.0,
                "v": 2000,
                "n": 10,
                "vw": 204.0
            }],
            "MSFT": []
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-06-30T00:00:00",
                open=200.0,
                high=210.0,
                low=195.0,
                close=205.0,
                volume=2000,
                num_trades=10,
                vwap=204.0
            )
        ]

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        result = collector.collect_daily_ticks_month_bulk(["AAPL", "MSFT"], 2025, 6, sleep_time=0.0)

        mock_alpaca.fetch_daily_month_bulk.assert_called_once_with(
            symbols=["AAPL", "MSFT"],
            year=2025,
            month=6,
            sleep_time=0.0,
            adjusted=True
        )
        assert result["AAPL"]["close"][0] == 205.0
        assert result["MSFT"].is_empty()

    def test_collect_daily_ticks_month_alpaca_exception(self):
        """Alpaca path returns empty and logs warning on exception."""
        from alphalab.storage.pipeline import TicksDataCollector

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_month_bulk.side_effect = RuntimeError("boom")
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_month("AAPL", 2025, 6)

        assert result.is_empty()
        mock_logger.warning.assert_called()

    def test_collect_daily_ticks_range_bulk(self):
        """Test range-based bulk fetch delegates to alpaca fetch_daily_range_bulk."""
        from alphalab.storage.pipeline import TicksDataCollector
        from alphalab.collection.models import TickDataPoint

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_range_bulk.return_value = {
            "AAPL": [{
                "t": "2020-06-15T05:00:00Z",
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 1000000,
                "n": 5000,
                "vw": 100.5
            }],
            "DISH": []
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2020-06-15T00:00:00",
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
                num_trades=5000,
                vwap=100.5
            )
        ]
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_range_bulk(
            ["AAPL", "DISH"], "2017-01-01", "2025-12-31"
        )

        mock_alpaca.fetch_daily_range_bulk.assert_called_once_with(
            ["AAPL", "DISH"],
            "2017-01-01T00:00:00Z",
            "2025-12-31T23:59:59Z",
            adjusted=True
        )
        assert result["AAPL"]["close"][0] == 101.0
        assert result["DISH"].is_empty()

    def test_normalize_daily_df_adds_missing_columns(self):
        """Missing columns are added and types normalized."""
        from alphalab.storage.pipeline import TicksDataCollector

        collector = TicksDataCollector(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        df = pl.DataFrame({
            "timestamp": ["2024-01-02"],
            "close": [101.123456]
        })

        result = collector._normalize_daily_df(df)

        for col in ["open", "high", "low", "volume"]:
            assert col in result.columns
        assert result["close"][0] == 101.1235


class TestFundamentalDataCollector:
    """Test FundamentalDataCollector class (refactored)"""

    def test_initialization_with_logger(self):
        """Test FundamentalDataCollector initialization with logger"""
        from alphalab.storage.pipeline import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        assert collector.logger == mock_logger
        assert isinstance(collector._fundamental_cache, OrderedDict)
        assert isinstance(collector._fundamental_cache_lock, type(threading.Lock()))

    @patch('alphalab.storage.pipeline.collectors.setup_logger')
    def test_initialization_without_logger(self, mock_setup_logger):
        """Test FundamentalDataCollector creates logger if not provided"""
        from alphalab.storage.pipeline import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        mock_setup_logger.return_value = mock_logger

        collector = FundamentalDataCollector()

        mock_setup_logger.assert_called_once()
        assert collector.logger == mock_logger

    def test_shared_cache_initialization(self):
        """Test FundamentalDataCollector uses shared cache when provided"""
        from alphalab.storage.pipeline import FundamentalDataCollector

        shared_cache = OrderedDict()
        shared_lock = threading.Lock()
        mock_logger = Mock(spec=logging.Logger)

        collector = FundamentalDataCollector(
            logger=mock_logger,
            fundamental_cache=shared_cache,
            fundamental_cache_lock=shared_lock
        )

        # Should use the shared cache
        assert collector._fundamental_cache is shared_cache
        assert collector._fundamental_cache_lock is shared_lock

    def test_load_concepts_with_provided_list(self):
        """Test _load_concepts returns provided list"""
        from alphalab.storage.pipeline import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        concepts = ['Revenue', 'Assets', 'NetIncome']
        result = collector._load_concepts(concepts=concepts)

        assert result == concepts

    @patch('builtins.open', create=True)
    @patch('yaml.safe_load')
    def test_load_concepts_from_config(self, mock_yaml_load, mock_open):
        """Test _load_concepts loads from config file"""
        from alphalab.storage.pipeline import FundamentalDataCollector
        from pathlib import Path

        mock_yaml_load.return_value = {'Revenue': 'mapping1', 'Assets': 'mapping2'}
        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        result = collector._load_concepts()

        assert 'Revenue' in result
        assert 'Assets' in result

    def test_get_or_create_fundamental_cache(self):
        """Cache hit returns same object; LRU eviction occurs."""
        from alphalab.storage.pipeline import collectors as dc

        mock_logger = Mock(spec=logging.Logger)
        collector = dc.FundamentalDataCollector(logger=mock_logger, fundamental_cache_size=1)

        with patch('alphalab.storage.pipeline.collectors.Fundamental') as mock_fundamental:
            f1 = Mock()
            f2 = Mock()
            f3 = Mock()
            mock_fundamental.side_effect = [f1, f2, f3]

            one = collector._get_or_create_fundamental("0001")
            two = collector._get_or_create_fundamental("0002")
            again = collector._get_or_create_fundamental("0001")

        assert one is f1
        assert two is f2
        assert again is f3

    def test_get_or_create_fundamental_no_cache(self):
        """Cache size <= 0 returns new instance each time."""
        from alphalab.storage.pipeline import collectors as dc

        mock_logger = Mock(spec=logging.Logger)
        collector = dc.FundamentalDataCollector(logger=mock_logger, fundamental_cache_size=0)

        with patch('alphalab.storage.pipeline.collectors.Fundamental') as mock_fundamental:
            f1 = Mock()
            f2 = Mock()
            mock_fundamental.side_effect = [f1, f2]

            one = collector._get_or_create_fundamental("0001")
            two = collector._get_or_create_fundamental("0001")

        assert one is f1
        assert two is f2
        assert len(collector._fundamental_cache) == 0

    def test_get_or_create_fundamental_cache_hit(self):
        """Cache hit returns existing instance without creating new one."""
        from alphalab.storage.pipeline import collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        cached = Mock()
        collector._fundamental_cache = OrderedDict([("0001", cached)])

        with patch('alphalab.storage.pipeline.collectors.Fundamental') as mock_fundamental:
            result = collector._get_or_create_fundamental("0001")

        assert result is cached
        mock_fundamental.assert_not_called()

    def test_get_or_create_fundamental_cache_hit_moves_to_end(self):
        """Cache hit moves item to end (LRU behavior) - covers lines 466-467."""
        from alphalab.storage.pipeline import collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger), fundamental_cache_size=3)
        cached1 = Mock()
        cached2 = Mock()
        cached3 = Mock()

        # Populate cache with 3 items
        collector._fundamental_cache = OrderedDict([
            ("0001", cached1),
            ("0002", cached2),
            ("0003", cached3)
        ])

        with patch('alphalab.storage.pipeline.collectors.Fundamental') as mock_fundamental:
            # Access the first item
            result = collector._get_or_create_fundamental("0001")

        # Should return cached item
        assert result is cached1
        mock_fundamental.assert_not_called()

        # Item should be moved to end (most recently used)
        keys_list = list(collector._fundamental_cache.keys())
        assert keys_list == ["0002", "0003", "0001"]

    def test_get_or_create_fundamental_evicts_oldest(self):
        """Cache evicts oldest when capacity exceeded."""
        from alphalab.storage.pipeline import collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger), fundamental_cache_size=1)

        with patch('alphalab.storage.pipeline.collectors.Fundamental') as mock_fundamental:
            f1 = Mock()
            f2 = Mock()
            mock_fundamental.side_effect = [f1, f2]

            collector._get_or_create_fundamental("0001")
            collector._get_or_create_fundamental("0002")

        assert "0001" not in collector._fundamental_cache
        assert "0002" in collector._fundamental_cache

    def test_collect_fundamental_long_records(self):
        """Builds records from concept data within date range."""
        from alphalab.storage.pipeline import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp = Mock()
        dp.timestamp = dt.date(2024, 6, 30)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 4, 1)
        dp.end_date = dt.date(2024, 6, 30)
        dp.frame = "CY2024Q2"
        dp.is_instant = False

        fund = Mock()
        fund.get_concept_data.return_value = [dp]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["sales"]
        )

        assert len(result) == 1
        assert result.select("concept").item() == "sales"

    def test_collect_fundamental_long_handles_concept_error(self):
        """Errors in one concept don't block others."""
        from alphalab.storage.pipeline import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp = Mock()
        dp.timestamp = dt.date(2024, 6, 30)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 4, 1)
        dp.end_date = dt.date(2024, 6, 30)
        dp.frame = "CY2024Q2"
        dp.is_instant = False

        fund = Mock()
        fund.get_concept_data.side_effect = [Exception("bad"), [dp]]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["bad", "sales"]
        )

        assert len(result) == 1
        assert result.select("concept").item() == "sales"

    def test_collect_fundamental_long_no_records(self):
        """No records returns empty and logs warning."""
        from alphalab.storage.pipeline import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)
        fund = Mock()
        fund.get_concept_data.return_value = []
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["sales"]
        )

        assert result.is_empty()
        mock_logger.warning.assert_called_once()

    def test_collect_fundamental_long_outer_exception(self):
        """Outer exception returns empty and logs error."""
        from alphalab.storage.pipeline import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)
        collector._load_concepts = Mock(side_effect=ValueError("bad"))

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL"
        )

        assert result.is_empty()
        mock_logger.error.assert_called_once()

    def test_collect_fundamental_long_filters_out_of_range_records(self):
        """Records outside date range are skipped - covers line 511."""
        from alphalab.storage.pipeline import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))

        # Create data points - one in range, one out of range
        dp_in_range = Mock()
        dp_in_range.timestamp = dt.date(2024, 6, 30)
        dp_in_range.accn = "0001"
        dp_in_range.form = "10-Q"
        dp_in_range.value = 123.0
        dp_in_range.start_date = dt.date(2024, 4, 1)
        dp_in_range.end_date = dt.date(2024, 6, 30)
        dp_in_range.frame = "CY2024Q2"
        dp_in_range.is_instant = False

        dp_out_of_range = Mock()
        dp_out_of_range.timestamp = dt.date(2025, 3, 31)  # Outside the end_date
        dp_out_of_range.accn = "0002"
        dp_out_of_range.form = "10-Q"
        dp_out_of_range.value = 456.0
        dp_out_of_range.start_date = dt.date(2025, 1, 1)
        dp_out_of_range.end_date = dt.date(2025, 3, 31)
        dp_out_of_range.frame = "CY2025Q1"
        dp_out_of_range.is_instant = False

        fund = Mock()
        fund.get_concept_data.return_value = [dp_out_of_range, dp_in_range]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["sales"]
        )

        # Should only include the in-range record
        assert len(result) == 1
        assert result.select("accn").item() == "0001"



class TestUniverseDataCollector:
    """Test UniverseDataCollector class (enhanced)"""

    def test_initialization_with_logger(self):
        """Test UniverseDataCollector initialization with logger"""
        from alphalab.storage.pipeline import UniverseDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)

        assert collector.logger == mock_logger

    @patch('alphalab.storage.pipeline.collectors.setup_logger')
    def test_initialization_without_logger(self, mock_setup_logger):
        """Test UniverseDataCollector creates logger if not provided"""
        from alphalab.storage.pipeline import UniverseDataCollector

        mock_logger = Mock(spec=logging.Logger)
        mock_setup_logger.return_value = mock_logger

        collector = UniverseDataCollector()

        mock_setup_logger.assert_called_once()
        assert collector.logger == mock_logger

    @patch('alphalab.storage.pipeline.collectors.fetch_all_stocks')
    def test_collect_current_universe(self, mock_fetch):
        """Test collecting current universe"""
        from alphalab.storage.pipeline import UniverseDataCollector

        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'Company Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.']
        })
        mock_fetch.return_value = mock_stocks

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)
        result = collector.collect_current_universe()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        mock_fetch.assert_called_with(with_filter=True, refresh=False)

    @patch('alphalab.storage.pipeline.collectors.fetch_all_stocks')
    def test_collect_universe_with_filter(self, mock_fetch):
        """Test collecting universe with filter"""
        from alphalab.storage.pipeline import UniverseDataCollector

        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'MSFT'],
            'Company Name': ['Apple Inc.', 'Microsoft Corp.']
        })
        mock_fetch.return_value = mock_stocks

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)
        result = collector.collect_current_universe(with_filter=True, refresh=True)

        assert isinstance(result, pl.DataFrame)
        mock_fetch.assert_called_with(with_filter=True, refresh=True)


class TestDataCollectorsOrchestrator:
    """Test DataCollectors orchestrator (delegation pattern)"""

    def test_initialization_creates_specialized_collectors(self):
        """Test DataCollectors creates all specialized collectors"""
        from alphalab.storage.pipeline import DataCollectors

        mock_alpaca = Mock()
        mock_headers = {}
        mock_logger = Mock(spec=logging.Logger)

        orchestrator = DataCollectors(
            alpaca_ticks=mock_alpaca,
            alpaca_headers=mock_headers,
            logger=mock_logger
        )

        # Verify specialized collectors were created
        assert hasattr(orchestrator, 'ticks_collector')
        assert hasattr(orchestrator, 'fundamental_collector')
        assert hasattr(orchestrator, 'universe_collector')

    def test_shared_fundamental_cache(self):
        """Test DataCollectors creates shared fundamental cache"""
        from alphalab.storage.pipeline import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger,
            fundamental_cache_size=256
        )

        # Orchestrator should have the cache
        assert hasattr(orchestrator, '_fundamental_cache')
        assert hasattr(orchestrator, '_fundamental_cache_lock')

        # Fundamental collector should share the cache
        assert orchestrator.fundamental_collector._fundamental_cache is orchestrator._fundamental_cache
        assert orchestrator.fundamental_collector._fundamental_cache_lock is orchestrator._fundamental_cache_lock

    def test_delegation_to_ticks_collector_range_bulk(self):
        """Delegates range bulk fetch to TicksDataCollector."""
        from alphalab.storage.pipeline import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.collect_daily_ticks_range_bulk = Mock(return_value={})

        orchestrator.collect_daily_ticks_range_bulk(["AAPL", "DISH"], "2017-01-01", "2025-12-31")

        orchestrator.ticks_collector.collect_daily_ticks_range_bulk.assert_called_once_with(
            ["AAPL", "DISH"], "2017-01-01", "2025-12-31"
        )

    def test_delegation_to_ticks_collector(self):
        """Test DataCollectors delegates to TicksDataCollector"""
        from alphalab.storage.pipeline import DataCollectors

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.return_value = {}
        mock_logger = Mock(spec=logging.Logger)

        orchestrator = DataCollectors(
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        # Call delegated method
        orchestrator.collect_daily_ticks_year('AAPL', 2025)

        # Should delegate to ticks_collector
        mock_alpaca.fetch_daily_year_bulk.assert_called_once()

    def test_delegation_to_ticks_collector_month_bulk(self):
        """Delegates bulk month fetch to TicksDataCollector."""
        from alphalab.storage.pipeline import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.collect_daily_ticks_month_bulk = Mock(return_value={})

        orchestrator.collect_daily_ticks_month_bulk(["AAPL"], 2025, 6, sleep_time=0.1)

        orchestrator.ticks_collector.collect_daily_ticks_month_bulk.assert_called_once_with(
            ["AAPL"], 2025, 6, 0.1
        )

    def test_delegation_to_ticks_collector_month(self):
        """Delegates single month fetch to TicksDataCollector."""
        from alphalab.storage.pipeline import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.collect_daily_ticks_month = Mock(return_value=pl.DataFrame())

        orchestrator.collect_daily_ticks_month("AAPL", 2024, 6)

        orchestrator.ticks_collector.collect_daily_ticks_month.assert_called_once_with(
            "AAPL", 2024, 6, year_df=None
        )

    def test_delegation_to_ticks_collector_year_bulk(self):
        """Delegates yearly bulk fetch to TicksDataCollector."""
        from alphalab.storage.pipeline import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.collect_daily_ticks_year_bulk = Mock(return_value={})

        orchestrator.collect_daily_ticks_year_bulk(["AAPL", "MSFT"], 2024)

        orchestrator.ticks_collector.collect_daily_ticks_year_bulk.assert_called_once_with(
            ["AAPL", "MSFT"], 2024
        )

    def test_delegation_to_fundamental_collector(self):
        """Test DataCollectors delegates to FundamentalDataCollector"""
        from alphalab.storage.pipeline import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        # Mock the fundamental collector's method
        orchestrator.fundamental_collector.collect_fundamental_long = Mock(return_value=pl.DataFrame())

        # Call delegated method
        result = orchestrator.collect_fundamental_long('320193', '2024-01-01', '2024-12-31', 'AAPL')

        # Should delegate to fundamental_collector
        orchestrator.fundamental_collector.collect_fundamental_long.assert_called_once_with(
            '320193', '2024-01-01', '2024-12-31', 'AAPL', None, None
        )

    def test_delegation_to_fundamental_collector_load_concepts(self):
        """Delegates _load_concepts to FundamentalDataCollector."""
        from alphalab.storage.pipeline import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.fundamental_collector._load_concepts = Mock(return_value=["sales"])

        result = orchestrator._load_concepts(concepts=["sales"], config_path=None)

        assert result == ["sales"]
        orchestrator.fundamental_collector._load_concepts.assert_called_once_with(["sales"], None)


    @patch('alphalab.storage.pipeline.collectors.fetch_all_stocks')
    def test_backward_compatibility(self, mock_fetch):
        """Test DataCollectors maintains backward compatibility"""
        from alphalab.storage.pipeline import DataCollectors

        mock_fetch.return_value = pl.DataFrame({'Ticker': ['AAPL']})
        mock_logger = Mock(spec=logging.Logger)

        # Should work with same constructor as before (crsp_ticks accepted but ignored)
        orchestrator = DataCollectors(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger,
            sec_rate_limiter=None,
            fundamental_cache_size=128,
            crsp_ticks=Mock()
        )

        # All old methods should still work
        assert hasattr(orchestrator, 'collect_daily_ticks_year')
        assert hasattr(orchestrator, 'collect_fundamental_long')
        assert hasattr(orchestrator, '_load_concepts')


class TestDataCollectorInheritance:
    """Test that all collectors properly inherit from DataCollector ABC"""

    def test_ticks_collector_inherits_datacollector(self):
        """Test TicksDataCollector inherits from DataCollector"""
        from alphalab.storage.pipeline import TicksDataCollector
        from alphalab.collection.models import DataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = TicksDataCollector(
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        assert isinstance(collector, DataCollector)
        assert hasattr(collector, 'logger')

    def test_fundamental_collector_inherits_datacollector(self):
        """Test FundamentalDataCollector inherits from DataCollector"""
        from alphalab.storage.pipeline import FundamentalDataCollector
        from alphalab.collection.models import DataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        assert isinstance(collector, DataCollector)
        assert hasattr(collector, 'logger')

    def test_universe_collector_inherits_datacollector(self):
        """Test UniverseDataCollector inherits from DataCollector"""
        from alphalab.storage.pipeline import UniverseDataCollector
        from alphalab.collection.models import DataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)

        assert isinstance(collector, DataCollector)
        assert hasattr(collector, 'logger')


class TestFundamentalDataCollectorCaching:
    """Test FundamentalDataCollector caching behavior"""

    def test_fundamental_cache_race_condition(self):
        """Test cache returns existing entry on race condition (lines 466,467)."""
        from alphalab.storage.pipeline import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(
            logger=mock_logger,
            sec_rate_limiter=Mock()
        )

        # Pre-populate cache to simulate race condition
        mock_fundamental = Mock()
        collector._fundamental_cache['0000320193'] = mock_fundamental

        # Now try to get the same CIK - should return cached value
        with patch('alphalab.storage.pipeline.collectors.Fundamental') as MockFundamental:
            MockFundamental.return_value = Mock()
            result = collector._get_or_create_fundamental('0000320193', 'AAPL')

        # Should return the cached entry
        assert result == mock_fundamental
        MockFundamental.assert_not_called()

