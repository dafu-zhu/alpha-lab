"""
Unit tests for storage.app module
Focus on upload flows with injected dependencies
"""
from unittest.mock import Mock, patch
from itertools import cycle
import datetime as dt
import queue
import threading
import polars as pl


def _make_app():
    from quantdl.storage.app import UploadApp

    app = UploadApp.__new__(UploadApp)
    app.logger = Mock()
    app.validator = Mock()
    app.data_collectors = Mock()
    app.data_publishers = Mock()
    app.data_collectors.ticks_collector = Mock(alpaca_start_year=2025)
    app.universe_manager = Mock()
    app.calendar = Mock()
    app.cik_resolver = Mock()
    app.sec_rate_limiter = Mock()
    return app


class TestUploadApp:
    def test_upload_daily_ticks_success_monthly(self):
        """Test upload with monthly partitions (default behavior)"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_month.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {
            "status": "success",
            "error": None
        }

        result = app.upload_daily_ticks(2024, use_monthly_partitions=True)

        assert result is None
        # Should be called for 2 symbols × 12 months = 24 times
        assert app.data_collectors.collect_daily_ticks_month.call_count == 24
        assert app.data_publishers.publish_daily_ticks.call_count == 24

    def test_upload_daily_ticks_success_yearly(self):
        """Test upload with yearly partitions (legacy)"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_year.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {
            "status": "success",
            "error": None
        }

        result = app.upload_daily_ticks(2024, use_monthly_partitions=False)

        assert result is None
        assert app.data_collectors.collect_daily_ticks_year.call_count == 2
        assert app.data_publishers.publish_daily_ticks.call_count == 2

    def test_upload_daily_ticks_by_year(self):
        """Test upload with monthly partitions using by_year publishing."""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.validator.data_exists.return_value = False
        app.data_collectors.collect_daily_ticks_year_bulk.return_value = {}
        app.data_publishers.publish_daily_ticks.return_value = {
            "status": "success",
            "error": None
        }

        result = app.upload_daily_ticks(2024, use_monthly_partitions=True, by_year=True)

        assert result is None
        app.data_collectors.collect_daily_ticks_year_bulk.assert_called_once_with(["AAPL", "MSFT"], 2024)
        assert app.data_publishers.publish_daily_ticks.call_count == 2
        app.data_collectors.collect_daily_ticks_month.assert_not_called()
        app.data_collectors.collect_daily_ticks_year.assert_not_called()

    def test_upload_daily_ticks_skips_existing_yearly(self):
        """Test that existing data is skipped for yearly partitions"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.data_exists.return_value = True

        result = app.upload_daily_ticks(2024, use_monthly_partitions=False)

        assert result is None
        app.data_collectors.collect_daily_ticks_year.assert_not_called()
        app.data_publishers.publish_daily_ticks.assert_not_called()

    def test_upload_daily_ticks_collection_error(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.data_exists.return_value = False
        app.data_collectors.collect_daily_ticks_year.side_effect = Exception("Collection failed")

        try:
            app.upload_daily_ticks(2024)
            assert False, "Expected exception"
        except Exception:
            pass

    def test_upload_daily_ticks_publish_error(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_year.return_value = df
        app.data_publishers.publish_daily_ticks.side_effect = Exception("Upload failed")

        try:
            app.upload_daily_ticks(2024)
            assert False, "Expected exception"
        except Exception:
            pass

    def test_publish_single_daily_ticks_skips_existing_yearly(self):
        """Test that existing data is skipped for yearly partitions"""
        app = _make_app()
        app.validator.data_exists.return_value = True

        result = app._publish_single_daily_ticks("AAPL", 2024, overwrite=False, use_monthly_partitions=False)

        assert result["status"] == "canceled"
        app.data_collectors.collect_daily_ticks_year.assert_not_called()
        app.data_publishers.publish_daily_ticks.assert_not_called()

    def test_publish_single_daily_ticks_skips_existing_monthly(self):
        """Test that existing data is skipped for monthly partitions"""
        app = _make_app()
        app.validator.data_exists.return_value = True

        result = app._publish_single_daily_ticks("AAPL", 2024, month=6, overwrite=False, use_monthly_partitions=True)

        assert result["status"] == "canceled"
        app.data_collectors.collect_daily_ticks_month.assert_not_called()
        app.data_publishers.publish_daily_ticks.assert_not_called()

    def test_publish_single_daily_ticks_success_yearly(self):
        """Test successful publish with yearly partition"""
        app = _make_app()
        app.validator.data_exists.return_value = False
        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_year.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success"}

        result = app._publish_single_daily_ticks("AAPL", 2024, overwrite=False, use_monthly_partitions=False)

        assert result["status"] == "success"
        app.data_collectors.collect_daily_ticks_year.assert_called_once()
        app.data_publishers.publish_daily_ticks.assert_called_once_with("AAPL", 2024, df, by_year=False)

    def test_publish_single_daily_ticks_success_monthly(self):
        """Test successful publish with monthly partition"""
        app = _make_app()
        app.validator.data_exists.return_value = False
        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_month.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success"}

        result = app._publish_single_daily_ticks("AAPL", 2024, month=6, overwrite=False, use_monthly_partitions=True)

        assert result["status"] == "success"
        app.data_collectors.collect_daily_ticks_month.assert_called_once_with("AAPL", 2024, 6)
        app.data_publishers.publish_daily_ticks.assert_called_once_with("AAPL", 2024, df, month=6, by_year=False)

    def test_publish_single_daily_ticks_overwrite_ignores_existing(self):
        """Test that overwrite ignores existing check"""
        app = _make_app()
        app.validator.data_exists.return_value = True
        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_year.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success"}

        result = app._publish_single_daily_ticks("AAPL", 2024, overwrite=True, use_monthly_partitions=False)

        assert result["status"] == "success"
        app.data_collectors.collect_daily_ticks_year.assert_called_once()
        app.data_publishers.publish_daily_ticks.assert_called_once()

    def test_upload_minute_ticks_basic_flow(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.calendar.load_trading_days.return_value = ["2024-06-03"]
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-03T09:30:00"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })
        app.data_collectors.fetch_minute_month.return_value = {"AAPL": []}
        app.data_collectors.parse_minute_bars_to_daily.return_value = {("AAPL", "2024-06-03"): df}

        def worker(data_queue, stats, stats_lock):
            while True:
                item = data_queue.get()
                if item is None:
                    break
                with stats_lock:
                    stats["success"] += 1

        app.data_publishers.minute_ticks_worker = worker

        app.upload_minute_ticks(2024, 6, overwrite=True, num_workers=1, chunk_size=1, sleep_time=0.0)

    def test_upload_minute_ticks_filters_symbols_with_missing_days(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.calendar.load_trading_days.return_value = ["2024-06-03", "2024-06-04"]
        app.validator.data_exists.side_effect = [True, True, False]

        df = pl.DataFrame({
            "timestamp": ["2024-06-03T09:30:00"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })
        app.data_collectors.fetch_minute_month.return_value = {"MSFT": []}
        app.data_collectors.parse_minute_bars_to_daily.return_value = {("MSFT", "2024-06-03"): df}

        def worker(data_queue, stats, stats_lock):
            while True:
                item = data_queue.get()
                if item is None:
                    break
                with stats_lock:
                    stats["success"] += 1

        app.data_publishers.minute_ticks_worker = worker

        app.upload_minute_ticks(2024, 6, overwrite=False, num_workers=1, chunk_size=2, sleep_time=0.0)

        app.data_collectors.fetch_minute_month.assert_called_once_with(["MSFT"], 2024, 6, sleep_time=0.0)

    def test_upload_minute_ticks_skips_all_existing(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.calendar.load_trading_days.return_value = ["2024-06-03", "2024-06-04"]
        app.validator.data_exists.return_value = True
        app.data_collectors.fetch_minute_month.return_value = {}

        app.upload_minute_ticks(2024, 6, overwrite=False, num_workers=1, chunk_size=1, sleep_time=0.0)

        app.data_collectors.fetch_minute_month.assert_not_called()

    def test_upload_fundamental_success_flow(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {"AAPL": "0000320193"}
        app._process_symbol_fundamental = Mock(return_value={"status": "success"})

        futures = [Mock()]
        futures[0].result.return_value = {"status": "success"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=futures[0])

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=futures):
                app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        executor.submit.assert_called_once()

    def test_upload_fundamental_skipped_result(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {"AAPL": "0000320193"}
        app._process_symbol_fundamental = Mock(return_value={"status": "skipped"})

        future = Mock()
        future.result.return_value = {"status": "skipped", "symbol": "AAPL", "cik": "0000320193"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future]):
                app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

    def test_upload_ttm_fundamental_success_flow(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {"AAPL": "0000320193"}
        app._process_symbol_ttm_fundamental = Mock(return_value={"status": "success"})

        future = Mock()
        future.result.return_value = {"status": "success"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future]):
                app.upload_ttm_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        executor.submit.assert_called_once()

    def test_upload_derived_fundamental_success_flow(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {"AAPL": "0000320193"}
        app._process_symbol_derived_fundamental = Mock(return_value={"status": "success"})

        future = Mock()
        future.result.return_value = {"status": "success"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future]):
                app.upload_derived_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        executor.submit.assert_called_once()

    def test_process_symbol_fundamental_resolves_cik(self):
        app = _make_app()
        app.validator.data_exists.return_value = False
        app.cik_resolver.get_cik.return_value = "0000320193"
        app.data_publishers.publish_fundamental.return_value = {"status": "success"}

        result = app._process_symbol_fundamental("AAPL", "2024-01-01", "2024-12-31", overwrite=False, cik=None)

        assert result["status"] == "success"
        app.cik_resolver.get_cik.assert_called_once()
        app.data_publishers.publish_fundamental.assert_called_once()

    def test_process_symbol_fundamental_uses_provided_cik(self):
        app = _make_app()
        app.validator.data_exists.return_value = True
        app.data_publishers.publish_fundamental.return_value = {"status": "success"}

        result = app._process_symbol_fundamental("AAPL", "2024-01-01", "2024-12-31", overwrite=False, cik="0000320193")

        assert result["status"] == "success"
        app.cik_resolver.get_cik.assert_not_called()
        app.data_publishers.publish_fundamental.assert_called_once()

    def test_process_symbol_ttm_fundamental_resolves_cik(self):
        app = _make_app()
        app.validator.data_exists.return_value = False
        app.cik_resolver.get_cik.return_value = "0000320193"
        app.data_publishers.publish_ttm_fundamental.return_value = {"status": "success"}

        result = app._process_symbol_ttm_fundamental("AAPL", "2024-01-01", "2024-12-31", overwrite=False, cik=None)

        assert result["status"] == "success"
        app.cik_resolver.get_cik.assert_called_once()
        app.data_publishers.publish_ttm_fundamental.assert_called_once()

    def test_process_symbol_ttm_fundamental_uses_provided_cik(self):
        app = _make_app()
        app.validator.data_exists.return_value = True
        app.data_publishers.publish_ttm_fundamental.return_value = {"status": "success"}

        result = app._process_symbol_ttm_fundamental("AAPL", "2024-01-01", "2024-12-31", overwrite=False, cik="0000320193")

        assert result["status"] == "success"
        app.cik_resolver.get_cik.assert_not_called()
        app.data_publishers.publish_ttm_fundamental.assert_called_once()

    def test_process_symbol_derived_fundamental_no_cik(self):
        app = _make_app()
        app.validator.data_exists.return_value = False
        app.cik_resolver.get_cik.return_value = None

        result = app._process_symbol_derived_fundamental("AAPL", "2024-01-01", "2024-12-31", overwrite=False, cik=None)

        assert result["status"] == "skipped"

    def test_process_symbol_derived_fundamental_empty(self):
        app = _make_app()
        app.validator.data_exists.return_value = False
        app.cik_resolver.get_cik.return_value = "0000320193"
        app.data_collectors.collect_derived_long.return_value = (pl.DataFrame(), "metrics_wide_empty")

        result = app._process_symbol_derived_fundamental("AAPL", "2024-01-01", "2024-12-31", overwrite=False, cik=None)

        assert result["status"] == "skipped"
        assert "metrics_wide_empty" in result["error"]

    def test_process_symbol_derived_fundamental_success(self):
        app = _make_app()
        app.validator.data_exists.return_value = False
        app.cik_resolver.get_cik.return_value = "0000320193"
        derived_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "metric": ["net_mgn"],
            "value": [0.1]
        })
        app.data_collectors.collect_derived_long.return_value = (derived_df, None)
        app.data_publishers.publish_derived_fundamental.return_value = {"status": "success"}

        result = app._process_symbol_derived_fundamental("AAPL", "2024-01-01", "2024-12-31", overwrite=False, cik=None)

        assert result["status"] == "success"
        app.data_publishers.publish_derived_fundamental.assert_called_once()

    def test_upload_minute_ticks_skips_empty_frames(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.calendar.load_trading_days.return_value = ["2024-06-03"]
        app.validator.data_exists.return_value = False
        app.data_collectors.fetch_minute_month.return_value = {"AAPL": []}
        app.data_collectors.parse_minute_bars_to_daily.return_value = {("AAPL", "2024-06-03"): pl.DataFrame()}

        processed = {"count": 0}

        def worker(data_queue, stats, stats_lock):
            while True:
                item = data_queue.get()
                if item is None:
                    break
                processed["count"] += 1

        app.data_publishers.minute_ticks_worker = worker

        app.upload_minute_ticks(2024, 6, overwrite=True, num_workers=1, chunk_size=1, sleep_time=0.0)

        assert processed["count"] == 0

    def test_upload_fundamental_no_symbols_with_cik(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {}
        app._process_symbol_fundamental = Mock()

        app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        app._process_symbol_fundamental.assert_not_called()

    def test_upload_ttm_fundamental_no_symbols_with_cik(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {}
        app._process_symbol_ttm_fundamental = Mock()

        app.upload_ttm_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        app._process_symbol_ttm_fundamental.assert_not_called()

    def test_upload_derived_fundamental_no_symbols_with_cik(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {}
        app._process_symbol_derived_fundamental = Mock()

        app.upload_derived_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        app._process_symbol_derived_fundamental.assert_not_called()

    def test_upload_derived_fundamental_warns_on_empty_year(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.side_effect = [[], ["AAPL"]]
        app.cik_resolver.batch_prefetch_ciks.return_value = {"AAPL": "0000320193"}
        app._process_symbol_derived_fundamental = Mock(return_value={"status": "success"})

        future = Mock()
        future.result.return_value = {"status": "success"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future]):
                app.upload_derived_fundamental("2024-01-01", "2025-12-31", max_workers=1, overwrite=False)

        app.logger.warning.assert_called()

    def test_upload_top_3000_monthly_success(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.top_3000_exists.return_value = False
        app.calendar.load_trading_days.return_value = ["2024-06-28"]
        app.universe_manager.get_top_3000.return_value = ["AAPL"]
        app.data_publishers.publish_top_3000.return_value = {"status": "success"}

        class FixedDate(dt.date):
            @classmethod
            def today(cls):
                return cls(2024, 7, 1)

        with patch('quantdl.storage.app.dt.date', FixedDate):
            app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.data_publishers.publish_top_3000.assert_called()

    def test_upload_top_3000_monthly_no_symbols(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = []

        app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.data_publishers.publish_top_3000.assert_not_called()

    def test_upload_top_3000_monthly_skips_existing(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.top_3000_exists.return_value = True

        app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.data_publishers.publish_top_3000.assert_not_called()

    def test_upload_top_3000_monthly_no_trading_days(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.top_3000_exists.return_value = False
        app.calendar.load_trading_days.return_value = []

        app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.data_publishers.publish_top_3000.assert_not_called()

    def test_upload_top_3000_monthly_skipped_result_logs(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.top_3000_exists.return_value = False
        app.calendar.load_trading_days.return_value = ["2024-06-28"]
        app.universe_manager.get_top_3000.return_value = ["AAPL"]
        app.data_publishers.publish_top_3000.return_value = {"status": "skipped", "error": "No symbols"}

        class FixedDate(dt.date):
            @classmethod
            def today(cls):
                return cls(2024, 7, 1)

        with patch('quantdl.storage.app.dt.date', FixedDate):
            app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.logger.warning.assert_called()

    def test_upload_top_3000_monthly_failed_result_logs(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.top_3000_exists.return_value = False
        app.calendar.load_trading_days.return_value = ["2024-06-28"]
        app.universe_manager.get_top_3000.return_value = ["AAPL"]
        app.data_publishers.publish_top_3000.return_value = {"status": "failed", "error": "boom"}

        class FixedDate(dt.date):
            @classmethod
            def today(cls):
                return cls(2024, 7, 1)

        with patch('quantdl.storage.app.dt.date', FixedDate):
            app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.logger.error.assert_called()

    def test_upload_top_3000_monthly_stops_future_month(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.top_3000_exists.return_value = False
        app.calendar.load_trading_days.return_value = ["2025-12-31"]

        class FixedDate(dt.date):
            @classmethod
            def today(cls):
                return cls(2024, 7, 1)

        with patch('quantdl.storage.app.dt.date', FixedDate):
            app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.data_publishers.publish_top_3000.assert_not_called()

    def test_upload_daily_ticks_monthly_bulk_alpaca(self):
        """Alpaca monthly bulk path uses collect_daily_ticks_month_bulk."""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.data_exists.return_value = False
        app.data_collectors.collect_daily_ticks_month_bulk.return_value = {
            "AAPL": pl.DataFrame({
                "timestamp": ["2025-06-30"],
                "open": [100.0],
                "high": [110.0],
                "low": [95.0],
                "close": [105.0],
                "volume": [1000]
            })
        }
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success", "error": None}

        app.upload_daily_ticks(2025, use_monthly_partitions=True, by_year=False, chunk_size=1, sleep_time=0.5)

        app.data_collectors.collect_daily_ticks_month_bulk.assert_any_call(
            ["AAPL"], 2025, 1, sleep_time=0.5
        )
        app.data_publishers.publish_daily_ticks.assert_called()

    def test_upload_daily_ticks_alpaca_by_year_ignored(self):
        """by_year=True is ignored for Alpaca years and uses bulk monthly fetch."""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.data_exists.return_value = False
        app.data_collectors.collect_daily_ticks_month_bulk.return_value = {
            "AAPL": pl.DataFrame({
                "timestamp": ["2025-06-30"],
                "open": [100.0],
                "high": [110.0],
                "low": [95.0],
                "close": [105.0],
                "volume": [1000]
            })
        }
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success", "error": None}

        app.upload_daily_ticks(2025, use_monthly_partitions=True, by_year=True, chunk_size=1, sleep_time=0.0)

        app.data_collectors.collect_daily_ticks_year_bulk.assert_not_called()
        app.data_collectors.collect_daily_ticks_month_bulk.assert_called()

    def test_upload_daily_ticks_yearly_bulk_alpaca(self):
        """Alpaca yearly bulk path uses collect_daily_ticks_year_bulk."""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.validator.data_exists.return_value = False
        app.data_collectors.collect_daily_ticks_year_bulk.return_value = {
            "AAPL": pl.DataFrame({"timestamp": ["2025-01-02"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}),
            "MSFT": pl.DataFrame({"timestamp": ["2025-01-02"], "open": [2], "high": [2], "low": [2], "close": [2], "volume": [2]})
        }
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success", "error": None}

        app.upload_daily_ticks(2025, use_monthly_partitions=False, chunk_size=2)

        app.data_collectors.collect_daily_ticks_year_bulk.assert_called_once_with(["AAPL", "MSFT"], 2025)
        assert app.data_publishers.publish_daily_ticks.call_count == 2

    def test_upload_daily_ticks_monthly_alpaca_skips_existing(self):
        """Monthly Alpaca bulk path skips when all data exists."""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.data_exists.return_value = True

        app.upload_daily_ticks(2025, use_monthly_partitions=True, by_year=False, chunk_size=1, sleep_time=0.0)

        app.data_collectors.collect_daily_ticks_month_bulk.assert_not_called()
        app.data_publishers.publish_daily_ticks.assert_not_called()

    def test_upload_daily_ticks_yearly_alpaca_skips_existing(self):
        """Yearly Alpaca bulk path skips when all data exists."""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.data_exists.return_value = True

        app.upload_daily_ticks(2025, use_monthly_partitions=False, chunk_size=1)

        app.data_collectors.collect_daily_ticks_year_bulk.assert_not_called()
        app.data_publishers.publish_daily_ticks.assert_not_called()

    def test_upload_top_3000_monthly_uses_alpaca_start_year(self):
        """Top3000 source uses alpaca_start_year threshold."""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.validator.top_3000_exists.return_value = False
        app.calendar.load_trading_days.return_value = ["2024-06-28"]
        app.universe_manager.get_top_3000.return_value = ["AAPL"]
        app.data_publishers.publish_top_3000.return_value = {"status": "success"}

        class FixedDate(dt.date):
            @classmethod
            def today(cls):
                return cls(2024, 7, 1)

        with patch('quantdl.storage.app.dt.date', FixedDate):
            app.upload_top_3000_monthly(2024, overwrite=False, auto_resolve=True)

        app.data_publishers.publish_top_3000.assert_called()

    def test_run_invokes_selected_flows(self):
        app = _make_app()
        app.upload_fundamental = Mock()
        app.upload_derived_fundamental = Mock()
        app.upload_ttm_fundamental = Mock()
        app.upload_daily_ticks = Mock()
        app.upload_minute_ticks = Mock()
        app.upload_top_3000_monthly = Mock()

        app.run(
            start_year=2024,
            end_year=2024,
            max_workers=1,
            overwrite=False,
            run_fundamental=True,
            run_derived_fundamental=True,
            run_ttm_fundamental=True,
            run_daily_ticks=True,
            run_minute_ticks=False,
            run_top_3000=True
        )

        app.upload_fundamental.assert_called_once()
        app.upload_derived_fundamental.assert_called_once()
        app.upload_ttm_fundamental.assert_called_once()
        app.upload_daily_ticks.assert_called_once()
        app.upload_top_3000_monthly.assert_called_once()

    def test_run_skips_minute_ticks_before_2017(self):
        app = _make_app()
        app.upload_minute_ticks = Mock()

        app.run(
            start_year=2016,
            end_year=2016,
            max_workers=1,
            overwrite=False,
            run_minute_ticks=True
        )

        app.upload_minute_ticks.assert_not_called()

    def test_run_minute_ticks_runs_all_months(self):
        app = _make_app()
        app.upload_minute_ticks = Mock()

        app.run(
            start_year=2017,
            end_year=2017,
            max_workers=1,
            overwrite=False,
            run_minute_ticks=True
        )

        assert app.upload_minute_ticks.call_count == 12

    def test_run_passes_daily_chunk_and_sleep(self):
        app = _make_app()
        app.upload_daily_ticks = Mock()

        app.run(
            start_year=2025,
            end_year=2025,
            max_workers=1,
            overwrite=False,
            daily_chunk_size=123,
            daily_sleep_time=0.12,
            run_daily_ticks=True,
            run_minute_ticks=False
        )

        app.upload_daily_ticks.assert_called_once_with(
            2025,
            False,
            by_year=True,
            chunk_size=123,
            sleep_time=0.12
        )

    def test_run_respects_minute_ticks_start_year(self):
        app = _make_app()
        app.upload_minute_ticks = Mock()

        app.run(
            start_year=2018,
            end_year=2018,
            max_workers=1,
            overwrite=False,
            minute_ticks_start_year=2019,
            run_minute_ticks=True
        )

        app.upload_minute_ticks.assert_not_called()

    def test_upload_minute_ticks_handles_fetch_error(self):
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.calendar.load_trading_days.return_value = ["2024-06-03"]
        app.validator.data_exists.return_value = False
        app.data_collectors.fetch_minute_month.side_effect = RuntimeError("boom")

        app.upload_minute_ticks(2024, 6, overwrite=True, num_workers=1, chunk_size=1, sleep_time=0.0)

    def test_close_closes_wrds_connection(self):
        app = _make_app()
        app.universe_manager = Mock()
        app.crsp_ticks = Mock()
        app.crsp_ticks.conn = Mock()

        app.close()

        app.universe_manager.close.assert_called_once()
        app.crsp_ticks.conn.close.assert_called_once()

    def test_upload_daily_ticks_monthly_with_canceled_status(self):
        """Test monthly upload with canceled status"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT", "GOOGL"]
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_month.return_value = df
        # First success, then canceled, then skipped
        app.data_publishers.publish_daily_ticks.side_effect = cycle([
            {"status": "success"},
            {"status": "canceled"},
            {"status": "skipped"},
        ])

        app.upload_daily_ticks(2024, use_monthly_partitions=True)

        # Should log all status types
        assert app.logger.info.called

    def test_upload_daily_ticks_monthly_with_failed_status(self):
        """Test monthly upload with failed status"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_month.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {"status": "failed", "error": "Upload error"}

        app.upload_daily_ticks(2024, use_monthly_partitions=True)

        assert app.logger.info.called

    def test_upload_daily_ticks_yearly_progress_logging(self):
        """Test progress logging every 50 symbols for yearly partitions"""
        app = _make_app()
        # Create 100 symbols to trigger progress logging at 50
        symbols = [f"SYM{i:03d}" for i in range(100)]
        app.universe_manager.load_symbols_for_year.return_value = symbols
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_year.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success"}

        app.upload_daily_ticks(2024, use_monthly_partitions=False)

        # Should log progress at 50 symbols
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        progress_logs = [c for c in info_calls if "Progress: 50/" in c]
        assert len(progress_logs) > 0

    def test_upload_daily_ticks_monthly_progress_logging(self):
        """Test progress logging every 100 symbols for monthly partitions"""
        app = _make_app()
        # Create 150 symbols to trigger progress logging at 100
        symbols = [f"SYM{i:03d}" for i in range(150)]
        app.universe_manager.load_symbols_for_year.return_value = symbols
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        app.data_collectors.collect_daily_ticks_month.return_value = df
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success"}

        app.upload_daily_ticks(2024, use_monthly_partitions=True)

        # Should log progress at 100 symbols for first month
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        progress_logs = [c for c in info_calls if "Progress: 100/" in c]
        assert len(progress_logs) > 0

    def test_upload_fundamental_with_skipped_symbols_detail_logging(self):
        """Test detailed logging of skipped symbols with company names"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {
            "AAPL": "0000320193",
            "MSFT": "0000789019"
        }

        # Mock crsp_ticks.security_master for company name lookup
        app.crsp_ticks = Mock()
        master_tb = pl.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "company": ["Apple Inc.", "Microsoft Corporation"],
            "cik": ["0000320193", "0000789019"]
        })
        app.crsp_ticks.security_master.master_tb = master_tb

        # First symbol succeeds, second is skipped
        future1 = Mock()
        future1.result.return_value = {"status": "success"}
        future2 = Mock()
        future2.result.return_value = {
            "status": "skipped",
            "symbol": "MSFT",
            "cik": "0000789019",
            "error": "No data available"
        }

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(side_effect=[future1, future2])

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future1, future2]):
                app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should log detailed skipped companies section
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        skipped_logs = [c for c in info_calls if "SKIPPED COMPANIES" in c or "MSFT" in c]
        assert len(skipped_logs) > 0

    def test_upload_fundamental_skipped_logging_error_handling(self):
        """Test error handling when fetching company details for skipped symbols fails"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {"AAPL": "0000320193"}

        # Mock crsp_ticks.security_master to raise an error
        app.crsp_ticks = Mock()
        master_tb = Mock()
        master_tb.filter.side_effect = Exception("DB error")
        app.crsp_ticks.security_master.master_tb = master_tb

        future = Mock()
        future.result.return_value = {
            "status": "skipped",
            "symbol": "AAPL",
            "cik": "0000320193",
            "error": "No data"
        }

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future]):
                app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should log error and fallback
        app.logger.error.assert_called()

    def test_upload_fundamental_non_sec_filers_large_list(self):
        """Test logging of non-SEC filers when list is large (>30 symbols)"""
        app = _make_app()
        # Create 50 symbols, only 10 have CIKs
        symbols = [f"SYM{i:03d}" for i in range(50)]
        app.universe_manager.load_symbols_for_year.return_value = symbols

        # Only first 10 have CIKs
        cik_map = {f"SYM{i:03d}": f"{i:010d}" for i in range(10)}
        app.cik_resolver.batch_prefetch_ciks.return_value = cik_map

        future = Mock()
        future.result.return_value = {"status": "success"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future] * 10):
                app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should log first 30 non-SEC filers
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        non_sec_logs = [c for c in info_calls if "showing first 30/" in c]
        assert len(non_sec_logs) > 0

    def test_upload_fundamental_non_sec_filers_small_list(self):
        """Test logging of non-SEC filers when list is small (<=30 symbols)"""
        app = _make_app()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        app.universe_manager.load_symbols_for_year.return_value = symbols

        # Only AAPL has CIK
        cik_map = {"AAPL": "0000320193"}
        app.cik_resolver.batch_prefetch_ciks.return_value = cik_map

        future = Mock()
        future.result.return_value = {"status": "success"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future]):
                app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should log all non-SEC filers (2 symbols)
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        non_sec_logs = [c for c in info_calls if "Non-SEC filers (skipped):" in c]
        assert len(non_sec_logs) > 0

    def test_upload_fundamental_progress_logging_every_50(self):
        """Test progress logging every 50 symbols in fundamental upload"""
        app = _make_app()
        # Create 100 symbols to trigger progress logging
        symbols = [f"SYM{i:03d}" for i in range(100)]
        app.universe_manager.load_symbols_for_year.return_value = symbols
        cik_map = {sym: f"{i:010d}" for i, sym in enumerate(symbols)}
        app.cik_resolver.batch_prefetch_ciks.return_value = cik_map

        future = Mock()
        future.result.return_value = {"status": "success"}
        futures = [future] * 100

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=futures):
                app.upload_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should log progress at 50 symbols
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        progress_logs = [c for c in info_calls if "Progress: 50/" in c]
        assert len(progress_logs) > 0

    def test_upload_ttm_fundamental_progress_logging(self):
        """Test progress logging in TTM fundamental upload"""
        app = _make_app()
        symbols = [f"SYM{i:03d}" for i in range(100)]
        app.universe_manager.load_symbols_for_year.return_value = symbols
        cik_map = {sym: f"{i:010d}" for i, sym in enumerate(symbols)}
        app.cik_resolver.batch_prefetch_ciks.return_value = cik_map

        future = Mock()
        future.result.return_value = {"status": "success"}
        futures = [future] * 100

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=futures):
                app.upload_ttm_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should log progress at 50 symbols
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        progress_logs = [c for c in info_calls if "Progress: 50/" in c]
        assert len(progress_logs) > 0

    def test_upload_ttm_fundamental_with_skipped_symbols(self):
        """Test TTM upload with skipped symbols tracking"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {
            "AAPL": "0000320193",
            "MSFT": "0000789019"
        }

        future1 = Mock()
        future1.result.return_value = {"status": "success"}
        future2 = Mock()
        future2.result.return_value = {
            "status": "skipped",
            "symbol": "MSFT",
            "cik": "0000789019",
            "error": "Insufficient data"
        }

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(side_effect=[future1, future2])

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future1, future2]):
                app.upload_ttm_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should complete successfully
        assert app.logger.info.called

    def test_upload_ttm_fundamental_with_canceled_and_failed(self):
        """Test TTM upload with various status codes"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOGL": "0001652044",
            "TSLA": "0001318605"
        }

        futures = [
            Mock(result=Mock(return_value={"status": "success"})),
            Mock(result=Mock(return_value={"status": "canceled"})),
            Mock(result=Mock(return_value={"status": "skipped", "symbol": "GOOGL", "cik": "0001652044", "error": "No data"})),
            Mock(result=Mock(return_value={"status": "failed"}))
        ]

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(side_effect=futures)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=futures):
                app.upload_ttm_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        assert app.logger.info.called

    def test_upload_derived_fundamental_progress_logging(self):
        """Test progress logging in derived fundamental upload"""
        app = _make_app()
        symbols = [f"SYM{i:03d}" for i in range(100)]
        app.universe_manager.load_symbols_for_year.return_value = symbols
        cik_map = {sym: f"{i:010d}" for i, sym in enumerate(symbols)}
        app.cik_resolver.batch_prefetch_ciks.return_value = cik_map

        future = Mock()
        future.result.return_value = {"status": "success"}
        futures = [future] * 100

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=futures):
                app.upload_derived_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        # Should log progress at 50 symbols
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        progress_logs = [c for c in info_calls if "Progress: 50/" in c]
        assert len(progress_logs) > 0

    def test_upload_derived_fundamental_with_various_statuses(self):
        """Test derived upload with mixed success, failed, canceled, skipped"""
        app = _make_app()
        app.universe_manager.load_symbols_for_year.return_value = ["A", "B", "C", "D"]
        app.cik_resolver.batch_prefetch_ciks.return_value = {
            "A": "0000000001",
            "B": "0000000002",
            "C": "0000000003",
            "D": "0000000004"
        }

        futures = [
            Mock(result=Mock(return_value={"status": "success"})),
            Mock(result=Mock(return_value={"status": "canceled"})),
            Mock(result=Mock(return_value={"status": "skipped"})),
            Mock(result=Mock(return_value={"status": "failed"}))
        ]

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(side_effect=futures)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=futures):
                app.upload_derived_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)

        assert app.logger.info.called

    def test_upload_minute_ticks_progress_logging_every_100(self):
        """Test progress logging every 100 tasks in minute ticks upload"""
        app = _make_app()
        # Create 150 symbols with 1 day each = 150 tasks
        symbols = [f"SYM{i:03d}" for i in range(150)]
        app.universe_manager.load_symbols_for_year.return_value = symbols
        app.calendar.load_trading_days.return_value = ["2024-06-03"]
        app.validator.data_exists.return_value = False

        df = pl.DataFrame({
            "timestamp": ["2024-06-03T09:30:00"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })

        # Mock fetch and parse to return data for all symbols
        def mock_fetch(chunk, year, month, sleep_time):
            return {sym: [] for sym in chunk}

        def mock_parse(symbol_bars, trading_days):
            return {(sym, day): df for sym in symbol_bars.keys() for day in trading_days}

        app.data_collectors.fetch_minute_month.side_effect = mock_fetch
        app.data_collectors.parse_minute_bars_to_daily.side_effect = mock_parse

        def worker(data_queue, stats, stats_lock):
            while True:
                item = data_queue.get()
                if item is None:
                    break
                with stats_lock:
                    stats["success"] += 1

        app.data_publishers.minute_ticks_worker = worker

        app.upload_minute_ticks(2024, 6, overwrite=True, num_workers=1, chunk_size=30, sleep_time=0.0)

        # Should log progress at 100 tasks
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        progress_logs = [c for c in info_calls if "Progress: 100/" in c]
        assert len(progress_logs) > 0

    def test_close_without_universe_manager(self):
        """Test close when universe_manager is None"""
        app = _make_app()
        app.universe_manager = None
        app.crsp_ticks = None

        # Should not raise an error
        app.close()

    def test_close_without_crsp_conn(self):
        """Test close when crsp_ticks.conn is None"""
        app = _make_app()
        app.universe_manager = Mock()
        app.crsp_ticks = Mock()
        app.crsp_ticks.conn = None

        app.close()

        app.universe_manager.close.assert_called_once()

    def test_upload_daily_ticks_by_year_crsp_bulk(self):
        """Test by_year mode with CRSP bulk fetch for year < alpaca_start_year"""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]
        app.validator.data_exists.return_value = False

        bulk_map = {
            "AAPL": pl.DataFrame({"timestamp": ["2024-01-02"], "open": [100], "high": [110], "low": [95], "close": [105], "volume": [1000]}),
            "MSFT": pl.DataFrame({"timestamp": ["2024-01-02"], "open": [200], "high": [210], "low": [195], "close": [205], "volume": [2000]})
        }
        app.data_collectors.collect_daily_ticks_year_bulk.return_value = bulk_map
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success"}

        app.upload_daily_ticks(2024, use_monthly_partitions=True, by_year=True)

        app.data_collectors.collect_daily_ticks_year_bulk.assert_called_once_with(["AAPL", "MSFT"], 2024)
        assert app.data_publishers.publish_daily_ticks.call_count == 2

    def test_upload_daily_ticks_by_year_checks_any_month_exists(self):
        """Test by_year mode skips symbols where any month exists"""
        app = _make_app()
        app.data_collectors.ticks_collector.alpaca_start_year = 2025
        app.universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]

        # AAPL has data for month 1, MSFT has no data
        def mock_exists(sym, data_type, year, month=None):
            if sym == "AAPL" and month == 1:
                return True
            return False

        app.validator.data_exists.side_effect = mock_exists

        bulk_map = {
            "MSFT": pl.DataFrame({"timestamp": ["2024-01-02"], "open": [200], "high": [210], "low": [195], "close": [205], "volume": [2000]})
        }
        app.data_collectors.collect_daily_ticks_year_bulk.return_value = bulk_map
        app.data_publishers.publish_daily_ticks.return_value = {"status": "success"}

        app.upload_daily_ticks(2024, use_monthly_partitions=True, by_year=True, overwrite=False)

        # Only MSFT should be published
        app.data_publishers.publish_daily_ticks.assert_called_once()

    def test_upload_fundamental_multi_year_symbol_collection(self):
        """Test fundamental upload collects symbols across multiple years correctly"""
        app = _make_app()

        # Different symbols for different years
        def mock_load_symbols(year, sym_type):
            if year == 2024:
                return ["AAPL", "MSFT"]
            elif year == 2025:
                return ["MSFT", "GOOGL"]  # MSFT appears in both
            return []

        app.universe_manager.load_symbols_for_year.side_effect = mock_load_symbols
        app.cik_resolver.batch_prefetch_ciks.return_value = {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOGL": "0001652044"
        }

        future = Mock()
        future.result.return_value = {"status": "success"}

        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit = Mock(return_value=future)

        with patch('quantdl.storage.app.ThreadPoolExecutor', return_value=executor):
            with patch('quantdl.storage.app.as_completed', return_value=[future, future, future]):
                app.upload_fundamental("2024-01-01", "2025-12-31", max_workers=1, overwrite=False)

        # Should have 3 unique symbols total
        assert executor.submit.call_count == 3
