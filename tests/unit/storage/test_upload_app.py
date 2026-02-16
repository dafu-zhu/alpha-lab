"""
Unit tests for storage.app module
Focus on UploadApp initialization and handler delegation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import datetime as dt
import logging
import os
import polars as pl
from quantdl.storage.utils import NoSuchKeyError


def _make_app():
    """Create a minimal mock UploadApp for testing."""
    from quantdl.storage.app import UploadApp

    app = UploadApp.__new__(UploadApp)
    app.logger = Mock()

    # Mock storage client - simulate no progress file exists
    app.client = Mock()
    app.client.get_object.side_effect = NoSuchKeyError('bucket', 'key')

    # Mock dependencies
    app.validator = Mock()
    app.data_collectors = Mock()
    app.data_publishers = Mock()
    app.data_collectors.ticks_collector = Mock(alpaca_start_year=2025)
    app.universe_manager = Mock()
    app.calendar = Mock()
    app.cik_resolver = Mock()
    app.sec_rate_limiter = Mock()
    app.security_master = Mock()
    app._start_year = 2017
    app.headers = {
        "APCA-API-KEY-ID": "test_key",
        "APCA-API-SECRET-KEY": "test_secret"
    }
    app.alpaca_ticks = Mock()

    return app


class TestUploadAppInitialization:
    """Test UploadApp constructor and initialization."""

    @patch('quantdl.storage.app.DataPublishers')
    @patch('quantdl.storage.app.DataCollectors')
    @patch('quantdl.storage.app.CIKResolver')
    @patch('quantdl.storage.app.RateLimiter')
    @patch('quantdl.storage.app.TradingCalendar')
    @patch('quantdl.storage.app.UniverseManager')
    @patch('quantdl.storage.app.SecurityMaster')
    @patch('quantdl.storage.app.Ticks')
    @patch('quantdl.storage.app.Validator')
    @patch('quantdl.storage.app.setup_logger')
    @patch('quantdl.storage.app.StorageClient')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret',
        'LOCAL_STORAGE_PATH': '/tmp/test-storage'
    })
    def test_initialization(
        self,
        mock_storage_client,
        mock_logger,
        mock_validator,
        mock_ticks,
        mock_security_master,
        mock_universe,
        mock_calendar,
        mock_rate_limiter,
        mock_cik_resolver,
        mock_collectors,
        mock_publishers
    ):
        """Test UploadApp constructor wiring and defaults."""
        from quantdl.storage.app import UploadApp

        mock_storage_instance = Mock()
        mock_storage_client.return_value = mock_storage_instance

        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        app = UploadApp(start_year=2017)

        assert app.client == mock_storage_instance
        assert app.logger == mock_logger_instance
        assert app.validator == mock_validator.return_value
        assert app.alpaca_ticks == mock_ticks.return_value
        assert app.security_master == mock_security_master.return_value
        assert app.universe_manager == mock_universe.return_value
        assert app.calendar == mock_calendar.return_value
        assert app.sec_rate_limiter == mock_rate_limiter.return_value
        assert app.cik_resolver == mock_cik_resolver.return_value
        assert app.data_collectors == mock_collectors.return_value
        assert app.data_publishers == mock_publishers.return_value

        mock_logger.assert_called_once_with(
            name="uploadapp",
            log_dir=Path("data/logs/upload"),
            level=logging.DEBUG,
            console_output=True
        )
        assert app.headers["APCA-API-KEY-ID"] == "test_key"
        assert app.headers["APCA-API-SECRET-KEY"] == "test_secret"
        assert app._start_year == 2017


class TestUploadAppDailyTicks:
    """Test daily ticks upload delegation to handler."""

    def test_upload_daily_ticks_delegates_to_handler(self):
        """Test upload_daily_ticks creates handler and calls upload_year."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload_year.return_value = None

        with patch.object(app, '_get_daily_ticks_handler', return_value=mock_handler):
            result = app.upload_daily_ticks(
                year=2024,
                overwrite=True,
                chunk_size=100,
                sleep_time=0.5,
            )

        mock_handler.upload_year.assert_called_once_with(
            year=2024,
            overwrite=True,
            chunk_size=100,
            sleep_time=0.5,
        )
        assert result is None

    def test_upload_daily_ticks_default_params(self):
        """Test upload_daily_ticks uses default parameters."""
        app = _make_app()
        mock_handler = Mock()

        with patch.object(app, '_get_daily_ticks_handler', return_value=mock_handler):
            app.upload_daily_ticks(year=2024)

        mock_handler.upload_year.assert_called_once_with(
            year=2024,
            overwrite=False,
            chunk_size=200,
            sleep_time=0.2,
        )


class TestUploadAppFundamental:
    """Test fundamental data upload delegation to handlers."""

    def test_upload_fundamental_delegates_to_handler(self):
        """Test upload_fundamental creates handler and calls upload."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload.return_value = {'success': 10, 'failed': 0}

        with patch.object(app, '_get_fundamental_handler', return_value=mock_handler):
            result = app.upload_fundamental(
                start_date='2024-01-01',
                end_date='2024-12-31',
                max_workers=25,
                overwrite=True
            )

        mock_handler.upload.assert_called_once_with(
            '2024-01-01', '2024-12-31', 25, True
        )
        assert result == {'success': 10, 'failed': 0}

class TestUploadAppTop3000:
    """Test top 3000 upload delegation to handler."""

    def test_upload_top_3000_monthly_delegates_to_handler(self):
        """Test upload_top_3000_monthly creates handler and calls upload_year."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload_year.return_value = None

        with patch.object(app, '_get_top3000_handler', return_value=mock_handler):
            app.upload_top_3000_monthly(
                year=2024,
                overwrite=True,
                auto_resolve=False
            )

        mock_handler.upload_year.assert_called_once_with(2024, True, False)


class TestUploadAppRun:
    """Test UploadApp.run() orchestration."""

    def test_run_invokes_selected_flows(self):
        """Test run() invokes only the selected upload methods."""
        app = _make_app()

        # Mock the internal methods that run() calls
        app._run_daily_ticks = Mock()
        app.upload_fundamental = Mock()
        app.upload_top_3000_monthly = Mock()

        app.run(
            start_year=2024,
            end_year=2024,
            run_fundamental=True,
            run_daily_ticks=True,
            run_top_3000=False
        )

        # Fundamental and daily ticks should be called
        assert app.upload_fundamental.called
        assert app._run_daily_ticks.called

        # Top 3000 should not be called
        assert not app.upload_top_3000_monthly.called

    def test_run_all_enables_all_flows(self):
        """Test run_all=True enables all upload methods."""
        app = _make_app()

        app._run_daily_ticks = Mock()
        app.upload_fundamental = Mock()
        app.upload_top_3000_monthly = Mock()
        mock_features_handler = Mock()
        app._get_features_handler = Mock(return_value=mock_features_handler)

        app.run(
            start_year=2024,
            end_year=2024,
            run_all=True,
        )

        assert app._run_daily_ticks.called
        assert app.upload_fundamental.called
        assert app.upload_top_3000_monthly.called
        assert mock_features_handler.build.called

    def test_run_passes_daily_chunk_and_sleep(self):
        """Test run() passes daily_chunk_size and daily_sleep_time to _run_daily_ticks."""
        app = _make_app()

        app._run_daily_ticks = Mock()
        app.upload_fundamental = Mock()

        app.run(
            start_year=2024,
            end_year=2024,
            run_daily_ticks=True,
            daily_chunk_size=100,
            daily_sleep_time=0.5
        )

        app._run_daily_ticks.assert_called_once_with(2024, 2024, False, 100, 0.5)


class TestUploadAppClose:
    """Test UploadApp resource cleanup."""

    def test_close_closes_security_master(self):
        """Test close() closes SecurityMaster."""
        app = _make_app()

        app.close()

        app.security_master.close.assert_called_once()

    def test_close_without_security_master(self):
        """Test close() handles missing SecurityMaster."""
        app = _make_app()
        app.security_master = None

        # Should not raise
        app.close()

    def test_close_without_universe_manager(self):
        """Test close() handles missing universe manager."""
        app = _make_app()
        app.universe_manager = None

        # Should not raise
        app.close()


class TestUploadAppHandlerFactories:
    """Test handler factory methods create correct handlers."""

    def test_get_daily_ticks_handler_creates_handler(self):
        """Test _get_daily_ticks_handler creates DailyTicksHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.ticks.DailyTicksHandler') as MockHandler:
            handler = app._get_daily_ticks_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['data_publishers'] == app.data_publishers
        assert call_kwargs['logger'] == app.logger

    def test_get_fundamental_handler_creates_handler(self):
        """Test _get_fundamental_handler creates FundamentalHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.fundamental.FundamentalHandler') as MockHandler:
            handler = app._get_fundamental_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['cik_resolver'] == app.cik_resolver
        assert call_kwargs['sec_rate_limiter'] == app.sec_rate_limiter
        assert call_kwargs['security_master'] == app.security_master

    def test_get_top3000_handler_creates_handler(self):
        """Test _get_top3000_handler creates Top3000Handler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.top3000.Top3000Handler') as MockHandler:
            handler = app._get_top3000_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['calendar'] == app.calendar
        assert call_kwargs['logger'] == app.logger
