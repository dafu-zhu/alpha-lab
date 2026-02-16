"""Unit tests for DailyTicksHandler."""

import datetime as dt

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging
import polars as pl

from quantdl.storage.handlers.ticks import DailyTicksHandler


# -- Shared fixtures ----------------------------------------------------------


@pytest.fixture
def daily_deps():
    """Mocked dependencies for DailyTicksHandler."""
    return {
        'data_publishers': Mock(),
        'data_collectors': Mock(),
        'security_master': Mock(),
        'validator': Mock(),
        'logger': Mock(spec=logging.Logger),
    }


# -- DailyTicksHandler -------------------------------------------------------


class TestDailyTicksHandler:

    @pytest.fixture
    def handler(self, daily_deps):
        return DailyTicksHandler(**daily_deps)

    def test_init(self, handler, daily_deps):
        assert handler.publishers is daily_deps['data_publishers']
        assert handler.collectors is daily_deps['data_collectors']
        assert handler.security_master is daily_deps['security_master']
        assert handler.validator is daily_deps['validator']

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_range_with_data(self, mock_tqdm, handler, daily_deps):
        """Test upload_range processes securities and publishes."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['security_master'].get_securities_in_range.return_value = [('AAPL', 1)]
        daily_deps['validator'].data_exists.return_value = False
        daily_deps['data_collectors'].collect_daily_ticks_range_bulk.return_value = {
            'AAPL': pl.DataFrame({'close': [100.0]})
        }
        daily_deps['data_publishers'].publish_daily_ticks.return_value = {'status': 'success'}

        result = handler.upload_range(2017, 2025)

        assert result['success'] == 1
        daily_deps['data_publishers'].publish_daily_ticks.assert_called_once_with(
            'AAPL', 1,
            daily_deps['data_collectors'].collect_daily_ticks_range_bulk.return_value['AAPL'],
            2017, 2025
        )

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_range_no_securities(self, mock_tqdm, handler, daily_deps):
        """Test upload_range with empty security list."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['security_master'].get_securities_in_range.return_value = []

        result = handler.upload_range(2017, 2025)

        assert result['success'] == 0
        daily_deps['data_publishers'].publish_daily_ticks.assert_not_called()

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_range_overwrite_skips_filter(self, mock_tqdm, handler, daily_deps):
        """Test upload_range with overwrite=True skips filtering."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['security_master'].get_securities_in_range.return_value = [('AAPL', 1)]
        daily_deps['data_collectors'].collect_daily_ticks_range_bulk.return_value = {
            'AAPL': pl.DataFrame({'close': [100.0]})
        }
        daily_deps['data_publishers'].publish_daily_ticks.return_value = {'status': 'success'}

        with patch.object(handler, '_filter_existing') as mock_filter:
            handler.upload_range(2017, 2025, overwrite=True)

        mock_filter.assert_not_called()

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_range_calls_range_bulk(self, mock_tqdm, handler, daily_deps):
        """Test upload_range uses collect_daily_ticks_range_bulk with correct dates."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['security_master'].get_securities_in_range.return_value = [('AAPL', 1)]
        daily_deps['validator'].data_exists.return_value = False
        daily_deps['data_collectors'].collect_daily_ticks_range_bulk.return_value = {
            'AAPL': pl.DataFrame()
        }
        daily_deps['data_publishers'].publish_daily_ticks.return_value = {'status': 'skipped'}

        handler.upload_range(2020, 2023)

        daily_deps['data_collectors'].collect_daily_ticks_range_bulk.assert_called_once_with(
            ['AAPL'], '2020-01-01', '2023-12-31'
        )

    def test_filter_existing(self, handler, daily_deps):
        daily_deps['validator'].data_exists.side_effect = [True, False]
        pbar = MagicMock()

        # sid None -> skipped, sid exists -> canceled, sid new -> fetch
        securities = [('SYM_NONE', None), ('SYM_EXISTS', 1), ('SYM_NEW', 2)]
        result = handler._filter_existing(securities, 2025, pbar)

        assert result == [('SYM_NEW', 2)]
        assert handler.stats['skipped'] == 1
        assert handler.stats['canceled'] == 1

    def test_update_stats_from_result(self, handler):
        handler._update_stats_from_result({'status': 'success'})
        handler._update_stats_from_result({'status': 'canceled'})
        handler._update_stats_from_result({'status': 'skipped'})
        handler._update_stats_from_result({'status': 'failed'})
        handler._update_stats_from_result({})  # defaults to failed

        assert handler.stats == {'success': 1, 'canceled': 1, 'skipped': 1, 'failed': 2}
