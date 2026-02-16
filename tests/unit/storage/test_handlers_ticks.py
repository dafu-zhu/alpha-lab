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
        'universe_manager': Mock(),
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

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_year_with_data(self, mock_tqdm, handler, daily_deps):
        """Test upload_year processes symbols and publishes."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        daily_deps['validator'].data_exists.return_value = False
        daily_deps['data_collectors'].collect_daily_ticks_year_bulk.return_value = {
            'AAPL': pl.DataFrame({'close': [100.0]})
        }
        daily_deps['data_publishers'].publish_daily_ticks.return_value = {'status': 'success'}

        with patch.object(handler, '_build_security_id_cache', return_value={'AAPL': 1}):
            result = handler.upload_year(2020)

        assert result['success'] == 1
        daily_deps['data_publishers'].publish_daily_ticks.assert_called_once_with(
            'AAPL', 2020, 1, daily_deps['data_collectors'].collect_daily_ticks_year_bulk.return_value['AAPL']
        )

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_year_no_symbols(self, mock_tqdm, handler, daily_deps):
        """Test upload_year with empty symbol list."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['universe_manager'].load_symbols_for_year.return_value = []

        with patch.object(handler, '_build_security_id_cache', return_value={}):
            result = handler.upload_year(2020)

        assert result['success'] == 0
        daily_deps['data_publishers'].publish_daily_ticks.assert_not_called()

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_year_overwrite_skips_filter(self, mock_tqdm, handler, daily_deps):
        """Test upload_year with overwrite=True skips filtering."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        daily_deps['data_collectors'].collect_daily_ticks_year_bulk.return_value = {
            'AAPL': pl.DataFrame({'close': [100.0]})
        }
        daily_deps['data_publishers'].publish_daily_ticks.return_value = {'status': 'success'}

        with patch.object(handler, '_build_security_id_cache', return_value={'AAPL': 1}), \
             patch.object(handler, '_filter_existing_symbols') as mock_filter:
            handler.upload_year(2020, overwrite=True)

        mock_filter.assert_not_called()

    def test_build_security_id_cache(self, handler, daily_deps):
        daily_deps['security_master'].get_security_id.side_effect = [100, ValueError('not found')]

        cache = handler._build_security_id_cache(['AAPL', 'BADTICKER'], 2020)

        assert cache == {'AAPL': 100, 'BADTICKER': None}

    def test_filter_existing_symbols(self, handler, daily_deps):
        daily_deps['validator'].data_exists.side_effect = [True, False]
        pbar = MagicMock()

        # sec_id None -> skipped, sec_id exists -> canceled, sec_id new -> fetch
        cache = {'SYM_NONE': None, 'SYM_EXISTS': 1, 'SYM_NEW': 2}
        result = handler._filter_existing_symbols(
            ['SYM_NONE', 'SYM_EXISTS', 'SYM_NEW'], 2020, cache, pbar
        )

        assert result == ['SYM_NEW']
        assert handler.stats['skipped'] == 1
        assert handler.stats['canceled'] == 1

    def test_update_stats_from_result(self, handler):
        handler._update_stats_from_result({'status': 'success'})
        handler._update_stats_from_result({'status': 'canceled'})
        handler._update_stats_from_result({'status': 'skipped'})
        handler._update_stats_from_result({'status': 'failed'})
        handler._update_stats_from_result({})  # defaults to failed

        assert handler.stats == {'success': 1, 'canceled': 1, 'skipped': 1, 'failed': 2}
