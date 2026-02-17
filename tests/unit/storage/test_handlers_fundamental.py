"""Unit tests for FundamentalHandler."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import logging
import polars as pl

from alphalab.storage.handlers.fundamental import FundamentalHandler


# -- Shared fixtures ----------------------------------------------------------


@pytest.fixture
def mock_deps():
    """Common mocked dependencies for fundamental handlers."""
    return {
        'data_publishers': Mock(),
        'data_collectors': Mock(),
        'cik_resolver': Mock(),
        'security_master': Mock(),
        'universe_manager': Mock(),
        'validator': Mock(),
        'sec_rate_limiter': Mock(),
        'logger': Mock(spec=logging.Logger),
    }


# -- FundamentalHandler -------------------------------------------------------


class TestFundamentalHandler:

    @pytest.fixture
    def handler(self, mock_deps):
        return FundamentalHandler(**mock_deps)

    def test_init(self, handler, mock_deps):
        assert handler.publishers is mock_deps['data_publishers']
        assert handler.collectors is mock_deps['data_collectors']
        assert handler.cik_resolver is mock_deps['cik_resolver']
        assert handler.security_master is mock_deps['security_master']
        assert handler.universe_manager is mock_deps['universe_manager']
        assert handler.validator is mock_deps['validator']
        assert handler.sec_rate_limiter is mock_deps['sec_rate_limiter']
        assert handler.stats == {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

    def test_upload_no_symbols_returns_early(self, handler):
        with patch.object(handler, '_prepare_symbols', return_value=([], {}, {}, 0.1)):
            result = handler.upload('2020-01-01', '2020-12-31')

        assert result['success'] == 0
        assert result['failed'] == 0
        handler.logger.warning.assert_called_once()

    @patch('alphalab.storage.handlers.fundamental.Progress')
    @patch('alphalab.storage.handlers.fundamental.ThreadPoolExecutor')
    def test_upload_processes_symbols(self, mock_executor_cls, mock_progress, handler):
        with patch.object(handler, '_prepare_symbols', return_value=(['AAPL'], {'AAPL': '123'}, {'AAPL': 100}, 0.1)):
            mock_future = Mock()
            mock_future.result.return_value = {'status': 'success'}
            mock_executor = MagicMock()
            mock_executor.__enter__.return_value = mock_executor
            mock_executor.submit.return_value = mock_future
            mock_executor_cls.return_value = mock_executor

            mock_ctx = MagicMock()
            mock_progress.return_value.__enter__ = Mock(return_value=mock_ctx)
            mock_progress.return_value.__exit__ = Mock(return_value=False)

            with patch('alphalab.storage.handlers.fundamental.as_completed', return_value=[mock_future]):
                result = handler.upload('2020-01-01', '2020-12-31')

        assert result['success'] == 1

    def test_process_symbol_with_cik_and_security_id(self, handler, mock_deps):
        mock_deps['data_publishers'].publish_fundamental.return_value = {'status': 'success'}

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '0000320193', 100)

        mock_deps['data_publishers'].publish_fundamental.assert_called_once_with(
            sym='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            cik='0000320193',
            security_id=100,
            sec_rate_limiter=mock_deps['sec_rate_limiter'],
        )
        assert result['status'] == 'success'

    def test_process_symbol_without_cik(self, handler, mock_deps):
        mock_deps['cik_resolver'].get_cik.return_value = '0000320193'
        mock_deps['data_publishers'].publish_fundamental.return_value = {'status': 'success'}

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, None, 100)

        mock_deps['cik_resolver'].get_cik.assert_called_once_with('AAPL', '2020-06-30', year=2020)
        assert result['status'] == 'success'

    def test_process_symbol_without_security_id(self, handler, mock_deps):
        """Test that _process_symbol skips when security_id is None."""
        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123', None)

        assert result['status'] == 'skipped'
        mock_deps['data_publishers'].publish_fundamental.assert_not_called()

    def test_process_symbol_skips_existing(self, handler, mock_deps):
        mock_deps['validator'].data_exists.return_value = True
        mock_deps['data_publishers'].publish_fundamental.return_value = {'status': 'success'}

        handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123', 100)

        mock_deps['validator'].data_exists.assert_called_once()
        handler.logger.debug.assert_called()

    def test_prepare_symbols_builds_list(self, handler, mock_deps):
        mock_deps['universe_manager'].load_symbols_for_year.side_effect = [
            ['AAPL', 'MSFT'],
            ['AAPL', 'GOOGL'],
        ]
        mock_deps['cik_resolver'].batch_prefetch_ciks.side_effect = [
            {'AAPL': '1', 'MSFT': '2'},
            {'GOOGL': '3'},
        ]
        mock_deps['security_master'].get_security_id.side_effect = [100, 200, 300]

        symbols, cik_map, security_id_cache, _ = handler._prepare_symbols('2020-01-01', '2021-12-31')

        # AAPL appears in both years but should be deduplicated
        assert set(symbols) == {'AAPL', 'MSFT', 'GOOGL'}
        assert cik_map == {'AAPL': '1', 'MSFT': '2', 'GOOGL': '3'}
        assert len(security_id_cache) == 3

    def test_prepare_symbols_filters_without_cik(self, handler, mock_deps):
        mock_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL', 'NOCIK']
        mock_deps['cik_resolver'].batch_prefetch_ciks.return_value = {
            'AAPL': '1',
            'NOCIK': None,
        }
        mock_deps['security_master'].get_security_id.return_value = 100

        symbols, _, _, _ = handler._prepare_symbols('2020-01-01', '2020-12-31')

        assert 'AAPL' in symbols
        assert 'NOCIK' not in symbols

    def test_prepare_symbols_logs_non_sec_filers(self, handler, mock_deps):
        mock_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL', 'XYZ']
        mock_deps['cik_resolver'].batch_prefetch_ciks.return_value = {
            'AAPL': '1',
            'XYZ': None,
        }
        mock_deps['security_master'].get_security_id.return_value = 100

        handler._prepare_symbols('2020-01-01', '2020-12-31')

        # <=30 non-filers -> logged individually (debug level)
        debug_calls = [str(c) for c in handler.logger.debug.call_args_list]
        assert any('Non-SEC filers' in c for c in debug_calls)

    def test_update_stats_all_statuses(self, handler):
        """Test update_stats_from_result (inherited from BaseHandler)."""
        handler.update_stats_from_result({'status': 'success'})
        handler.update_stats_from_result({'status': 'canceled'})
        handler.update_stats_from_result({'status': 'skipped'})
        handler.update_stats_from_result({'status': 'failed'})
        handler.update_stats_from_result({'status': 'unknown'})  # defaults to failed
        handler.update_stats_from_result({})  # no status -> failed

        assert handler.stats == {'success': 1, 'canceled': 1, 'skipped': 1, 'failed': 3}

    def test_build_result(self, handler):
        import time
        handler.stats = {'success': 5, 'failed': 1, 'skipped': 2, 'canceled': 0}
        start = time.time() - 10

        result = handler._build_result(start, 1.0, 5.0, 10)

        assert result['success'] == 5
        assert result['failed'] == 1
        assert result['prefetch_time'] == 1.0
        assert result['fetch_time'] == 5.0
        assert result['avg_rate'] == 2.0  # 10 / 5.0

    def test_build_result_zero_fetch_time(self, handler):
        import time
        start = time.time()

        result = handler._build_result(start, 0.0, 0.0, 0)

        assert result['avg_rate'] == 0
