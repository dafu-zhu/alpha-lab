"""Unit tests for master.security_master module."""
import datetime as dt
import os
from unittest.mock import Mock, patch

import polars as pl
import pytest
import requests

from alphalab.master.security_master import SecurityMaster, SymbolNormalizer


def _make_security_master(master_tb: pl.DataFrame, **overrides) -> SecurityMaster:
    """Create a SecurityMaster with injected data, bypassing __init__."""
    sm = SecurityMaster.__new__(SecurityMaster)
    sm.master_tb = master_tb
    sm.logger = overrides.pop('logger', Mock())
    for key, value in overrides.items():
        setattr(sm, key, value)
    return sm


class TestSymbolNormalizer:
    """Test SymbolNormalizer class."""

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_initialization(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'BRK.B', 'GOOGL', 'ABC.D']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Verify fetch was called
        mock_fetch.assert_called_once_with(with_filter=True, refresh=False)

        assert 'AAPL' in normalizer.sym_map
        assert 'BRKB' in normalizer.sym_map
        assert 'GOOGL' in normalizer.sym_map
        assert 'ABCD' in normalizer.sym_map

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_initialization_with_security_master(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.security_master == mock_sm

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_simple_match(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('MSFT') == 'MSFT'

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_separator(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B', 'ABC-D']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('BRKB') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK.B') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK-B') == 'BRK.B'
        assert normalizer.to_nasdaq_format('ABCD') == 'ABC-D'

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_not_in_current_list(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('DELISTD') == 'DELISTD'
        assert normalizer.to_nasdaq_format('oldstock') == 'OLDSTOCK'

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_same_security(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.to_nasdaq_format('BRKB', day='2024-01-01') == 'BRK.B'
        assert mock_sm.get_security_id.call_count == 2

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_different_security(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['ABC.D']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.side_effect = ['security_old', 'security_new']

        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.to_nasdaq_format('ABCD', day='2022-01-01') == 'ABCD'

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_validation_error(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.side_effect = ValueError("Symbol not found")

        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.to_nasdaq_format('AAPL', day='2024-01-01') == 'AAPL'

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_empty_symbol(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('') == ''
        assert normalizer.to_nasdaq_format(None) is None

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_no_date_context(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.to_nasdaq_format('BRKB') == 'BRK.B'
        mock_sm.get_security_id.assert_not_called()

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_batch_normalize(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        symbols = ['aapl', 'BRKB', 'googl']
        result = normalizer.batch_normalize(symbols)

        assert result == ['AAPL', 'BRK.B', 'GOOGL']

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_batch_normalize_with_date(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        symbols = ['AAPL', 'BRKB']
        result = normalizer.batch_normalize(symbols, day='2024-01-01')

        assert result == ['AAPL', 'BRK.B']

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_day_no_security_master(self, mock_logger, mock_fetch):
        """Date context without SecurityMaster uses Nasdaq format."""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer(security_master=None)

        assert normalizer.to_nasdaq_format('BRKB', day='2024-01-01') == 'BRK.B'

    def test_to_crsp_format(self):
        assert SymbolNormalizer.to_crsp_format('BRK.B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('BRK-B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('ABC.D.E') == 'ABCDE'
        assert SymbolNormalizer.to_crsp_format('AAPL') == 'AAPL'
        assert SymbolNormalizer.to_crsp_format('aapl') == 'AAPL'

    def test_to_sec_format(self):
        assert SymbolNormalizer.to_sec_format('BRK.B') == 'BRK-B'
        assert SymbolNormalizer.to_sec_format('aapl') == 'AAPL'

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_handles_non_string_tickers(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', None, 'MSFT', 'GOOGL']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert 'AAPL' in normalizer.sym_map
        assert 'MSFT' in normalizer.sym_map
        assert 'GOOGL' in normalizer.sym_map


class TestSymbolNormalizerEdgeCases:
    """Test edge cases for SymbolNormalizer."""

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_case_insensitivity(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('AaPl') == 'AAPL'
        assert normalizer.to_nasdaq_format('brkb') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BrK.b') == 'BRK.B'

    @patch('alphalab.master.security_master.fetch_all_stocks')
    @patch('alphalab.master.security_master.setup_logger')
    def test_multiple_separators(self, mock_logger, mock_fetch):
        mock_stocks = pl.DataFrame({'Ticker': ['A.B.C', 'X-Y-Z']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('ABC') == 'A.B.C'
        assert normalizer.to_nasdaq_format('XYZ') == 'X-Y-Z'


class TestSecurityMaster:
    """Test SecurityMaster core behaviors with injected data."""

    _SINGLE_ROW = pl.DataFrame({
        'security_id': [101],
        'symbol': ['AAA'],
        'start_date': [dt.date(2020, 1, 1)],
        'end_date': [dt.date(2020, 12, 31)],
    })

    def test_get_security_id_exact_match(self):
        sm = _make_security_master(self._SINGLE_ROW, auto_resolve=Mock(return_value=999))

        assert sm.get_security_id('AAA', '2020-06-30', auto_resolve=True) == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_no_match_no_auto_resolve(self):
        sm = _make_security_master(self._SINGLE_ROW)

        with pytest.raises(ValueError, match="Symbol BBB not found"):
            sm.get_security_id('BBB', '2020-06-30', auto_resolve=False)

    def test_get_security_id_no_match_auto_resolve(self):
        sm = _make_security_master(self._SINGLE_ROW, auto_resolve=Mock(return_value=555))

        assert sm.get_security_id('BBB', '2020-06-30', auto_resolve=True) == 555
        sm.auto_resolve.assert_called_once_with('BBB', '2020-06-30')

    def test_get_security_id_exact_match_no_auto_resolve_flag(self):
        sm = _make_security_master(self._SINGLE_ROW, auto_resolve=Mock(return_value=999))

        assert sm.get_security_id('AAA', '2020-06-30', auto_resolve=False) == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_rejects_none_security_id(self):
        master_tb = pl.DataFrame({
            'security_id': [None],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        with pytest.raises(ValueError, match="security_id is None"):
            sm.get_security_id('AAA', '2020-06-30', auto_resolve=False)

    def test_auto_resolve_selects_closest_symbol_usage(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'company': ['OldCo', 'OldCo', 'NewCo'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2018, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2022, 12, 31), dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(return_value="info"))

        assert sm.auto_resolve('AAA', '2020-06-15') == 2

    def test_auto_resolve_symbol_never_existed(self):
        master_tb = pl.DataFrame({
            'security_id': [1],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        with pytest.raises(ValueError, match="never existed"):
            sm.auto_resolve('ZZZ', '2020-06-15')

    def test_sid_to_info(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        assert sm.sid_to_info(101, '2020-06-30', info='company') == 'TestCo'

    def test_get_symbol_history(self):
        master_tb = pl.DataFrame({
            'security_id': [101, 101],
            'symbol': ['AAA', 'BBB'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        assert set(sm.get_symbol_history(101)) == {
            ('AAA', '2020-01-01', '2020-12-31'),
            ('BBB', '2021-01-01', '2021-12-31'),
        }

    def test_auto_resolve_filters_null_security_ids(self):
        master_tb = pl.DataFrame({
            'security_id': [None, 101],
            'symbol': ['AAA', 'AAA'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(return_value="info"))

        assert sm.auto_resolve('AAA', '2020-06-30') == 101
        sm.logger.warning.assert_not_called()

    def test_auto_resolve_logs_error_on_sid_to_info_failure(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(side_effect=RuntimeError("boom")))

        assert sm.auto_resolve('AAA', '2020-06-30') == 101
        sm.logger.debug.assert_called()

    def test_auto_resolve_no_active_security(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        with pytest.raises(ValueError, match="was not active"):
            sm.auto_resolve('AAA', '2020-06-30')

    def test_auto_resolve_multiple_candidates_selects_closest(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 6, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(return_value="info"))

        assert sm.auto_resolve('AAA', '2020-06-15') == 2
        sm.logger.debug.assert_called()

    def test_ensure_schema_drops_cusip(self):
        df = pl.DataFrame({
            'security_id': [1],
            'symbol': ['AAA'],
            'cusip': ['12345678'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)],
        })
        result = SecurityMaster._ensure_schema(df)
        assert 'cusip' not in result.columns
        assert 'exchange' in result.columns

    def test_ensure_schema_adds_missing_columns(self):
        df = pl.DataFrame({
            'security_id': [1],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)],
        })
        result = SecurityMaster._ensure_schema(df)
        for col in ('exchange', 'sector', 'industry', 'subindustry'):
            assert col in result.columns


class TestGetSecuritiesInRange:
    """Test SecurityMaster.get_securities_in_range()."""

    def test_returns_active_securities(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 2, 3],
            'symbol': ['AAPL', 'MSFT', 'OLD'],
            'start_date': [dt.date(2015, 1, 1), dt.date(2010, 1, 1), dt.date(2005, 1, 1)],
            'end_date': [dt.date(2025, 12, 31), dt.date(2025, 12, 31), dt.date(2010, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        result = sm.get_securities_in_range(2017, 2025)

        assert len(result) == 2
        assert {r[0] for r in result} == {'AAPL', 'MSFT'}

    def test_picks_latest_symbol_for_renamed_security(self):
        """For FB->META, should return META (latest symbol)."""
        master_tb = pl.DataFrame({
            'security_id': [100, 100],
            'symbol': ['FB', 'META'],
            'start_date': [dt.date(2012, 5, 18), dt.date(2022, 6, 9)],
            'end_date': [dt.date(2022, 6, 8), dt.date(2025, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        result = sm.get_securities_in_range(2017, 2025)

        assert len(result) == 1
        assert result[0] == ('META', 100)

    def test_includes_delisted_stocks_in_range(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 2],
            'symbol': ['AAPL', 'DISH'],
            'start_date': [dt.date(2015, 1, 1), dt.date(2010, 1, 1)],
            'end_date': [dt.date(2025, 12, 31), dt.date(2023, 12, 29)],
        })
        sm = _make_security_master(master_tb)

        result = sm.get_securities_in_range(2017, 2025)

        assert len(result) == 2
        assert 'DISH' in {r[0] for r in result}

    def test_excludes_securities_outside_range(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 2],
            'symbol': ['AAPL', 'OLD'],
            'start_date': [dt.date(2015, 1, 1), dt.date(2000, 1, 1)],
            'end_date': [dt.date(2025, 12, 31), dt.date(2016, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        result = sm.get_securities_in_range(2017, 2025)

        assert len(result) == 1
        assert result[0][0] == 'AAPL'

    def test_sorted_by_security_id(self):
        master_tb = pl.DataFrame({
            'security_id': [300, 100, 200],
            'symbol': ['C', 'A', 'B'],
            'start_date': [dt.date(2015, 1, 1)] * 3,
            'end_date': [dt.date(2025, 12, 31)] * 3,
        })
        sm = _make_security_master(master_tb)

        result = sm.get_securities_in_range(2017, 2025)

        assert [r[1] for r in result] == [100, 200, 300]


class TestSecurityMasterInitialization:
    """Test SecurityMaster initialization"""

    _MOCK_MASTER = pl.DataFrame({
        'security_id': [1],
        'symbol': ['AAA'],
        'company': ['AAA Corp'],
        'permno': [1],
        'cik': ['0000000001'],
        'start_date': [dt.date(2020, 1, 1)],
        'end_date': [dt.date(2020, 12, 31)]
    })

    @patch('alphalab.master.security_master.SecurityMaster._load_from_local')
    @patch('alphalab.master.security_master.setup_logger')
    def test_init_loads_from_local(self, mock_logger, mock_load_local):
        """Test initialization loads from local parquet."""
        mock_load_local.return_value = self._MOCK_MASTER

        sm = SecurityMaster()

        assert sm.master_tb is not None
        assert len(sm.master_tb) == 1
        mock_load_local.assert_called_once()

    @patch('alphalab.master.security_master.SecurityMaster._load_from_local')
    @patch('alphalab.master.security_master.setup_logger')
    def test_init_creates_logger(self, mock_logger, mock_load_local):
        """Test that initialization creates a logger"""
        mock_load_local.return_value = self._MOCK_MASTER

        sm = SecurityMaster()

        # Verify logger setup was called
        mock_logger.assert_called_once()
        assert sm.logger is not None

    @patch('alphalab.master.security_master.SecurityMaster._load_from_local')
    @patch('alphalab.master.security_master.setup_logger')
    def test_init_sets_gics_mapping_cache_to_none(self, mock_logger, mock_load_local):
        """Test that _gics_mapping is initialized to None"""
        mock_load_local.return_value = self._MOCK_MASTER

        sm = SecurityMaster()

        assert sm._gics_mapping is None

    def test_init_raises_file_not_found(self, tmp_path):
        """Test that init raises FileNotFoundError when parquet doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Security master not found"):
            SecurityMaster(local_path=tmp_path / "nonexistent.parquet")


class TestSecurityMasterAutoResolve:
    """Test SecurityMaster auto_resolve edge cases."""

    def test_auto_resolve_with_null_candidates_defensive_code(self):
        master_tb = pl.DataFrame({
            'security_id': [None, None, 101],
            'symbol': ['AAA', 'AAA', 'AAA'],
            'company': ['AAA Corp 1', 'AAA Corp 2', 'AAA Corp 3'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 6, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(return_value="info"))

        assert sm.auto_resolve('AAA', '2020-06-30') == 101
        sm.logger.warning.assert_not_called()

    def test_auto_resolve_verifies_null_filter_works(self):
        """Verify that nulls are filtered before the loop."""
        master_tb = pl.DataFrame({
            'security_id': [None, None, 101],
            'symbol': ['AAA', 'AAA', 'AAA'],
            'company': ['AAA Corp', 'AAA Corp', 'AAA Corp'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(return_value="info"))

        assert sm.auto_resolve('AAA', '2020-06-30') == 101
        sm.logger.warning.assert_not_called()

        candidates = (
            master_tb.filter(pl.col('symbol').eq('AAA'))
            .select('security_id')
            .unique()
            .filter(pl.col('security_id').is_not_null())
        )
        assert candidates.height == 1
        assert candidates['security_id'][0] == 101

    def test_auto_resolve_date_before_symbol_start(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2, 2],
            'symbol': ['BBB', 'AAA', 'CCC', 'AAA'],
            'company': ['OldCo', 'OldCo', 'NewCo', 'NewCo'],
            'start_date': [dt.date(2018, 1, 1), dt.date(2020, 1, 1), dt.date(2018, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2020, 12, 31), dt.date(2022, 12, 31), dt.date(2022, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(return_value="info"))

        assert sm.auto_resolve('AAA', '2019-06-15') == 1

    def test_auto_resolve_date_after_symbol_end(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2, 2],
            'symbol': ['AAA', 'BBB', 'AAA', 'CCC'],
            'company': ['OldCo', 'OldCo', 'NewCo', 'NewCo'],
            'start_date': [dt.date(2018, 1, 1), dt.date(2020, 1, 1), dt.date(2019, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2020, 12, 31), dt.date(2019, 6, 30), dt.date(2020, 12, 31)],
        })
        sm = _make_security_master(master_tb, sid_to_info=Mock(return_value="info"))

        assert sm.auto_resolve('AAA', '2020-06-15') == 1


class TestSecurityMasterClose:
    """Test SecurityMaster close method."""

    def test_close_is_noop(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.close()  # Should not raise


class TestSecurityMasterSecOperations:
    """Test SecurityMaster SEC operations."""

    def test_fetch_sec_exchange_mapping(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_response = Mock()
        mock_response.json.return_value = {
            'fields': ['cik', 'name', 'ticker', 'exchange'],
            'data': [
                [320193, 'Apple Inc', 'AAPL', 'Nasdaq'],
                [1018724, 'Amazon.com Inc', 'AMZN', 'Nasdaq']
            ]
        }

        with patch('alphalab.master.security_master.requests.get', return_value=mock_response):
            result = sm._fetch_sec_exchange_mapping()

        assert len(result) == 2
        assert 'ticker' in result.columns
        assert 'cik' in result.columns
        assert 'company' in result.columns
        assert 'exchange' in result.columns

        assert result['ticker'][0] == 'AAPL'

    def test_auto_resolve_null_candidates_warning(self):
        """Raises error when security not active on query date."""
        master_tb = pl.DataFrame({
            'security_id': [101, 102],
            'symbol': ['AAA', 'AAA'],
            'permno': [None, None],
            'cik': [None, None],
            'start_date': [dt.date(2019, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2021, 12, 31)],
        })
        sm = _make_security_master(master_tb)

        with pytest.raises(ValueError, match="was not active on"):
            sm.auto_resolve('AAA', '2020-06-30')

    def test_update_extends_end_dates(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'permno': [555],
            'cik': ['0000320193'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)],
            'exchange': ['Nasdaq'],
            'sector': ['Information Technology'],
            'industry': ['Technology Hardware & Equipment'],
            'subindustry': ['Technology Hardware, Storage & Peripherals'],
        })
        sm = _make_security_master(master_tb, _gics_mapping=None)

        import pandas as pd
        mock_nasdaq = pd.DataFrame({'Ticker': ['AAPL']})
        sec_df = pl.DataFrame({
            'ticker': ['AAPL'],
            'cik': ['0000320193'],
            'company': ['Apple Inc'],
            'exchange': ['Nasdaq'],
        })

        with patch.object(sm, '_fetch_sec_exchange_mapping', return_value=sec_df), \
             patch.object(sm, '_fetch_nasdaq_universe', return_value={'AAPL'}), \
             patch.object(sm, '_load_prev_universe', return_value=({'AAPL'}, '2025-02-01')), \
             patch.object(sm, '_save_prev_universe'), \
             patch.object(sm, 'save_local'):
            stats = sm.update()

        assert stats['extended'] == 1
        assert stats['added'] == 0

        updated_row = sm.master_tb.filter(pl.col('security_id') == 101)
        assert updated_row['end_date'][0] > dt.date(2024, 12, 31)

    def test_update_adds_new_ipos_with_gics(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'permno': [555],
            'cik': ['0000320193'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)],
            'exchange': ['Nasdaq'],
            'sector': ['Information Technology'],
            'industry': ['Technology Hardware & Equipment'],
            'subindustry': ['Technology Hardware, Storage & Peripherals'],
        })
        sm = _make_security_master(master_tb, _gics_mapping=None)

        sec_df = pl.DataFrame({
            'ticker': ['AAPL', 'NEWIPO'],
            'cik': ['0000320193', '9999999999'],
            'company': ['Apple Inc', 'New IPO Corp'],
            'exchange': ['Nasdaq', 'NYSE'],
        })

        # Mock yfinance returning classification
        yf_meta = {'NEWIPO': {'sector': 'Technology', 'industry': 'Semiconductors'}}

        with patch.object(sm, '_fetch_sec_exchange_mapping', return_value=sec_df), \
             patch.object(sm, '_fetch_nasdaq_universe', return_value={'AAPL', 'NEWIPO'}), \
             patch.object(sm, '_load_prev_universe', return_value=({'AAPL'}, '2025-02-01')), \
             patch.object(sm, '_fetch_openfigi_mapping', return_value={'NEWIPO': 'FIGI_NEW'}), \
             patch.object(sm, '_fetch_yfinance_metadata', return_value=yf_meta), \
             patch.object(sm, '_map_to_gics', return_value={
                 'sector': 'Information Technology',
                 'industry': 'Semiconductors & Semiconductor Equipment',
                 'subindustry': 'Semiconductors',
             }), \
             patch.object(sm, '_save_prev_universe'), \
             patch.object(sm, 'save_local'):
            stats = sm.update()

        assert stats['added'] == 1
        assert stats['extended'] == 1

        new_rows = sm.master_tb.filter(pl.col('cik') == '9999999999')
        assert len(new_rows) == 1
        assert new_rows['symbol'][0] == 'NEWIPO'
        assert new_rows['exchange'][0] == 'NYSE'
        assert new_rows['sector'][0] == 'Information Technology'


class TestGICSMapping:
    """Test Morningstar to GICS mapping functionality."""

    def test_map_to_gics_industry_match(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._gics_mapping = {
            'sectors': {'Technology': 'Information Technology'},
            'industries': {
                'Semiconductors': {
                    'gics_sector': 'Information Technology',
                    'gics_industry_group': 'Semiconductors & Semiconductor Equipment',
                    'gics_sub_industry': 'Semiconductors',
                }
            }
        }

        result = sm._map_to_gics('Technology', 'Semiconductors')

        assert result['sector'] == 'Information Technology'
        assert result['industry'] == 'Semiconductors & Semiconductor Equipment'
        assert result['subindustry'] == 'Semiconductors'

    def test_map_to_gics_sector_fallback(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._gics_mapping = {
            'sectors': {'Technology': 'Information Technology'},
            'industries': {}
        }

        result = sm._map_to_gics('Technology', 'Unknown Industry')

        assert result['sector'] == 'Information Technology'
        assert result['industry'] is None
        assert result['subindustry'] is None

    def test_map_to_gics_no_match(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._gics_mapping = {'sectors': {}, 'industries': {}}

        result = sm._map_to_gics('Unknown', 'Unknown')

        assert result == {'sector': None, 'industry': None, 'subindustry': None}

    def test_load_gics_mapping_caches(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        cached = {'sectors': {'A': 'B'}, 'industries': {}}
        sm._gics_mapping = cached

        result = sm._load_gics_mapping()

        assert result is cached

    def test_load_gics_mapping_missing_file(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._gics_mapping = None

        with patch('alphalab.master.security_master.GICS_MAPPING_PATH') as mock_path:
            mock_path.exists.return_value = False
            result = sm._load_gics_mapping()

        assert result == {'sectors': {}, 'industries': {}}


class TestYfinanceMetadata:
    """Test yfinance metadata fetching."""

    def test_fetch_yfinance_metadata_success(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_ticker = Mock()
        mock_ticker.info = {'sector': 'Technology', 'industry': 'Semiconductors'}

        with patch.dict('sys.modules', {'yfinance': Mock()}):
            import sys
            yf_mock = sys.modules['yfinance']
            yf_mock.Ticker.return_value = mock_ticker

            with patch('alphalab.master.security_master.time.sleep'):
                result = sm._fetch_yfinance_metadata(['AAPL'])

        assert 'AAPL' in result
        assert result['AAPL']['sector'] == 'Technology'

    def test_fetch_yfinance_metadata_import_error(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        with patch.dict('sys.modules', {'yfinance': None}):
            # Force ImportError
            with patch('builtins.__import__', side_effect=ImportError("no yfinance")):
                result = sm._fetch_yfinance_metadata(['AAPL'])

        assert result == {}


class TestOpenFIGIIntegration:
    """Test OpenFIGI integration methods."""

    def test_fetch_openfigi_mapping_success(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [
            {"data": [{"shareClassFIGI": "BBG001S5N8V8"}]},
            {"data": [{"shareClassFIGI": "BBG000B9XRY4"}]},
            {"error": "No identifier found."}
        ]

        with patch('alphalab.master.security_master.requests.post', return_value=mock_response):
            with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                result = sm._fetch_openfigi_mapping(['AAPL', 'MSFT', 'UNKNOWN'])

        assert result['AAPL'] == 'BBG001S5N8V8'
        assert result['MSFT'] == 'BBG000B9XRY4'
        assert result['UNKNOWN'] is None

    def test_fetch_openfigi_mapping_api_error(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        with patch('alphalab.master.security_master.requests.post') as mock_post:
            mock_post.side_effect = Exception("API error")
            with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                result = sm._fetch_openfigi_mapping(['AAPL'])

        assert result['AAPL'] is None
        sm.logger.error.assert_called()

    def test_fetch_openfigi_mapping_batching(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        # Create 250 tickers to test batching (with API key: 100 per batch = 3 batches)
        tickers = [f'SYM{i:03d}' for i in range(250)]

        call_count = 0
        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            batch_size = len(kwargs.get('json', []))
            mock_resp.json.return_value = [
                {"data": [{"shareClassFIGI": f"FIGI{i}"}]}
                for i in range(batch_size)
            ]
            return mock_resp

        with patch.dict(os.environ, {'OPENFIGI_API_KEY': 'test-key'}):
            with patch('alphalab.master.security_master.requests.post', side_effect=mock_post):
                with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                    mock_rl.return_value.acquire = Mock()
                    result = sm._fetch_openfigi_mapping(tickers)

        assert call_count == 3
        assert len(result) == 250

    def test_fetch_openfigi_uses_api_key_when_available(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_response = Mock()
        mock_response.json.return_value = [{"data": [{"shareClassFIGI": "FIGI123"}]}]

        with patch.dict(os.environ, {'OPENFIGI_API_KEY': 'test-key'}):
            with patch('alphalab.master.security_master.requests.post', return_value=mock_response) as mock_post:
                with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                    mock_rl.return_value.acquire = Mock()
                    sm._fetch_openfigi_mapping(['AAPL'])

        call_args = mock_post.call_args
        headers = call_args.kwargs.get('headers', {})
        assert headers.get('X-OPENFIGI-APIKEY') == 'test-key'


class TestNasdaqUniverse:
    """Test Nasdaq universe fetching."""

    @patch('alphalab.master.security_master.fetch_all_stocks')
    def test_fetch_nasdaq_universe_success(self, mock_fetch):
        import pandas as pd
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_fetch.return_value = pd.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})

        result = sm._fetch_nasdaq_universe()

        assert result == {'AAPL', 'MSFT', 'GOOGL'}
        mock_fetch.assert_called_once_with(with_filter=True, refresh=True, logger=sm.logger)

    @patch('alphalab.master.security_master.fetch_all_stocks')
    def test_fetch_nasdaq_universe_error(self, mock_fetch):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_fetch.side_effect = Exception("FTP error")

        result = sm._fetch_nasdaq_universe()

        assert result == set()
        sm.logger.error.assert_called()


class TestRebrandDetection:
    """Test rebrand detection logic."""

    def test_detect_rebrands_finds_match(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        disappeared = {'FB'}
        appeared = {'META'}
        figi_mapping = {
            'FB': 'BBG000MM2P62',
            'META': 'BBG000MM2P62'  # Same FIGI = rebrand
        }

        result = sm._detect_rebrands(disappeared, appeared, figi_mapping)

        assert len(result) == 1
        assert result[0] == ('FB', 'META', 'BBG000MM2P62')

    def test_detect_rebrands_no_match(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        disappeared = {'OLDCO'}
        appeared = {'NEWCO'}
        figi_mapping = {
            'OLDCO': 'FIGI_OLD',
            'NEWCO': 'FIGI_NEW'  # Different FIGIs
        }

        result = sm._detect_rebrands(disappeared, appeared, figi_mapping)

        assert len(result) == 0

    def test_detect_rebrands_missing_figi(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        disappeared = {'OLD'}
        appeared = {'NEW'}
        figi_mapping = {
            'OLD': None,  # No FIGI
            'NEW': 'FIGI_NEW'
        }

        result = sm._detect_rebrands(disappeared, appeared, figi_mapping)

        assert len(result) == 0


class TestOpenFIGIRetryBehavior:
    """Test OpenFIGI retry and backoff behavior."""

    def test_fetch_openfigi_429_retries_with_backoff(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = Mock()
            # First 2 calls: 429
            if call_count <= 2:
                mock_resp.status_code = 429
                return mock_resp
            # Third call: success
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            mock_resp.json.return_value = [{"data": [{"shareClassFIGI": "FIGI1"}]}]
            return mock_resp

        with patch('alphalab.master.security_master.requests.post', side_effect=mock_post):
            with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('alphalab.master.security_master.time.sleep') as mock_sleep:
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        assert call_count == 3
        assert mock_sleep.call_count == 2
        assert result['AAPL'] == 'FIGI1'

    def test_fetch_openfigi_5xx_retries(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = Mock()
            if call_count == 1:
                mock_resp.status_code = 503
                return mock_resp
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            mock_resp.json.return_value = [{"data": [{"shareClassFIGI": "FIGI1"}]}]
            return mock_resp

        with patch('alphalab.master.security_master.requests.post', side_effect=mock_post):
            with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('alphalab.master.security_master.time.sleep'):
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        assert call_count == 2
        assert result['AAPL'] == 'FIGI1'

    def test_fetch_openfigi_exhausts_retries(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        def mock_post(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 500
            return mock_resp

        with patch('alphalab.master.security_master.requests.post', side_effect=mock_post):
            with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('alphalab.master.security_master.time.sleep'):
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        assert result['AAPL'] is None
        sm.logger.warning.assert_called()

    def test_fetch_openfigi_summary_logging(self):
        """Test that OpenFIGI logs final summary (progress is now via tqdm)."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        # With API key: 100 per batch. 1500 tickers = 15 batches
        tickers = [f'SYM{i:04d}' for i in range(1500)]

        def mock_post(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            batch_size = len(kwargs.get('json', []))
            mock_resp.json.return_value = [
                {"data": [{"shareClassFIGI": f"FIGI{i}"}]}
                for i in range(batch_size)
            ]
            return mock_resp

        with patch.dict(os.environ, {'OPENFIGI_API_KEY': 'test-key'}):
            with patch('alphalab.master.security_master.requests.post', side_effect=mock_post):
                with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                    mock_rl.return_value.acquire = Mock()
                    result = sm._fetch_openfigi_mapping(tickers)

        # Verify final summary log was called (now at debug level)
        debug_calls = [str(call) for call in sm.logger.debug.call_args_list]
        summary_logs = [c for c in debug_calls if 'OpenFIGI' in c and 'mapped' in c]
        assert len(summary_logs) >= 1

    def test_fetch_openfigi_request_exception_retries(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.RequestException("Connection error")
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            mock_resp.json.return_value = [{"data": [{"shareClassFIGI": "FIGI1"}]}]
            return mock_resp

        with patch('alphalab.master.security_master.requests.post', side_effect=mock_post):
            with patch('alphalab.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('alphalab.master.security_master.time.sleep'):
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        assert call_count == 2
        assert result['AAPL'] == 'FIGI1'
