"""
Unit tests for master.security_master module
Tests symbol normalization and security master functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import pandas as pd
import datetime as dt
import os
from quantdl.master.security_master import SymbolNormalizer, SecurityMaster


class TestSymbolNormalizer:
    """Test SymbolNormalizer class"""

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_initialization(self, mock_logger, mock_fetch):
        """Test SymbolNormalizer initialization"""
        # Mock current stock list
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'BRK.B', 'GOOGL', 'ABC.D']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Verify fetch was called
        mock_fetch.assert_called_once_with(with_filter=True, refresh=False)

        # Verify symbol map was created
        assert 'AAPL' in normalizer.sym_map
        assert 'BRKB' in normalizer.sym_map  # Dots removed
        assert 'GOOGL' in normalizer.sym_map
        assert 'ABCD' in normalizer.sym_map  # Dots removed

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_initialization_with_security_master(self, mock_logger, mock_fetch):
        """Test initialization with SecurityMaster instance"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.security_master == mock_sm

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_simple_match(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with simple symbol match"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Test uppercase conversion
        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('MSFT') == 'MSFT'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_separator(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with symbols containing separators"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B', 'ABC-D']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # BRKB should normalize to BRK.B
        assert normalizer.to_nasdaq_format('BRKB') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK.B') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK-B') == 'BRK.B'

        # ABCD should normalize to ABC-D
        assert normalizer.to_nasdaq_format('ABCD') == 'ABC-D'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_not_in_current_list(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with delisted symbol not in current list"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Symbol not in current list should return as-is (uppercased)
        assert normalizer.to_nasdaq_format('DELISTD') == 'DELISTD'
        assert normalizer.to_nasdaq_format('oldstock') == 'OLDSTOCK'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_same_security(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with SecurityMaster validation (same security)"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        # Mock same security_id for both dates
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('BRKB', day='2024-01-01')

        # Should return Nasdaq format since same security
        assert result == 'BRK.B'

        # Verify get_security_id was called twice
        assert mock_sm.get_security_id.call_count == 2

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_different_security(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with SecurityMaster validation (different security)"""
        mock_stocks = pl.DataFrame({'Ticker': ['ABC.D']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        # Mock different security_ids
        mock_sm.get_security_id.side_effect = ['security_old', 'security_new']

        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('ABCD', day='2022-01-01')

        # Should return original format since different security
        assert result == 'ABCD'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_validation_error(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format when validation raises error"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.side_effect = ValueError("Symbol not found")

        normalizer = SymbolNormalizer(security_master=mock_sm)

        # Should return original format when validation fails
        result = normalizer.to_nasdaq_format('AAPL', day='2024-01-01')
        assert result == 'AAPL'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_empty_symbol(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with empty symbol"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('') == ''
        assert normalizer.to_nasdaq_format(None) is None

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_no_date_context(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format without date context"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('BRKB')

        # Should return Nasdaq format without validation
        assert result == 'BRK.B'

        # Verify get_security_id was NOT called
        mock_sm.get_security_id.assert_not_called()

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_batch_normalize(self, mock_logger, mock_fetch):
        """Test batch_normalize"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        symbols = ['aapl', 'BRKB', 'googl']
        result = normalizer.batch_normalize(symbols)

        assert result == ['AAPL', 'BRK.B', 'GOOGL']

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_batch_normalize_with_date(self, mock_logger, mock_fetch):
        """Test batch_normalize with date context"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        symbols = ['AAPL', 'BRKB']
        result = normalizer.batch_normalize(symbols, day='2024-01-01')

        assert result == ['AAPL', 'BRK.B']

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_day_no_security_master(self, mock_logger, mock_fetch):
        """Date context without SecurityMaster uses Nasdaq format."""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer(security_master=None)

        assert normalizer.to_nasdaq_format('BRKB', day='2024-01-01') == 'BRK.B'

    def test_to_crsp_format(self):
        """Test to_crsp_format static method"""
        # Test various input formats
        assert SymbolNormalizer.to_crsp_format('BRK.B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('BRK-B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('ABC.D.E') == 'ABCDE'
        assert SymbolNormalizer.to_crsp_format('AAPL') == 'AAPL'
        assert SymbolNormalizer.to_crsp_format('aapl') == 'AAPL'

    def test_to_sec_format(self):
        """Test to_sec_format static method"""
        assert SymbolNormalizer.to_sec_format('BRK.B') == 'BRK-B'
        assert SymbolNormalizer.to_sec_format('aapl') == 'AAPL'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_handles_non_string_tickers(self, mock_logger, mock_fetch):
        """Test that initialization handles non-string tickers gracefully"""
        # Include NaN and None values
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', None, 'MSFT', 'GOOGL']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Should skip non-string values
        assert 'AAPL' in normalizer.sym_map
        assert 'MSFT' in normalizer.sym_map
        assert 'GOOGL' in normalizer.sym_map
        # None should not cause issues


class TestSymbolNormalizerEdgeCases:
    """Test edge cases for SymbolNormalizer"""

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_case_insensitivity(self, mock_logger, mock_fetch):
        """Test that symbol matching is case-insensitive"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('AaPl') == 'AAPL'
        assert normalizer.to_nasdaq_format('brkb') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BrK.b') == 'BRK.B'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_multiple_separators(self, mock_logger, mock_fetch):
        """Test symbols with multiple separators"""
        mock_stocks = pl.DataFrame({'Ticker': ['A.B.C', 'X-Y-Z']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('ABC') == 'A.B.C'
        assert normalizer.to_nasdaq_format('XYZ') == 'X-Y-Z'


class TestSecurityMaster:
    """Test SecurityMaster core behaviors with injected data"""

    def test_get_security_id_exact_match(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=999)

        result = sm.get_security_id('AAA', '2020-06-30', auto_resolve=True)

        assert result == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_no_match_no_auto_resolve(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="Symbol BBB not found"):
            sm.get_security_id('BBB', '2020-06-30', auto_resolve=False)

    def test_get_security_id_no_match_auto_resolve(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=555)

        result = sm.get_security_id('BBB', '2020-06-30', auto_resolve=True)

        assert result == 555
        sm.auto_resolve.assert_called_once_with('BBB', '2020-06-30')

    def test_get_security_id_exact_match_no_auto_resolve_flag(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=999)

        result = sm.get_security_id('AAA', '2020-06-30', auto_resolve=False)

        assert result == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_rejects_none_security_id(self):
        master_tb = pl.DataFrame({
            'security_id': [None],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="security_id is None"):
            sm.get_security_id('AAA', '2020-06-30', auto_resolve=False)

    def test_auto_resolve_selects_closest_symbol_usage(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'company': ['OldCo', 'OldCo', 'NewCo'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2018, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2022, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-15')

        assert result == 2

    def test_auto_resolve_symbol_never_existed(self):
        master_tb = pl.DataFrame({
            'security_id': [1],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="never existed"):
            sm.auto_resolve('ZZZ', '2020-06-15')

    def test_sid_to_permno(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [555]
        }))

        result = sm.sid_to_permno(101)

        assert result == 555
    
    def test_sid_to_permno_none(self):
        sm = SecurityMaster.__new__(SecurityMaster)

        with pytest.raises(ValueError, match="security_id is None"):
            sm.sid_to_permno(None)

    def test_sid_to_info(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb

        result = sm.sid_to_info(101, '2020-06-30', info='company')

        assert result == 'TestCo'

    def test_get_symbol_history(self):
        master_tb = pl.DataFrame({
            'security_id': [101, 101],
            'symbol': ['AAA', 'BBB'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_table = Mock(return_value=master_tb)

        result = sm.get_symbol_history(101)

        assert set(result) == {
            ('AAA', '2020-01-01', '2020-12-31'),
            ('BBB', '2021-01-01', '2021-12-31')
        }

    def test_auto_resolve_filters_null_security_ids(self):
        master_tb = pl.DataFrame({
            'security_id': [None, 101],
            'symbol': ['AAA', 'AAA'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-30')

        assert result == 101
        sm.logger.warning.assert_not_called()

    def test_auto_resolve_logs_error_on_sid_to_info_failure(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(side_effect=RuntimeError("boom"))

        result = sm.auto_resolve('AAA', '2020-06-30')

        assert result == 101
        sm.logger.error.assert_called_once()

    def test_auto_resolve_no_active_security(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="was not active"):
            sm.auto_resolve('AAA', '2020-06-30')

    def test_auto_resolve_multiple_candidates_selects_closest(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 6, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-15')

        assert result == 2
        sm.logger.info.assert_called()

    def test_fetch_sec_cik_mapping_cached(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        cached = pl.DataFrame({'ticker': ['AAA'], 'cik': ['0000000001']})
        sm._sec_cik_cache = cached

        with patch('quantdl.master.security_master.requests.get') as mock_get:
            result = sm._fetch_sec_cik_mapping()

        assert result is cached
        mock_get.assert_not_called()

    def test_fetch_sec_cik_mapping_success(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "BRK.B", "cik_str": 123456},
            "1": {"ticker": "AAPL", "cik_str": 320193}
        }

        with patch('quantdl.master.security_master.requests.get', return_value=mock_response) as mock_get:
            result = sm._fetch_sec_cik_mapping()

        assert not result.is_empty()
        assert sm._sec_cik_cache is result
        mock_get.assert_called_once()

    def test_fetch_sec_cik_mapping_failure(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        with patch('quantdl.master.security_master.requests.get', side_effect=Exception("boom")):
            result = sm._fetch_sec_cik_mapping()

        assert result.is_empty()
        assert set(result.columns) == {"ticker", "cik"}

    def test_fetch_sec_cik_mapping_filters_zero_cik(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "ZERO", "cik_str": 0},
            "1": {"ticker": "AAPL", "cik_str": 320193}
        }

        with patch('quantdl.master.security_master.requests.get', return_value=mock_response):
            result = sm._fetch_sec_cik_mapping()

        tickers = result['ticker'].to_list()
        assert "ZERO" not in tickers
        assert "AAPL" in tickers

    def test_cik_cusip_mapping_sec_fallback_unavailable(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': [None],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({'ticker': [], 'cik': []}))

        result = sm.cik_cusip_mapping()

        assert result.filter(pl.col('cik').is_null()).height == 1
        sm.logger.warning.assert_called()

    def test_cik_cusip_mapping_sec_fallback_fills_nulls(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1, 2],
            'ticker': ['AAA', 'BBB'],
            'tsymbol': ['AAA', 'BBB'],
            'comnam': ['AAA Corp', 'BBB Corp'],
            'ncusip': ['12345678', '87654321'],
            'cik': [None, '0000000002'],
            'cikdate1': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')]
        })

        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({
            'ticker': ['AAA'],
            'cik': ['0000000001']
        }))

        result = sm.cik_cusip_mapping()

        filled = result.filter(pl.col('symbol') == 'AAA').select('cik').item()
        assert filled == '0000000001'

    def test_cik_cusip_mapping_no_fallback_needed(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()
        sm._fetch_sec_cik_mapping = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        result = sm.cik_cusip_mapping()

        assert result.filter(pl.col('cik').is_null()).height == 0
        sm._fetch_sec_cik_mapping.assert_not_called()

    def test_security_map_new_business_on_symbol_change(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 1],
            'symbol': ['AAA', 'BBB'],
            'company': ['AAA Corp', 'BBB Corp'],
            'cik': ['0001', '0002'],
            'cusip': ['11111111', '22222222'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 2

    def test_security_map_same_business_with_cik_overlap(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 1],
            'symbol': ['AAA', 'BBB'],
            'company': ['AAA Corp', 'BBB Corp'],
            'cik': ['0001', '0001'],
            'cusip': ['11111111', '22222222'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 1

    def test_security_map_new_business_on_permno_change(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 2],
            'symbol': ['AAA', 'AAA'],
            'company': ['AAA Corp', 'AAA Corp'],
            'cik': ['0001', '0001'],
            'cusip': ['11111111', '11111111'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 2

    def test_master_table_includes_security_id(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.cik_cusip = pl.DataFrame({
            'permno': [1],
            'symbol': ['AAA'],
            'company': ['AAA Corp'],
            'cik': ['0001'],
            'cusip': ['11111111'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [1],
            'symbol': ['AAA'],
            'company': ['AAA Corp'],
            'cik': ['0001'],
            'cusip': ['11111111'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        }))

        result = sm.master_table()

        assert 'security_id' in result.columns
        assert result.select('security_id').item() == 101

    def test_master_table_preserves_security_id_with_null_cik(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.cik_cusip = pl.DataFrame({
            'permno': [1],
            'symbol': ['AAC'],
            'company': ['AAC Corp'],
            'cik': [None],
            'cusip': ['12345678'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2009, 12, 31)]
        })
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [1],
            'symbol': ['AAC'],
            'company': ['AAC Corp'],
            'cik': [None],
            'cusip': ['12345678'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2009, 12, 31)]
        }))

        result = sm.master_table()

        assert result.select('security_id').item() == 101


class TestSecurityMasterInitialization:
    """Test SecurityMaster initialization with different configurations"""

    @patch('quantdl.master.security_master.wrds.Connection')
    @patch('quantdl.master.security_master.setup_logger')
    @patch.dict(os.environ, {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    def test_init_without_db_connection(self, mock_logger, mock_wrds):
        """Test initialization without providing db connection (creates new connection)"""
        # Mock the connection
        mock_db_instance = Mock()
        mock_wrds.return_value = mock_db_instance

        # Mock the methods that are called during initialization
        mock_db_instance.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db_instance.raw_sql.return_value):
            sm = SecurityMaster(db=None)

        # Verify connection was created with env variables
        mock_wrds.assert_called_once_with(
            wrds_username='test_user',
            wrds_password='test_pass'
        )
        assert sm.db == mock_db_instance

    @patch('quantdl.master.security_master.setup_logger')
    def test_init_with_db_connection(self, mock_logger):
        """Test initialization with provided db connection"""
        mock_db = Mock()

        # Mock the methods that are called during initialization
        mock_db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db.raw_sql.return_value):
            sm = SecurityMaster(db=mock_db)

        # Verify provided db connection was used
        assert sm.db == mock_db

    @patch('quantdl.master.security_master.setup_logger')
    def test_init_creates_logger(self, mock_logger):
        """Test that initialization creates a logger"""
        mock_db = Mock()
        mock_db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db.raw_sql.return_value):
            sm = SecurityMaster(db=mock_db)

        # Verify logger setup was called
        mock_logger.assert_called_once()
        assert sm.logger is not None

    @patch('quantdl.master.security_master.setup_logger')
    def test_init_sets_sec_cik_cache_to_none(self, mock_logger):
        """Test that _sec_cik_cache is initialized to None"""
        mock_db = Mock()
        mock_db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db.raw_sql.return_value):
            sm = SecurityMaster(db=mock_db)

        # Verify cache is initialized to None
        assert sm._sec_cik_cache is None


class TestSecurityMasterCikCusipMapping:
    """Test SecurityMaster cik_cusip_mapping edge cases"""

    def test_cik_cusip_mapping_logs_null_symbols_when_more_than_50(self):
        """Test logging when more than 50 null CIK symbols exist"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        # Create data with many null CIKs (60 symbols)
        symbols = [f'SYM{i:03d}' for i in range(60)]
        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': list(range(1, 61)),
            'ticker': symbols,
            'tsymbol': symbols,
            'comnam': [f'{sym} Corp' for sym in symbols],
            'ncusip': [f'{i:08d}' for i in range(1, 61)],
            'cik': [None] * 60,
            'cikdate1': [pd.Timestamp('2020-01-01')] * 60,
            'cikdate2': [pd.Timestamp('2020-12-31')] * 60,
            'namedt': [pd.Timestamp('2020-01-01')] * 60,
            'nameenddt': [pd.Timestamp('2020-12-31')] * 60
        })

        # SEC fallback returns non-empty to trigger the logging path
        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({'ticker': ['XYZ'], 'cik': ['0000000999']}))

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=sm.db.raw_sql.return_value):
            result = sm.cik_cusip_mapping()

        # Verify logging was called - should log "... and X more (see detailed log below)"
        # since we have 60 unique symbols with NULL CIK
        log_calls = [str(call) for call in sm.logger.info.call_args_list]
        assert any('more (see detailed log below)' in str(call) for call in log_calls)

    def test_cik_cusip_mapping_logs_detailed_examples_when_more_than_20(self):
        """Test detailed logging when more than 20 null CIK records exist"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        # Create data with many null CIKs (30 records)
        symbols = [f'SYM{i:03d}' for i in range(30)]
        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': list(range(1, 31)),
            'ticker': symbols,
            'tsymbol': symbols,
            'comnam': [f'{sym} Corp' for sym in symbols],
            'ncusip': [f'{i:08d}' for i in range(1, 31)],
            'cik': [None] * 30,
            'cikdate1': [pd.Timestamp('2020-01-01')] * 30,
            'cikdate2': [pd.Timestamp('2020-12-31')] * 30,
            'namedt': [pd.Timestamp('2020-01-01')] * 30,
            'nameenddt': [pd.Timestamp('2020-12-31')] * 30
        })

        # SEC fallback returns non-empty to trigger the logging path
        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({'ticker': ['XYZ'], 'cik': ['0000000999']}))

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=sm.db.raw_sql.return_value):
            result = sm.cik_cusip_mapping()

        # Verify detailed logging was done - should log "... and X more records"
        log_calls = [str(call) for call in sm.logger.info.call_args_list]
        assert any('more records' in str(call) for call in log_calls)


class TestSecurityMasterAutoResolve:
    """Test SecurityMaster auto_resolve edge cases"""

    def test_auto_resolve_with_null_candidates_defensive_code(self):
        """Test defensive null-checking code in auto_resolve (lines 544-546, 565).

        Note: These lines are defensive code that would handle null candidates.
        In normal execution, line 533 filters nulls, making this code unreachable.
        This test verifies the overall behavior when master_tb contains nulls.
        """
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Create a master_tb with null security_ids
        master_tb = pl.DataFrame({
            'security_id': [None, None, 101],
            'symbol': ['AAA', 'AAA', 'AAA'],
            'company': ['AAA Corp 1', 'AAA Corp 2', 'AAA Corp 3'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 6, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm.master_tb = master_tb

        # The auto_resolve method filters nulls at line 533 before the loop
        # So it will only see security_id=101
        result = sm.auto_resolve('AAA', '2020-06-30')

        # Should successfully resolve to the non-null candidate
        assert result == 101
        # Warning should NOT be called because nulls were filtered before the loop
        sm.logger.warning.assert_not_called()

    def test_auto_resolve_verifies_null_filter_works(self):
        """Verify that lines 544-546 and 565 are defensive code (currently unreachable).

        Lines 544-546 check `if candidate_sid is None` and increment null_candidates.
        Line 565 logs a warning if null_candidates > 0.

        These lines are unreachable because line 533 filters out nulls before the loop:
        .filter(pl.col('security_id').is_not_null())

        This test documents that the defensive code exists but cannot be reached
        in normal execution due to the earlier filter.
        """
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Create master_tb with mixed null and valid security_ids
        master_tb = pl.DataFrame({
            'security_id': [None, None, 101],
            'symbol': ['AAA', 'AAA', 'AAA'],
            'company': ['AAA Corp', 'AAA Corp', 'AAA Corp'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm.master_tb = master_tb

        # The filter at line 533 will remove nulls before the loop at line 543
        result = sm.auto_resolve('AAA', '2020-06-30')

        # Should resolve to the only non-null security
        assert result == 101

        # Verify the warning was NOT logged (because nulls were filtered before the loop)
        # This confirms lines 544-546 and 565 were not executed
        sm.logger.warning.assert_not_called()

        # Verify that candidates were properly filtered by checking the intermediate result
        # The candidates query (lines 529-534) should only return non-null security_ids
        candidates = (
            master_tb.filter(pl.col('symbol').eq('AAA'))
            .select('security_id')
            .unique()
            .filter(pl.col('security_id').is_not_null())
        )
        # Should only have one candidate (101), nulls filtered out
        assert candidates.height == 1
        assert candidates['security_id'][0] == 101

    def test_auto_resolve_date_before_symbol_start(self):
        """Test auto_resolve when query date is before symbol start date"""
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2, 2],
            'symbol': ['BBB', 'AAA', 'CCC', 'AAA'],
            'company': ['OldCo', 'OldCo', 'NewCo', 'NewCo'],
            'start_date': [dt.date(2018, 1, 1), dt.date(2020, 1, 1), dt.date(2018, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2020, 12, 31), dt.date(2022, 12, 31), dt.date(2022, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Query date is before AAA was used by security_id=1 (2020-01-01)
        # Security 1 was active 2018-2020, used AAA in 2020
        # Security 2 was active 2018-2022, used AAA in 2021
        result = sm.auto_resolve('AAA', '2019-06-15')

        # Should pick security_id=1 (distance = 200 days to future use)
        # vs security_id=2 (distance = 565 days to future use)
        assert result == 1

    def test_auto_resolve_date_after_symbol_end(self):
        """Test auto_resolve when query date is after symbol end date"""
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2, 2],
            'symbol': ['AAA', 'BBB', 'AAA', 'CCC'],
            'company': ['OldCo', 'OldCo', 'NewCo', 'NewCo'],
            'start_date': [dt.date(2018, 1, 1), dt.date(2020, 1, 1), dt.date(2019, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2020, 12, 31), dt.date(2019, 6, 30), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Query date is after AAA was used by both securities
        # Security 1: used AAA until 2019-12-31, active until 2020-12-31
        # Security 2: used AAA until 2019-06-30, active until 2020-12-31
        result = sm.auto_resolve('AAA', '2020-06-15')

        # Should pick security_id=1 (distance = 167 days from 2019-12-31)
        # vs security_id=2 (distance = 351 days from 2019-06-30)
        assert result == 1


class TestSecurityMasterClose:
    """Test SecurityMaster close method"""

    def test_close_calls_db_close(self):
        """Test that close method calls db.close()"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.db = Mock()

        sm.close()

        sm.db.close.assert_called_once()
