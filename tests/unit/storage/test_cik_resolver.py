"""
Unit tests for storage.cik_resolver module
Tests CIKResolver functionality against SecurityMaster-backed data.
"""
import datetime as dt
from unittest.mock import Mock

import polars as pl

from alphalab.storage.utils import CIKResolver


class TestCIKResolver:
    """Test CIKResolver class"""

    def _make_master_tb(self, security_id: str, cik_value):
        return pl.DataFrame(
            {
                "security_id": [security_id],
                "start_date": [dt.date(2020, 1, 1)],
                "end_date": [dt.date(2025, 12, 31)],
                "cik": [cik_value],
            }
        )

    def test_get_cik_primary_date(self):
        """Uses master table for the primary date."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik == "0000320193"
        security_master.get_security_id.assert_called_once_with(
            symbol="AAPL",
            day="2024-06-30",
            auto_resolve=True,
        )

    def test_get_cik_null_cik_returns_none(self):
        """NULL CIK in master table returns None."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", None)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik is None

    def test_get_cik_uses_fallback_dates(self):
        """Falls back to later dates when primary date is inactive."""
        security_master = Mock()

        def _get_sid(symbol, day, auto_resolve=True):
            if day == "2024-01-15":
                raise ValueError("not active on")
            return "sid_1"

        security_master.get_security_id.side_effect = _get_sid
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-01-15", year=2024)

        assert cik == "0000320193"

    def test_get_cik_prefers_sec_mapping_for_2025_plus(self):
        """Uses SEC mapping when year is 2025+."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", 320193)
        security_master._fetch_sec_exchange_mapping.return_value = pl.DataFrame(
            {"ticker": ["AAPL"], "cik": ["0000320193"]}
        )

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2025-06-30", year=2025)

        assert cik == "0000320193"

    def test_get_cik_sec_mapping_exception_logs_debug(self):
        """Logs debug when SEC mapping lookup fails."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", 320193)
        security_master._fetch_sec_exchange_mapping.side_effect = RuntimeError("sec down")

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2025-06-30", year=2025)

        assert cik == "0000320193"
        logger.debug.assert_called()

    def test_get_cik_value_error_logs_warning(self):
        """Logs warning on unexpected ValueError from SecurityMaster."""
        security_master = Mock()
        security_master.get_security_id.side_effect = ValueError("boom")
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik is None
        logger.warning.assert_called()

    def test_get_cik_unexpected_exception_logs_error(self):
        """Logs error on unexpected exceptions from SecurityMaster."""
        security_master = Mock()
        security_master.get_security_id.side_effect = RuntimeError("boom")
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik is None
        logger.error.assert_called()

    def test_batch_prefetch_uses_cache(self):
        """Returns cached results without calling get_cik."""
        security_master = Mock()
        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)
        resolver._cik_cache[("AAPL", 2024)] = "0000320193"

        resolver.get_cik = Mock()

        result = resolver.batch_prefetch_ciks(["AAPL"], year=2024)

        assert result == {"AAPL": "0000320193"}
        resolver.get_cik.assert_not_called()

    def test_batch_prefetch_populates_cache(self):
        """Populates cache entries for missing symbols."""
        security_master = Mock()
        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        resolver.get_cik = Mock(side_effect=["0000320193", None])

        result = resolver.batch_prefetch_ciks(["AAPL", "NOPE"], year=2024, batch_size=2)

        assert result == {"AAPL": "0000320193", "NOPE": None}
        assert resolver._cik_cache[("AAPL", 2024)] == "0000320193"
        assert resolver._cik_cache[("NOPE", 2024)] is None

    def test_get_cik_continues_when_security_id_none(self):
        """When security_id is None, continues to next date - covers line 91."""
        security_master = Mock()
        # First call returns None, second call returns valid ID
        security_master.get_security_id.side_effect = [None, "sid_1"]
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik == "0000320193"
        # Should have called get_security_id twice
        assert security_master.get_security_id.call_count == 2

    def test_get_cik_continues_when_cik_record_empty(self):
        """When CIK record is empty, continues to next date - covers line 102."""
        security_master = Mock()
        # Both calls return valid security_id
        security_master.get_security_id.side_effect = ["sid_1", "sid_2"]

        # Create master table with only sid_2 having a CIK
        import polars as pl
        import datetime as dt
        security_master.master_tb = pl.DataFrame({
            'security_id': ['sid_2'],
            'cik': [320193],
            'start_date': [dt.date(2024, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)]
        })

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik == "0000320193"
        # Should have tried both dates
        assert security_master.get_security_id.call_count == 2

    def test_batch_prefetch_logs_progress_every_500(self):
        """Logs progress every 500 symbols - covers line 210."""
        security_master = Mock()
        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        # Create 600 symbols to trigger 500-symbol progress log
        symbols = [f"SYM{i:03d}" for i in range(600)]
        resolver.get_cik = Mock(return_value="0000123456")

        resolver.batch_prefetch_ciks(symbols, year=2024, batch_size=600)

        # Should log progress at 500 symbols (debug level)
        debug_calls = [str(call) for call in logger.debug.call_args_list]
        assert any("500/" in c and "progress" in c.lower() for c in debug_calls)

    def test_clear_cache_clears_and_logs(self):
        """Clear cache clears entries and logs - covers lines 257-258."""
        security_master = Mock()
        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        # Add some entries to cache
        resolver._cik_cache[("AAPL", 2024)] = "0000320193"
        resolver._cik_cache[("MSFT", 2024)] = "0000789019"

        resolver.clear_cache()

        assert len(resolver._cik_cache) == 0
        logger.info.assert_called_with("CIK cache cleared")

    def test_get_cache_stats_returns_correct_counts(self):
        """Get cache stats returns correct entry counts - covers line 262."""
        security_master = Mock()
        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        # Add mixed entries
        resolver._cik_cache[("AAPL", 2024)] = "0000320193"
        resolver._cik_cache[("MSFT", 2024)] = "0000789019"
        resolver._cik_cache[("NOPE", 2024)] = None

        stats = resolver.get_cache_stats()

        assert stats['total_entries'] == 3
        assert stats['cached_ciks'] == 2
        assert stats['null_entries'] == 1
