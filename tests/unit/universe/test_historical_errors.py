"""Tests for universe.historical error paths."""

import pytest
import polars as pl
from datetime import date
from unittest.mock import Mock


class TestGetHistUniverseLocal:
    def test_with_security_master_instance(self):
        """Test using security_master kwarg."""
        from quantdl.universe.historical import get_hist_universe_local

        mock_sm = Mock()
        mock_sm.master_tb = pl.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "company": ["Apple", "Microsoft"],
            "start_date": [date(2020, 1, 1), date(2020, 1, 1)],
            "end_date": [date(2025, 12, 31), date(2025, 12, 31)],
        })

        result = get_hist_universe_local(2024, security_master=mock_sm)
        assert len(result) == 2
        assert "Ticker" in result.columns
        assert "Name" in result.columns

    def test_no_master_file_raises(self, tmp_path, monkeypatch):
        """Test error when no security master exists."""
        from quantdl.universe import historical
        monkeypatch.setattr(historical, "LOCAL_MASTER_PATH", tmp_path / "nonexistent.parquet")

        with pytest.raises(FileNotFoundError, match="No security master found"):
            historical.get_hist_universe_local(2024)


class TestCrspWithoutWrds:
    def test_crsp_raises_import_error(self):
        """get_hist_universe_crsp raises ImportError without wrds."""
        from quantdl.universe.historical import get_hist_universe_crsp, HAS_WRDS
        if not HAS_WRDS:
            with pytest.raises(ImportError, match="wrds package"):
                get_hist_universe_crsp(2024)

    def test_nasdaq_raises_import_error(self):
        """get_hist_universe_nasdaq raises ImportError without wrds."""
        from quantdl.universe.historical import get_hist_universe_nasdaq, HAS_WRDS
        if not HAS_WRDS:
            with pytest.raises(ImportError, match="wrds package"):
                get_hist_universe_nasdaq(2024)
