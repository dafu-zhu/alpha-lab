"""Tests for FeatureBuilder orchestrator."""

import pytest
import polars as pl
from datetime import date
from pathlib import Path
from unittest.mock import Mock


class TestFeatureBuilder:
    @pytest.fixture
    def mock_security_master(self):
        mock = Mock()
        mock.master_tb = pl.DataFrame({
            "security_id": [1, 2],
            "symbol": ["AAPL", "MSFT"],
            "sector": ["IT", "IT"],
            "industry": ["HW", "SW"],
            "subindustry": ["CE", "SS"],
            "exchange": ["NASDAQ", "NASDAQ"],
        })
        return mock

    def test_build_all_writes_arrow_files(self, raw_data_dir, trading_days, security_ids, mock_security_master):
        from quantdl.features.builder import FeatureBuilder
        builder = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written = builder.build_all(trading_days, security_ids)

        assert len(written) > 0
        for field_name, path in written.items():
            assert Path(path).exists()
            assert path.endswith(".arrow")

    def test_build_all_creates_close(self, raw_data_dir, trading_days, security_ids, mock_security_master):
        from quantdl.features.builder import FeatureBuilder
        builder = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written = builder.build_all(trading_days, security_ids)

        assert "close" in written
        df = pl.read_ipc(written["close"])
        assert "Date" in df.columns
        assert len(df) == len(trading_days)

    def test_skip_existing(self, raw_data_dir, trading_days, security_ids, mock_security_master):
        from quantdl.features.builder import FeatureBuilder

        # Build once
        builder = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written1 = builder.build_all(trading_days, security_ids, overwrite=False)

        # Build again (should skip)
        builder2 = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written2 = builder2.build_all(trading_days, security_ids, overwrite=False)

        assert set(written1.keys()) == set(written2.keys())

    def test_overwrite_rebuilds(self, raw_data_dir, trading_days, security_ids, mock_security_master):
        from quantdl.features.builder import FeatureBuilder

        builder = FeatureBuilder(str(raw_data_dir), mock_security_master)
        builder.build_all(trading_days, security_ids, overwrite=False)

        # Overwrite should rebuild
        builder2 = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written2 = builder2.build_all(trading_days, security_ids, overwrite=True)
        assert len(written2) > 0

    def test_dependencies_built_first(self, raw_data_dir, trading_days, security_ids, mock_security_master):
        from quantdl.features.builder import FeatureBuilder
        builder = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written = builder.build_all(trading_days, security_ids)

        # Returns depends on close; both should be built
        assert "close" in written
        assert "returns" in written

    def test_derived_fields_built(self, raw_data_dir, trading_days, security_ids, mock_security_master):
        from quantdl.features.builder import FeatureBuilder
        builder = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written = builder.build_all(trading_days, security_ids)

        # Derived fields should all be present
        for field in ["debt", "ebitda", "eps", "working_capital"]:
            assert field in written

    def test_group_fields_built(self, raw_data_dir, trading_days, security_ids, mock_security_master):
        from quantdl.features.builder import FeatureBuilder
        builder = FeatureBuilder(str(raw_data_dir), mock_security_master)
        written = builder.build_all(trading_days, security_ids)

        for field in ["sector", "industry", "subindustry", "exchange"]:
            assert field in written
