"""Tests for group feature builder."""

import pytest
import polars as pl
from datetime import date
from unittest.mock import Mock


class TestGroupFeatureBuilder:
    @pytest.fixture
    def mock_security_master(self):
        """Mock security master with GICS columns."""
        mock = Mock()
        mock.master_tb = pl.DataFrame({
            "security_id": [1, 2],
            "symbol": ["AAPL", "MSFT"],
            "sector": ["Information Technology", "Information Technology"],
            "industry": ["Technology Hardware", "Software"],
            "subindustry": ["Consumer Electronics", "Systems Software"],
            "exchange": ["NASDAQ", "NASDAQ"],
        })
        return mock

    def test_build_sector(self, mock_security_master, trading_days):
        from alphalab.features.builders.groups import GroupFeatureBuilder
        builder = GroupFeatureBuilder(mock_security_master)
        result = builder.build("sector", trading_days, ["1", "2"])

        assert "Date" in result.columns
        assert len(result) == len(trading_days)
        assert result["1"][0] == "Information Technology"
        assert result["2"][0] == "Information Technology"

    def test_build_exchange(self, mock_security_master, trading_days):
        from alphalab.features.builders.groups import GroupFeatureBuilder
        builder = GroupFeatureBuilder(mock_security_master)
        result = builder.build("exchange", trading_days, ["1", "2"])
        assert result["1"][0] == "NASDAQ"

    def test_constant_broadcast(self, mock_security_master, trading_days):
        from alphalab.features.builders.groups import GroupFeatureBuilder
        builder = GroupFeatureBuilder(mock_security_master)
        result = builder.build("sector", trading_days, ["1"])
        # All rows should have the same value
        vals = result["1"].to_list()
        assert all(v == "Information Technology" for v in vals)

    def test_missing_column_returns_nulls(self, mock_security_master, trading_days):
        from alphalab.features.builders.groups import GroupFeatureBuilder
        builder = GroupFeatureBuilder(mock_security_master)
        result = builder.build("nonexistent_col", trading_days, ["1"])
        assert result["1"].null_count() == len(trading_days)

    def test_missing_sid_returns_null(self, mock_security_master, trading_days):
        from alphalab.features.builders.groups import GroupFeatureBuilder
        builder = GroupFeatureBuilder(mock_security_master)
        result = builder.build("sector", trading_days, ["1", "999"])
        assert result["999"].null_count() == len(trading_days)

    def test_no_master_tb_returns_nulls(self, trading_days):
        from alphalab.features.builders.groups import GroupFeatureBuilder
        mock = Mock()
        mock.master_tb = None
        builder = GroupFeatureBuilder(mock)
        result = builder.build("sector", trading_days, ["1"])
        assert result["1"].null_count() == len(trading_days)
