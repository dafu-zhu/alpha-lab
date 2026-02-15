"""Tests for fundamental feature builder."""

import pytest
import polars as pl
from datetime import date


class TestFundamentalBuilderRaw:
    def test_build_raw_assets(self, raw_data_dir, trading_days, security_ids):
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(str(raw_data_dir))
        result = builder.build_raw("assets", "assets", trading_days, security_ids)

        assert "timestamp" in result.columns
        assert len(result) == len(trading_days)
        for sid in security_ids:
            assert sid in result.columns

    def test_forward_fill_works(self, raw_data_dir, trading_days, security_ids):
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(str(raw_data_dir))
        result = builder.build_raw("assets", "assets", trading_days, security_ids)

        # Data is at date(2024,1,5) which is index 3 in trading_days
        # Days 0-2 should be null (before filing), days 3-9 should be filled
        vals = result["SEC001"].to_list()
        assert vals[0] is None  # before filing date
        assert vals[3] == 1e9  # on filing date
        assert vals[9] == 1e9  # forward-filled

    def test_missing_concept_returns_nulls(self, raw_data_dir, trading_days, security_ids):
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(str(raw_data_dir))
        result = builder.build_raw("nonexistent", "nonexistent_concept", trading_days, security_ids)

        # Should return calendar-only with no value columns added
        assert "timestamp" in result.columns
        assert len(result) == len(trading_days)


class TestFundamentalBuilderDerived:
    @pytest.fixture
    def built_deps(self, raw_data_dir, trading_days, security_ids):
        """Pre-build fundamental dependencies."""
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(str(raw_data_dir))
        deps = {}
        for concept in ["assets", "liabilities", "income", "equity", "sharesout",
                        "debt_lt", "debt_st", "cash", "assets_curr", "liabilities_curr",
                        "sales", "cogs", "operating_income", "depre_amort",
                        "sga_expense", "inventory"]:
            deps[concept] = builder.build_raw(concept, concept, trading_days, security_ids)
        return deps, builder

    def test_debt_is_sum(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("debt", deps, trading_days, security_ids)
        # debt = debt_lt + debt_st, both are 1e9 for SEC001
        vals = result["SEC001"].to_list()
        assert vals[3] == 2e9

    def test_eps_safe_division(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("eps", deps, trading_days, security_ids)
        # eps = income / sharesout = 1e9 / 1e9 = 1.0 for SEC001
        vals = result["SEC001"].to_list()
        assert vals[3] == 1.0

    def test_ebitda(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("ebitda", deps, trading_days, security_ids)
        # ebitda = operating_income + depre_amort = 1e9 + 1e9 = 2e9
        vals = result["SEC001"].to_list()
        assert vals[3] == 2e9

    def test_working_capital(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("working_capital", deps, trading_days, security_ids)
        # working_capital = assets_curr - liabilities_curr = 1e9 - 1e9 = 0
        vals = result["SEC001"].to_list()
        assert vals[3] == 0.0

    def test_current_ratio(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("current_ratio", deps, trading_days, security_ids)
        # current_ratio = assets_curr / liabilities_curr = 1
        vals = result["SEC001"].to_list()
        assert vals[3] == 1.0

    def test_division_by_zero_returns_none(self, trading_days, security_ids):
        """Test that division by zero in derived fields produces None."""
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(".")

        # Create deps with zero denominator
        calendar = pl.DataFrame({"timestamp": trading_days})
        income = calendar.clone()
        equity_zero = calendar.clone()
        for sid in security_ids:
            income = income.with_columns(pl.lit(100.0).alias(sid))
            equity_zero = equity_zero.with_columns(pl.lit(0.0).alias(sid))

        deps = {"income": income, "equity": equity_zero}
        result = builder.build_derived("return_equity", deps, trading_days, security_ids)
        # All values should be None (division by zero)
        assert result["SEC001"].null_count() == len(trading_days)

    def test_unknown_derived_raises(self, trading_days, security_ids):
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(".")
        with pytest.raises(ValueError, match="Unknown derived"):
            builder.build_derived("nonexistent_metric", {}, trading_days, security_ids)

    def test_sales_growth(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("sales_growth", deps, trading_days, security_ids)
        # With only 10 trading days, shift(63) should produce all nulls
        assert result["SEC001"].null_count() == len(trading_days)
