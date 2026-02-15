"""Tests for fundamental feature builder."""

import pytest
import polars as pl
from datetime import date


class TestFundamentalBuilderRaw:
    def test_build_raw_assets(self, raw_data_dir, trading_days, security_ids):
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(str(raw_data_dir))
        result = builder.build_raw("assets", "assets", trading_days, security_ids)

        assert "Date" in result.columns
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
        assert "Date" in result.columns
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
        calendar = pl.DataFrame({"Date": trading_days})
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

    def test_invested_capital(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("invested_capital", deps, trading_days, security_ids)
        # invested_capital = equity + debt_lt - cash = 1e9 + 1e9 - 1e9 = 1e9
        vals = result["SEC001"].to_list()
        assert vals[3] == 1e9

    def test_enterprise_value_requires_cap(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        # enterprise_value needs "cap" which is not in built_deps
        # Add a mock cap
        calendar = pl.DataFrame({"Date": trading_days})
        cap = calendar.clone()
        for sid in security_ids:
            cap = cap.with_columns(pl.lit(5e9).alias(sid))
        deps["cap"] = cap
        result = builder.build_derived("enterprise_value", deps, trading_days, security_ids)
        # ev = cap + debt_lt - cash = 5e9 + 1e9 - 1e9 = 5e9
        vals = result["SEC001"].to_list()
        assert vals[3] == 5e9

    def test_operating_expense(self, built_deps, trading_days, security_ids):
        deps, builder = built_deps
        result = builder.build_derived("operating_expense", deps, trading_days, security_ids)
        # op_exp = cogs + sga_expense + depre_amort = 3e9
        vals = result["SEC001"].to_list()
        assert vals[3] == 3e9

    def test_derived_empty_wide_no_common_sids(self, trading_days):
        """Derived field returns empty wide table when no common sids."""
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(".")
        calendar = pl.DataFrame({"Date": trading_days})
        deps = {
            "income": calendar.with_columns(pl.lit(1.0).alias("A")),
            "equity": calendar.with_columns(pl.lit(1.0).alias("B")),
        }
        result = builder.build_derived("return_equity", deps, trading_days, ["C"])
        assert result.columns == ["Date"]

    def test_sales_growth_empty_cols(self, trading_days):
        """sales_growth returns empty wide when no matching sids in sales."""
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(".")
        calendar = pl.DataFrame({"Date": trading_days})
        deps = {"sales": calendar.with_columns(pl.lit(1.0).alias("X"))}
        result = builder.build_derived("sales_growth", deps, trading_days, ["Y"])
        assert result.columns == ["Date"]


class TestFundamentalEdgeCases:
    def test_missing_file_returns_none(self, tmp_path, trading_days):
        """Security without parquet file returns no data."""
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        builder = FundamentalFeatureBuilder(str(tmp_path))
        result = builder.build_raw("assets", "assets", trading_days, ["NOSID"])
        assert "Date" in result.columns
        assert "NOSID" not in result.columns

    def test_string_as_of_date_cast(self, tmp_path, trading_days):
        """String as_of_date column is auto-cast to Date."""
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        fnd_dir = tmp_path / "data" / "raw" / "fundamental" / "STRDATE"
        fnd_dir.mkdir(parents=True)
        df = pl.DataFrame({
            "as_of_date": ["2024-01-05"],
            "concept": ["assets"],
            "value": [42.0],
        })
        df.write_parquet(str(fnd_dir / "fundamental.parquet"))

        builder = FundamentalFeatureBuilder(str(tmp_path))
        result = builder.build_raw("assets", "assets", trading_days, ["STRDATE"])
        assert "STRDATE" in result.columns

    def test_corrupted_file_returns_none(self, tmp_path, trading_days):
        """Corrupted parquet is handled gracefully."""
        from quantdl.features.builders.fundamental import FundamentalFeatureBuilder
        fnd_dir = tmp_path / "data" / "raw" / "fundamental" / "BAD"
        fnd_dir.mkdir(parents=True)
        (fnd_dir / "fundamental.parquet").write_text("corrupted")

        builder = FundamentalFeatureBuilder(str(tmp_path))
        result = builder.build_raw("assets", "assets", trading_days, ["BAD"])
        assert "BAD" not in result.columns
