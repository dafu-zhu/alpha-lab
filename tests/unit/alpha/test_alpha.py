"""Tests for alpha expression DSL."""

from datetime import date

import polars as pl
import pytest

import quantdl.api.operators as ops
from quantdl.alpha import (
    Alpha,
    AlphaParseError,
    ColumnMismatchError,
    DateMismatchError,
    alpha_eval,
)


@pytest.fixture
def wide_df() -> pl.DataFrame:
    """Sample wide DataFrame."""
    return pl.DataFrame({
        "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
        "AAPL": [100.0, 102.0, 101.0, 103.0, 105.0],
        "MSFT": [200.0, 202.0, 201.0, 203.0, 205.0],
    })


@pytest.fixture
def wide_df2() -> pl.DataFrame:
    """Second sample DataFrame for binary ops."""
    return pl.DataFrame({
        "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
        "AAPL": [10.0, 20.0, 30.0, 40.0, 50.0],
        "MSFT": [5.0, 10.0, 15.0, 20.0, 25.0],
    })


class TestAlphaClass:
    """Tests for Alpha class."""

    def test_init(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha initialization."""
        alpha = Alpha(wide_df)
        assert alpha.data.equals(wide_df)

    def test_repr(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha repr."""
        alpha = Alpha(wide_df)
        assert "5 rows" in repr(alpha)
        assert "3 cols" in repr(alpha)

    def test_add_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha + scalar."""
        alpha = Alpha(wide_df)
        result = alpha + 10
        assert result.data["AAPL"][0] == 110.0
        assert result.data["MSFT"][0] == 210.0

    def test_radd_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test scalar + Alpha."""
        alpha = Alpha(wide_df)
        result = 10 + alpha
        assert result.data["AAPL"][0] == 110.0

    def test_sub_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha - scalar."""
        alpha = Alpha(wide_df)
        result = alpha - 10
        assert result.data["AAPL"][0] == 90.0

    def test_rsub_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test scalar - Alpha."""
        alpha = Alpha(wide_df)
        result = 200 - alpha
        assert result.data["AAPL"][0] == 100.0

    def test_mul_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha * scalar."""
        alpha = Alpha(wide_df)
        result = alpha * 2
        assert result.data["AAPL"][0] == 200.0

    def test_rmul_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test scalar * Alpha."""
        alpha = Alpha(wide_df)
        result = 2 * alpha
        assert result.data["AAPL"][0] == 200.0

    def test_truediv_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha / scalar."""
        alpha = Alpha(wide_df)
        result = alpha / 2
        assert result.data["AAPL"][0] == 50.0

    def test_rtruediv_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test scalar / Alpha."""
        alpha = Alpha(wide_df)
        result = 1000 / alpha
        assert result.data["AAPL"][0] == 10.0

    def test_pow_scalar(self) -> None:
        """Test Alpha ** scalar."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [2.0, 3.0, 4.0],
        })
        alpha = Alpha(df)
        result = alpha ** 2
        assert result.data["A"][0] == 4.0
        assert result.data["A"][1] == 9.0

    def test_neg(self, wide_df: pl.DataFrame) -> None:
        """Test -Alpha."""
        alpha = Alpha(wide_df)
        result = -alpha
        assert result.data["AAPL"][0] == -100.0

    def test_abs(self) -> None:
        """Test abs(Alpha)."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [-10.0, 20.0, -30.0],
        })
        alpha = Alpha(df)
        result = abs(alpha)
        assert result.data["A"][0] == 10.0
        assert result.data["A"][1] == 20.0
        assert result.data["A"][2] == 30.0

    def test_add_alpha(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test Alpha + Alpha."""
        a1 = Alpha(wide_df)
        a2 = Alpha(wide_df2)
        result = a1 + a2
        assert result.data["AAPL"][0] == 110.0  # 100 + 10

    def test_sub_alpha(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test Alpha - Alpha."""
        a1 = Alpha(wide_df)
        a2 = Alpha(wide_df2)
        result = a1 - a2
        assert result.data["AAPL"][0] == 90.0  # 100 - 10

    def test_mul_alpha(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test Alpha * Alpha."""
        a1 = Alpha(wide_df)
        a2 = Alpha(wide_df2)
        result = a1 * a2
        assert result.data["AAPL"][0] == 1000.0  # 100 * 10

    def test_truediv_alpha(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test Alpha / Alpha."""
        a1 = Alpha(wide_df)
        a2 = Alpha(wide_df2)
        result = a1 / a2
        assert result.data["AAPL"][0] == 10.0  # 100 / 10

    def test_lt_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha < scalar."""
        alpha = Alpha(wide_df)
        result = alpha < 102
        assert result.data["AAPL"][0] == 1.0  # 100 < 102
        assert result.data["AAPL"][1] == 0.0  # 102 < 102 is False

    def test_le_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha <= scalar."""
        alpha = Alpha(wide_df)
        result = alpha <= 102
        assert result.data["AAPL"][0] == 1.0  # 100 <= 102
        assert result.data["AAPL"][1] == 1.0  # 102 <= 102

    def test_gt_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha > scalar."""
        alpha = Alpha(wide_df)
        result = alpha > 102
        assert result.data["AAPL"][0] == 0.0  # 100 > 102 is False
        assert result.data["AAPL"][2] == 0.0  # 101 > 102 is False
        assert result.data["AAPL"][3] == 1.0  # 103 > 102

    def test_ge_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha >= scalar."""
        alpha = Alpha(wide_df)
        result = alpha >= 102
        assert result.data["AAPL"][1] == 1.0  # 102 >= 102

    def test_eq_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha == scalar."""
        alpha = Alpha(wide_df)
        result = alpha == 102
        assert result.data["AAPL"][0] == 0.0
        assert result.data["AAPL"][1] == 1.0  # 102 == 102

    def test_ne_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test Alpha != scalar."""
        alpha = Alpha(wide_df)
        result = alpha != 102
        assert result.data["AAPL"][0] == 1.0  # 100 != 102
        assert result.data["AAPL"][1] == 0.0  # 102 != 102 is False

    def test_and_alpha(self) -> None:
        """Test Alpha & Alpha (logical and)."""
        df1 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [1.0, 0.0, 1.0],
        })
        df2 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [1.0, 1.0, 0.0],
        })
        a1 = Alpha(df1)
        a2 = Alpha(df2)
        result = a1 & a2
        assert result.data["A"][0] == 1.0  # 1 & 1
        assert result.data["A"][1] == 0.0  # 0 & 1
        assert result.data["A"][2] == 0.0  # 1 & 0

    def test_or_alpha(self) -> None:
        """Test Alpha | Alpha (logical or)."""
        df1 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [1.0, 0.0, 0.0],
        })
        df2 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [0.0, 1.0, 0.0],
        })
        a1 = Alpha(df1)
        a2 = Alpha(df2)
        result = a1 | a2
        assert result.data["A"][0] == 1.0  # 1 | 0
        assert result.data["A"][1] == 1.0  # 0 | 1
        assert result.data["A"][2] == 0.0  # 0 | 0

    def test_invert(self) -> None:
        """Test ~Alpha (logical not)."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [0.0, 1.0, 5.0],
        })
        alpha = Alpha(df)
        result = ~alpha
        assert result.data["A"][0] == 1.0  # ~0 = 1
        assert result.data["A"][1] == 0.0  # ~1 = 0
        assert result.data["A"][2] == 0.0  # ~5 = 0


class TestAlphaValidation:
    """Tests for alignment validation."""

    def test_column_mismatch(self) -> None:
        """Test ColumnMismatchError on mismatched columns."""
        df1 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [1.0, 2.0, 3.0],
        })
        df2 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "B": [1.0, 2.0, 3.0],
        })
        a1 = Alpha(df1)
        a2 = Alpha(df2)
        with pytest.raises(ColumnMismatchError):
            _ = a1 + a2

    def test_date_mismatch(self) -> None:
        """Test DateMismatchError on mismatched row counts."""
        df1 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [1.0, 2.0, 3.0],
        })
        df2 = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 5), eager=True),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        a1 = Alpha(df1)
        a2 = Alpha(df2)
        with pytest.raises(DateMismatchError):
            _ = a1 + a2


class TestAlphaEval:
    """Tests for alpha_eval string DSL."""

    def test_simple_add(self, wide_df: pl.DataFrame) -> None:
        """Test simple addition expression."""
        result = alpha_eval("close + 10", {"close": wide_df})
        assert result.data["AAPL"][0] == 110.0

    def test_simple_mul(self, wide_df: pl.DataFrame) -> None:
        """Test simple multiplication."""
        result = alpha_eval("close * 2", {"close": wide_df})
        assert result.data["AAPL"][0] == 200.0

    def test_binary_alpha_ops(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test Alpha + Alpha in DSL."""
        result = alpha_eval("close + volume", {"close": wide_df, "volume": wide_df2})
        assert result.data["AAPL"][0] == 110.0

    def test_neg(self, wide_df: pl.DataFrame) -> None:
        """Test unary negation."""
        result = alpha_eval("-close", {"close": wide_df})
        assert result.data["AAPL"][0] == -100.0

    def test_comparison(self, wide_df: pl.DataFrame) -> None:
        """Test comparison operators."""
        result = alpha_eval("close > 102", {"close": wide_df})
        assert result.data["AAPL"][0] == 0.0  # 100 > 102 is False
        assert result.data["AAPL"][3] == 1.0  # 103 > 102 is True

    def test_builtin_min(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test min builtin."""
        result = alpha_eval("min(close, volume)", {"close": wide_df, "volume": wide_df2})
        assert result.data["AAPL"][0] == 10.0  # min(100, 10)
        assert result.data["MSFT"][0] == 5.0   # min(200, 5)

    def test_builtin_max(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test max builtin."""
        result = alpha_eval("max(close, volume)", {"close": wide_df, "volume": wide_df2})
        assert result.data["AAPL"][0] == 100.0  # max(100, 10)
        assert result.data["MSFT"][0] == 200.0  # max(200, 5)

    def test_builtin_abs(self, wide_df: pl.DataFrame) -> None:
        """Test abs builtin."""
        result = alpha_eval("abs(-close)", {"close": wide_df})
        assert result.data["AAPL"][0] == 100.0

    def test_ops_namespace(self, wide_df: pl.DataFrame) -> None:
        """Test ops.function style."""
        result = alpha_eval("ops.ts_delta(close, 1)", {"close": wide_df}, ops=ops)
        assert result.data["AAPL"][0] is None  # First diff is null
        assert result.data["AAPL"][1] == 2.0   # 102 - 100

    def test_ops_rank(self, wide_df: pl.DataFrame) -> None:
        """Test ops.rank."""
        result = alpha_eval("ops.rank(close, rate=0)", {"close": wide_df}, ops=ops)
        # AAPL < MSFT so AAPL rank = 0, MSFT rank = 1
        assert result.data["AAPL"][0] == 0.0
        assert result.data["MSFT"][0] == 1.0

    def test_complex_expression(self, wide_df: pl.DataFrame) -> None:
        """Test complex nested expression."""
        result = alpha_eval(
            "ops.rank(-ops.ts_delta(close, 1))",
            {"close": wide_df},
            ops=ops,
        )
        assert result is not None
        assert result.data.columns == wide_df.columns

    def test_syntax_error(self, wide_df: pl.DataFrame) -> None:
        """Test AlphaParseError on syntax error."""
        with pytest.raises(AlphaParseError):
            alpha_eval("close +", {"close": wide_df})

    def test_unknown_variable(self, wide_df: pl.DataFrame) -> None:
        """Test AlphaParseError on unknown variable."""
        with pytest.raises(AlphaParseError, match="Unknown variable"):
            alpha_eval("unknown + 1", {"close": wide_df})

    def test_ternary_if_else(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test ternary if-else expression."""
        result = alpha_eval(
            "a if close > 102 else b",
            {"close": wide_df, "a": wide_df, "b": wide_df2},
        )
        # Where close > 102, use close; else use volume
        assert result.data["AAPL"][0] == 10.0   # 100 <= 102, use volume
        assert result.data["AAPL"][3] == 103.0  # 103 > 102, use close


class TestAlphaWithOperators:
    """Integration tests with quantdl.api.operators."""

    def test_ts_mean_then_rank(self, wide_df: pl.DataFrame) -> None:
        """Test ts_mean -> rank composition."""
        alpha = Alpha(wide_df)
        ma = ops.ts_mean(alpha.data, 3)
        ranked = ops.rank(ma)
        assert ranked.columns == wide_df.columns

    def test_operator_chaining(self, wide_df: pl.DataFrame) -> None:
        """Test chaining operators with Alpha arithmetic."""
        alpha = Alpha(wide_df)
        delta = ops.ts_delta(alpha.data, 1)
        result = Alpha(delta) * -1
        # Should be negative of delta
        assert result.data["AAPL"][1] == -2.0  # -(102-100)

    def test_returns_calculation(self, wide_df: pl.DataFrame) -> None:
        """Test typical returns calculation."""
        alpha = Alpha(wide_df)
        lagged = ops.ts_delay(alpha.data, 1)
        returns = alpha / Alpha(lagged) - 1
        # Return at idx 1: 102/100 - 1 = 0.02
        assert abs(returns.data["AAPL"][1] - 0.02) < 0.001


class TestBuiltinFunctions:
    """Tests for builtin functions in parser."""

    def test_log(self, wide_df: pl.DataFrame) -> None:
        """Test log builtin."""
        import math
        result = alpha_eval("log(close)", {"close": wide_df})
        expected = math.log(100.0)
        assert abs(result.data["AAPL"][0] - expected) < 0.001

    def test_sqrt(self) -> None:
        """Test sqrt builtin."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [4.0, 9.0, 16.0],
        })
        result = alpha_eval("sqrt(x)", {"x": df})
        assert result.data["A"][0] == 2.0
        assert result.data["A"][1] == 3.0
        assert result.data["A"][2] == 4.0

    def test_sign(self) -> None:
        """Test sign builtin."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 4), eager=True),
            "A": [-5.0, 0.0, 10.0, -1.0],
        })
        result = alpha_eval("sign(x)", {"x": df})
        assert result.data["A"][0] == -1.0
        assert result.data["A"][1] == 0.0
        assert result.data["A"][2] == 1.0
        assert result.data["A"][3] == -1.0

    def test_min_with_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test min with scalar argument."""
        result = alpha_eval("min(close, 101)", {"close": wide_df})
        assert result.data["AAPL"][0] == 100.0  # min(100, 101)
        assert result.data["AAPL"][1] == 101.0  # min(102, 101)

    def test_max_with_scalar(self, wide_df: pl.DataFrame) -> None:
        """Test max with scalar argument."""
        result = alpha_eval("max(close, 101)", {"close": wide_df})
        assert result.data["AAPL"][0] == 101.0  # max(100, 101)
        assert result.data["AAPL"][1] == 102.0  # max(102, 101)


class TestBooleanOperations:
    """Tests for boolean operations in DSL."""

    def test_and_expression(self) -> None:
        """Test 'and' expression."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [100.0, 102.0, 98.0],
        })
        result = alpha_eval("(x > 99) & (x < 101)", {"x": df})
        assert result.data["A"][0] == 1.0  # 100: 99 < 100 < 101
        assert result.data["A"][1] == 0.0  # 102: not < 101
        assert result.data["A"][2] == 0.0  # 98: not > 99

    def test_or_expression(self) -> None:
        """Test 'or' expression."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [100.0, 102.0, 98.0],
        })
        result = alpha_eval("(x < 99) | (x > 101)", {"x": df})
        assert result.data["A"][0] == 0.0  # 100: neither
        assert result.data["A"][1] == 1.0  # 102: > 101
        assert result.data["A"][2] == 1.0  # 98: < 99

    def test_chained_comparison(self) -> None:
        """Test chained comparison: a < b < c."""
        df = pl.DataFrame({
            "Date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "A": [50.0, 100.0, 150.0],
        })
        result = alpha_eval("0 < x < 100", {"x": df})
        assert result.data["A"][0] == 1.0  # 0 < 50 < 100
        assert result.data["A"][1] == 0.0  # 0 < 100 < 100 is False
        assert result.data["A"][2] == 0.0  # 0 < 150 < 100 is False


class TestCleanSyntax:
    """Tests for clean syntax without ops. prefix (GP/RL friendly)."""

    def test_ts_delta_clean(self, wide_df: pl.DataFrame) -> None:
        """Test ts_delta without ops. prefix."""
        result = alpha_eval("ts_delta(close, 1)", {"close": wide_df}, ops=ops)
        assert result.data["AAPL"][0] is None  # First diff is null
        assert result.data["AAPL"][1] == 2.0   # 102 - 100

    def test_rank_clean(self, wide_df: pl.DataFrame) -> None:
        """Test rank without ops. prefix."""
        result = alpha_eval("rank(close, rate=0)", {"close": wide_df}, ops=ops)
        # AAPL < MSFT so AAPL rank = 0, MSFT rank = 1
        assert result.data["AAPL"][0] == 0.0
        assert result.data["MSFT"][0] == 1.0

    def test_nested_clean(self, wide_df: pl.DataFrame) -> None:
        """Test nested operators without ops. prefix."""
        result = alpha_eval(
            "rank(-ts_delta(close, 1))",
            {"close": wide_df},
            ops=ops,
        )
        assert result is not None
        assert result.data.columns == wide_df.columns

    def test_ts_mean_clean(self, wide_df: pl.DataFrame) -> None:
        """Test ts_mean without ops. prefix."""
        result = alpha_eval("ts_mean(close, 3)", {"close": wide_df}, ops=ops)
        assert result is not None
        # Partial windows allowed: row 0 has mean of [100], row 1 has mean of [100, 102]
        assert result.data["AAPL"][0] == 100.0
        assert result.data["AAPL"][1] == 101.0  # (100 + 102) / 2
        # Third value: avg(100, 102, 101) = 101
        assert result.data["AAPL"][2] == 101.0

    def test_zscore_clean(self, wide_df: pl.DataFrame) -> None:
        """Test zscore without ops. prefix."""
        result = alpha_eval("zscore(close)", {"close": wide_df}, ops=ops)
        assert result is not None
        # Row-wise z-score: AAPL < MSFT so AAPL is negative, MSFT is positive
        assert result.data["AAPL"][0] < 0
        assert result.data["MSFT"][0] > 0

    def test_complex_alpha_clean(self, wide_df: pl.DataFrame) -> None:
        """Test complex alpha expression without ops. prefix."""
        result = alpha_eval(
            "scale(rank(ts_delta(close, 1)))",
            {"close": wide_df},
            ops=ops,
        )
        assert result is not None
        assert result.data.columns == wide_df.columns

    def test_mixed_syntax_backwards_compat(self, wide_df: pl.DataFrame) -> None:
        """Test that ops.func still works (backward compatibility)."""
        # Both should produce same result
        result_old = alpha_eval("ops.ts_delta(close, 1)", {"close": wide_df}, ops=ops)
        result_new = alpha_eval("ts_delta(close, 1)", {"close": wide_df}, ops=ops)
        assert result_old.data["AAPL"][1] == result_new.data["AAPL"][1]

    def test_ts_delay_clean(self, wide_df: pl.DataFrame) -> None:
        """Test ts_delay without ops. prefix."""
        result = alpha_eval("ts_delay(close, 1)", {"close": wide_df}, ops=ops)
        assert result.data["AAPL"][0] is None  # First value shifted out
        assert result.data["AAPL"][1] == 100.0  # Previous day's value

    def test_ts_corr_clean(self, wide_df: pl.DataFrame, wide_df2: pl.DataFrame) -> None:
        """Test ts_corr without ops. prefix."""
        result = alpha_eval(
            "ts_corr(x, y, 3)",
            {"x": wide_df, "y": wide_df2},
            ops=ops,
        )
        assert result is not None
        # Correlation requires at least 3 periods
        assert result.data["AAPL"][0] is None
        assert result.data["AAPL"][1] is None


class TestAlphaQuery:
    """Tests for multi-statement alpha_query()."""

    def test_single_statement(self, wide_df: pl.DataFrame) -> None:
        """Single expression works in alpha_query."""
        from quantdl.alpha.parser import alpha_query
        result = alpha_query("close + 1", {"close": wide_df}, ops=ops)
        assert result.data["AAPL"][0] == 101.0

    def test_variable_assignment(self, wide_df: pl.DataFrame) -> None:
        """Variable assignment and reuse."""
        from quantdl.alpha.parser import alpha_query
        result = alpha_query("x = close + 1; x + 1", {"close": wide_df}, ops=ops)
        assert result.data["AAPL"][0] == 102.0

    def test_multi_assignment(self, wide_df: pl.DataFrame) -> None:
        """Multiple assignments chained."""
        from quantdl.alpha.parser import alpha_query
        result = alpha_query(
            "a = close + 1; b = a + 1; b + 1",
            {"close": wide_df},
            ops=ops,
        )
        assert result.data["AAPL"][0] == 103.0

    def test_semicolons_with_ops(self, wide_df: pl.DataFrame) -> None:
        """Operators work inside multi-statement query."""
        from quantdl.alpha.parser import alpha_query
        result = alpha_query(
            "delta = ts_delta(close, 1); rank(delta)",
            {"close": wide_df},
            ops=ops,
        )
        assert result is not None
        assert result.data.columns == wide_df.columns

    def test_last_expression_is_result(self, wide_df: pl.DataFrame) -> None:
        """The last expression's value is the return value."""
        from quantdl.alpha.parser import alpha_query
        result = alpha_query(
            "x = close + 100; close",
            {"close": wide_df},
            ops=ops,
        )
        # Should return close, not x
        assert result.data["AAPL"][0] == 100.0

    def test_syntax_error_raises(self) -> None:
        """Invalid syntax raises AlphaParseError."""
        from quantdl.alpha.parser import alpha_query
        with pytest.raises(AlphaParseError):
            alpha_query("x = ", {}, ops=ops)
