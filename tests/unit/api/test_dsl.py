"""Tests for alphalab.api.dsl module."""

from datetime import date

import polars as pl
import pytest

from alphalab.alpha.parser import AlphaParseError
from alphalab.api.dsl import compute


class TestDslCompute:
    """Tests for dsl.compute() function."""

    def test_compute_single_variable(self) -> None:
        """Test compute with single variable."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
        })
        result = compute("rank(x)", x=df)
        assert result.columns == ["Date", "A", "B"]
        assert len(result) == 2

    def test_compute_multiple_variables(self) -> None:
        """Test compute with multiple variables."""
        df1 = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [10.0],
            "B": [20.0],
        })
        df2 = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
            "B": [2.0],
        })
        result = compute("x - y", x=df1, y=df2)
        assert result["A"][0] == 9.0
        assert result["B"][0] == 18.0

    def test_compute_multiline_expression(self) -> None:
        """Test compute with multi-line expression and assignments."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "A": [1.0, 2.0, 3.0],
        })
        result = compute("""
        delta = ts_delta(x, 1)
        rank(delta)
        """, x=df)
        assert result.columns == ["Date", "A"]

    def test_compute_no_ops_required(self) -> None:
        """Test that operators work without passing ops parameter."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [1.0, 2.0],
        })
        # These should work without ops= parameter
        result = compute("ts_mean(x, 2)", x=df)
        assert result is not None

    def test_compute_returns_dataframe(self) -> None:
        """Test that compute returns pl.DataFrame, not Alpha."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
        })
        result = compute("x + 1", x=df)
        assert isinstance(result, pl.DataFrame)


class TestDslComputeErrorCases:
    """Tests for error handling in dsl.compute()."""

    def test_compute_invalid_syntax(self) -> None:
        """Test compute with invalid syntax raises AlphaParseError."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [1.0]})
        with pytest.raises(AlphaParseError, match="Invalid expression syntax"):
            compute("rank(", x=df)

    def test_compute_unknown_variable(self) -> None:
        """Test compute with undefined variable raises error."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [1.0]})
        with pytest.raises(AlphaParseError, match="Unknown variable"):
            compute("rank(y)", x=df)


class TestDslComputeEdgeCases:
    """Tests for edge cases in dsl.compute()."""

    def test_compute_single_row(self) -> None:
        """Test compute with single row DataFrame."""
        df = pl.DataFrame({"Date": [date(2024, 1, 1)], "A": [1.0]})
        result = compute("rank(x)", x=df)
        assert len(result) == 1

    def test_compute_chained_operators(self) -> None:
        """Test compute with chained operator calls."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 11)],
            "A": [float(i) for i in range(1, 11)],
        })
        result = compute("rank(zscore(ts_mean(x, 3)))", x=df)
        assert result.columns == ["Date", "A"]

    def test_compute_string_with_equals(self) -> None:
        """Test compute handles comparison operators correctly."""
        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [1.0, 2.0],
        })
        # Expression with == should work (not confused with assignment)
        result = compute("x == x", x=df)
        assert result is not None
