"""Tests for alphalab.api.dsl module."""

from datetime import date

import polars as pl
import pytest


class TestDslCompute:
    """Tests for dsl.compute() function."""

    def test_compute_single_variable(self) -> None:
        """Test compute with single variable."""
        from alphalab.api.dsl import compute

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
        from alphalab.api.dsl import compute

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
        from alphalab.api.dsl import compute

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
        from alphalab.api.dsl import compute

        df = pl.DataFrame({
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [1.0, 2.0],
        })
        # These should work without ops= parameter
        result = compute("ts_mean(x, 2)", x=df)
        assert result is not None

    def test_compute_returns_dataframe(self) -> None:
        """Test that compute returns pl.DataFrame, not Alpha."""
        from alphalab.api.dsl import compute

        df = pl.DataFrame({
            "Date": [date(2024, 1, 1)],
            "A": [1.0],
        })
        result = compute("x + 1", x=df)
        assert isinstance(result, pl.DataFrame)
