"""Core Alpha class with operator overloading."""

from __future__ import annotations

import polars as pl

from quantdl.alpha.types import AlphaLike, Scalar
from quantdl.alpha.validation import _validate_alignment


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


def _scalar_expr(col: str, op: str, right: int | float) -> pl.Expr:
    """Build expression for scalar binary operation."""
    c = pl.col(col)
    if op == "+":
        return c + right
    if op == "-":
        return c - right
    if op == "*":
        return c * right
    if op == "/":
        return c / right
    if op == "**":
        return c ** right
    if op == "<":
        return (c < right).cast(pl.Float64)
    if op == "<=":
        return (c <= right).cast(pl.Float64)
    if op == ">":
        return (c > right).cast(pl.Float64)
    if op == ">=":
        return (c >= right).cast(pl.Float64)
    if op == "==":
        return (c == right).cast(pl.Float64)
    if op == "!=":
        return (c != right).cast(pl.Float64)
    raise ValueError(f"Unknown op: {op}")


def _df_expr(col: str, op: str, right_col: pl.Series) -> pl.Expr:
    """Build expression for DataFrame binary operation."""
    c = pl.col(col)
    if op == "+":
        return c + right_col
    if op == "-":
        return c - right_col
    if op == "*":
        return c * right_col
    if op == "/":
        return c / right_col
    if op == "**":
        return c ** right_col
    if op == "<":
        return (c < right_col).cast(pl.Float64)
    if op == "<=":
        return (c <= right_col).cast(pl.Float64)
    if op == ">":
        return (c > right_col).cast(pl.Float64)
    if op == ">=":
        return (c >= right_col).cast(pl.Float64)
    if op == "==":
        return (c == right_col).cast(pl.Float64)
    if op == "!=":
        return (c != right_col).cast(pl.Float64)
    if op == "&":
        return (c.cast(pl.Boolean) & right_col.cast(pl.Boolean)).cast(pl.Float64)
    if op == "|":
        return (c.cast(pl.Boolean) | right_col.cast(pl.Boolean)).cast(pl.Float64)
    raise ValueError(f"Unknown op: {op}")


def _apply_binary_op(left: Alpha, right: AlphaLike, op: str) -> Alpha:
    """Apply binary operation between Alpha and AlphaLike.

    Args:
        left: Left operand (Alpha)
        right: Right operand (Alpha, DataFrame, or scalar)
        op: Operation name (+, -, *, /, etc.)

    Returns:
        New Alpha with result
    """
    left_df = left.data
    date_col = left_df.columns[0]
    value_cols = _get_value_cols(left_df)

    if isinstance(right, (int, float)):
        exprs = [_scalar_expr(c, op, right) for c in value_cols]
        return Alpha(left_df.select(pl.col(date_col), *exprs))

    right_df = right.data if isinstance(right, Alpha) else right
    _validate_alignment(left_df, right_df)

    exprs = [_df_expr(c, op, right_df[c]) for c in value_cols]
    return Alpha(left_df.select(pl.col(date_col), *exprs))


def _reverse_scalar_expr(col: str, left: Scalar, op: str) -> pl.Expr:
    """Build expression for reverse scalar binary operation (scalar op Alpha)."""
    c = pl.col(col)
    if op == "+":
        return (left + c).alias(col)
    if op == "-":
        return (left - c).alias(col)
    if op == "*":
        return (left * c).alias(col)
    if op == "/":
        return (left / c).alias(col)
    if op == "**":
        return (pl.lit(left) ** c).alias(col)
    if op == "<":
        return (pl.lit(left) < c).cast(pl.Float64).alias(col)
    if op == "<=":
        return (pl.lit(left) <= c).cast(pl.Float64).alias(col)
    if op == ">":
        return (pl.lit(left) > c).cast(pl.Float64).alias(col)
    if op == ">=":
        return (pl.lit(left) >= c).cast(pl.Float64).alias(col)
    raise ValueError(f"Unknown op: {op}")


def _apply_reverse_binary_op(right: Alpha, left: Scalar, op: str) -> Alpha:
    """Apply reverse binary operation (scalar op Alpha).

    Args:
        right: Right operand (Alpha)
        left: Left operand (scalar)
        op: Operation name

    Returns:
        New Alpha with result
    """
    right_df = right.data
    date_col = right_df.columns[0]
    value_cols = _get_value_cols(right_df)

    exprs = [_reverse_scalar_expr(c, left, op) for c in value_cols]
    return Alpha(right_df.select(pl.col(date_col), *exprs))


class Alpha:
    """Wrapper for wide DataFrames with operator overloading.

    Alpha wraps a Polars DataFrame in wide format (date column + symbol columns)
    and provides arithmetic/comparison operators for composing alpha expressions.

    Example:
        >>> prices = Alpha(price_df)
        >>> volume = Alpha(volume_df)
        >>> alpha = (prices / ts_delay(prices.data, 1) - 1) * volume
    """

    __slots__ = ("_data",)

    def __init__(self, data: pl.DataFrame) -> None:
        """Initialize Alpha with a wide DataFrame.

        Args:
            data: Wide DataFrame with date column first, then symbol columns
        """
        self._data = data

    @property
    def data(self) -> pl.DataFrame:
        """Get underlying DataFrame."""
        return self._data

    def __repr__(self) -> str:
        return f"Alpha({self._data.shape[0]} rows x {self._data.shape[1]} cols)"

    # Arithmetic operators

    def __add__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "+")

    def __radd__(self, other: Scalar) -> Alpha:
        return _apply_reverse_binary_op(self, other, "+")

    def __sub__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "-")

    def __rsub__(self, other: Scalar) -> Alpha:
        return _apply_reverse_binary_op(self, other, "-")

    def __mul__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "*")

    def __rmul__(self, other: Scalar) -> Alpha:
        return _apply_reverse_binary_op(self, other, "*")

    def __truediv__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "/")

    def __rtruediv__(self, other: Scalar) -> Alpha:
        return _apply_reverse_binary_op(self, other, "/")

    def __pow__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "**")

    def __rpow__(self, other: Scalar) -> Alpha:
        return _apply_reverse_binary_op(self, other, "**")

    def __neg__(self) -> Alpha:
        date_col = self._data.columns[0]
        value_cols = _get_value_cols(self._data)
        result = self._data.select(
            pl.col(date_col),
            *[-pl.col(c) for c in value_cols],
        )
        return Alpha(result)

    def __abs__(self) -> Alpha:
        date_col = self._data.columns[0]
        value_cols = _get_value_cols(self._data)
        result = self._data.select(
            pl.col(date_col),
            *[pl.col(c).abs() for c in value_cols],
        )
        return Alpha(result)

    # Comparison operators (return 0.0/1.0 for False/True)

    def __lt__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "<")

    def __le__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "<=")

    def __gt__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, ">")

    def __ge__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, ">=")

    def __eq__(self, other: object) -> Alpha:  # type: ignore[override]
        if not isinstance(other, (Alpha, pl.DataFrame, int, float)):
            return NotImplemented
        return _apply_binary_op(self, other, "==")

    def __ne__(self, other: object) -> Alpha:  # type: ignore[override]
        if not isinstance(other, (Alpha, pl.DataFrame, int, float)):
            return NotImplemented
        return _apply_binary_op(self, other, "!=")

    # Logical operators (treat non-zero as True)

    def __and__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "&")

    def __or__(self, other: AlphaLike) -> Alpha:
        return _apply_binary_op(self, other, "|")

    def __invert__(self) -> Alpha:
        """Logical NOT: ~alpha returns 1.0 where value is 0, else 0.0."""
        date_col = self._data.columns[0]
        value_cols = _get_value_cols(self._data)
        result = self._data.select(
            pl.col(date_col),
            *[(pl.col(c) == 0).cast(pl.Float64) for c in value_cols],
        )
        return Alpha(result)
