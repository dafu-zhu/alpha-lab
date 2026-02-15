"""Validation utilities for alpha expressions."""

from __future__ import annotations

import polars as pl


class AlphaError(Exception):
    """Base exception for alpha operations."""


class ColumnMismatchError(AlphaError):
    """Raised when DataFrames have mismatched columns."""

    def __init__(self, left_cols: list[str], right_cols: list[str]) -> None:
        self.left_cols = left_cols
        self.right_cols = right_cols
        super().__init__(
            f"Column mismatch: {left_cols} vs {right_cols}"
        )


class DateMismatchError(AlphaError):
    """Raised when DataFrames have mismatched date indices."""

    def __init__(self, left_dates: int, right_dates: int) -> None:
        self.left_dates = left_dates
        self.right_dates = right_dates
        super().__init__(
            f"Date mismatch: {left_dates} rows vs {right_dates} rows"
        )


def _validate_alignment(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Validate that two DataFrames are aligned for operations.

    Checks:
    1. Same number of rows (dates)
    2. Same columns (symbols)

    Args:
        left: Left DataFrame
        right: Right DataFrame

    Raises:
        ColumnMismatchError: If columns don't match
        DateMismatchError: If row counts don't match
    """
    if left.columns != right.columns:
        raise ColumnMismatchError(left.columns, right.columns)

    if len(left) != len(right):
        raise DateMismatchError(len(left), len(right))
