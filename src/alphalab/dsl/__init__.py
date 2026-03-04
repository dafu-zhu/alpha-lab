"""Alpha expression DSL — parser, operators, and compute() entry point.

Example:
    >>> from alphalab.dsl import compute
    >>> result = compute("rank(-ts_delta(x, 5))", x=close_df)
"""

from __future__ import annotations

import polars as pl

from alphalab.dsl import operators
from alphalab.dsl.core import Alpha
from alphalab.dsl.parser import AlphaParseError, _evaluate
from alphalab.dsl.types import AlphaLike, Scalar
from alphalab.dsl.validation import (
    AlphaError,
    ColumnMismatchError,
    DateMismatchError,
)


def compute(expr: str, **variables: pl.DataFrame) -> pl.DataFrame:
    """Evaluate alpha expression with auto-injected operators.

    Args:
        expr: Alpha expression string. Supports:
            - Single expressions: "rank(-ts_delta(x, 5))"
            - Multi-line with assignments: "y = rank(x); y + 1"
        **variables: Variable name to DataFrame mappings

    Returns:
        Wide DataFrame with computed result

    Example:
        >>> # Single variable
        >>> result = compute("rank(-ts_delta(x, 5))", x=close_df)

        >>> # Multiple variables
        >>> result = compute("rank(x - y)", x=close_df, y=vwap_df)

        >>> # Multi-line
        >>> result = compute('''
        ... momentum = ts_delta(close, 5)
        ... rank(-momentum)
        ... ''', close=close_df)
    """
    result = _evaluate(expr, variables, ops=operators)
    return result.data


__all__ = [
    "compute",
    "Alpha",
    "AlphaLike",
    "Scalar",
    "AlphaError",
    "AlphaParseError",
    "ColumnMismatchError",
    "DateMismatchError",
    "operators",
]
