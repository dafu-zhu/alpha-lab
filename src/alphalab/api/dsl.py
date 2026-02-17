"""Standalone DSL for alpha expression evaluation.

Example:
    >>> from alphalab.api.dsl import compute
    >>> result = compute("rank(-ts_delta(x, 5))", x=close_df)
"""

from __future__ import annotations

import polars as pl

from alphalab.api import operators
from alphalab.alpha.parser import _evaluate


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
