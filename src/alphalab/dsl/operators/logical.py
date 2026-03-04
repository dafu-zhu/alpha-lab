"""Logical and comparison operators for wide tables.

All operators preserve the wide table structure:
- First column (date) is unchanged
- Operations applied column-wise to symbol columns
- Comparison operators return Boolean dtype
"""


import polars as pl


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


def _ensure_df(
    x: pl.DataFrame | float | int,
    template: pl.DataFrame,
) -> pl.DataFrame:
    """Broadcast scalar to DataFrame matching template structure."""
    if isinstance(x, pl.DataFrame):
        return x
    date_col = template.columns[0]
    value_cols = _get_value_cols(template)
    return template.select(
        pl.col(date_col),
        *[pl.lit(x).alias(c) for c in value_cols],
    )


def and_(x: pl.DataFrame, y: pl.DataFrame) -> pl.DataFrame:
    """Logical AND of two boolean DataFrames.

    Args:
        x: Wide DataFrame with boolean values
        y: Wide DataFrame with boolean values

    Returns:
        Wide DataFrame with AND result
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) & y[c]).alias(c) for c in value_cols],
    )


def or_(x: pl.DataFrame, y: pl.DataFrame) -> pl.DataFrame:
    """Logical OR of two boolean DataFrames.

    Args:
        x: Wide DataFrame with boolean values
        y: Wide DataFrame with boolean values

    Returns:
        Wide DataFrame with OR result
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) | y[c]).alias(c) for c in value_cols],
    )


def not_(x: pl.DataFrame) -> pl.DataFrame:
    """Logical NOT of a boolean DataFrame.

    Args:
        x: Wide DataFrame with boolean values

    Returns:
        Wide DataFrame with NOT result
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(~pl.col(c)).alias(c) for c in value_cols],
    )


def if_else(
    cond: pl.DataFrame,
    then_: pl.DataFrame | float | int,
    else_: pl.DataFrame | float | int,
) -> pl.DataFrame:
    """Conditional selection based on boolean DataFrame.

    Args:
        cond: Wide DataFrame with boolean values
        then_: Value(s) when condition is True (DataFrame or scalar)
        else_: Value(s) when condition is False (DataFrame or scalar)

    Returns:
        Wide DataFrame with selected values
    """
    then_df = _ensure_df(then_, cond)
    else_df = _ensure_df(else_, cond)

    date_col = cond.columns[0]
    value_cols = _get_value_cols(cond)
    return cond.select(
        pl.col(date_col),
        *[
            pl.when(pl.col(c)).then(then_df[c]).otherwise(else_df[c]).alias(c)
            for c in value_cols
        ],
    )


def is_nan(x: pl.DataFrame) -> pl.DataFrame:
    """Check for NaN/null values.

    Detects both Polars null and floating-point NaN.

    Args:
        x: Wide DataFrame

    Returns:
        Wide DataFrame with boolean values (True where NaN/null)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c).is_null() | pl.col(c).is_nan()).alias(c) for c in value_cols],
    )


def lt(
    x: pl.DataFrame,
    y: pl.DataFrame | float | int,
) -> pl.DataFrame:
    """Less than comparison: x < y.

    Args:
        x: Wide DataFrame
        y: Comparison value (DataFrame or scalar)

    Returns:
        Wide DataFrame with boolean values
    """
    y_df = _ensure_df(y, x)
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) < y_df[c]).alias(c) for c in value_cols],
    )


def le(
    x: pl.DataFrame,
    y: pl.DataFrame | float | int,
) -> pl.DataFrame:
    """Less than or equal comparison: x <= y.

    Args:
        x: Wide DataFrame
        y: Comparison value (DataFrame or scalar)

    Returns:
        Wide DataFrame with boolean values
    """
    y_df = _ensure_df(y, x)
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) <= y_df[c]).alias(c) for c in value_cols],
    )


def gt(
    x: pl.DataFrame,
    y: pl.DataFrame | float | int,
) -> pl.DataFrame:
    """Greater than comparison: x > y.

    Args:
        x: Wide DataFrame
        y: Comparison value (DataFrame or scalar)

    Returns:
        Wide DataFrame with boolean values
    """
    y_df = _ensure_df(y, x)
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) > y_df[c]).alias(c) for c in value_cols],
    )


def ge(
    x: pl.DataFrame,
    y: pl.DataFrame | float | int,
) -> pl.DataFrame:
    """Greater than or equal comparison: x >= y.

    Args:
        x: Wide DataFrame
        y: Comparison value (DataFrame or scalar)

    Returns:
        Wide DataFrame with boolean values
    """
    y_df = _ensure_df(y, x)
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) >= y_df[c]).alias(c) for c in value_cols],
    )


def eq(
    x: pl.DataFrame,
    y: pl.DataFrame | float | int,
) -> pl.DataFrame:
    """Equality comparison: x == y.

    Args:
        x: Wide DataFrame
        y: Comparison value (DataFrame or scalar)

    Returns:
        Wide DataFrame with boolean values
    """
    y_df = _ensure_df(y, x)
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) == y_df[c]).alias(c) for c in value_cols],
    )


def ne(
    x: pl.DataFrame,
    y: pl.DataFrame | float | int,
) -> pl.DataFrame:
    """Not equal comparison: x != y.

    Args:
        x: Wide DataFrame
        y: Comparison value (DataFrame or scalar)

    Returns:
        Wide DataFrame with boolean values
    """
    y_df = _ensure_df(y, x)
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) != y_df[c]).alias(c) for c in value_cols],
    )
