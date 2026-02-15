"""Arithmetic operators for wide tables.

All operators preserve the wide table structure:
- First column (date) is unchanged
- Operations applied element-wise to symbol columns

Note: min, max, abs are designed to be drop-in replacements for Python built-ins.
They detect input types and delegate to built-ins when appropriate.
"""

import builtins
from typing import Any

import polars as pl


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


def _has_dataframe(*args: Any) -> bool:
    """Check if any argument is a DataFrame."""
    return any(isinstance(a, pl.DataFrame) for a in args)


def abs(x: Any) -> Any:
    """Absolute value - compatible with Python built-in.

    Args:
        x: Scalar, Wide DataFrame, or any object with __abs__

    Returns:
        Absolute value (type matches input)
    """
    if isinstance(x, pl.DataFrame):
        date_col = x.columns[0]
        value_cols = _get_value_cols(x)
        return x.select(
            pl.col(date_col),
            *[pl.col(c).abs().alias(c) for c in value_cols],
        )
    # Fallback to Python built-in
    return builtins.abs(x)


def add(*args: pl.DataFrame, filter: bool = False) -> pl.DataFrame:
    """Element-wise addition of two or more DataFrames.

    Args:
        *args: Two or more wide DataFrames with matching structure
        filter: If True, treat NaN as 0

    Returns:
        Wide DataFrame with summed values
    """
    if len(args) < 2:
        raise ValueError("add requires at least 2 inputs")

    date_col = args[0].columns[0]
    value_cols = _get_value_cols(args[0])

    result = args[0]
    for df in args[1:]:
        if filter:
            result = result.select(
                pl.col(date_col),
                *[
                    (pl.col(c).fill_null(0) + df[c].fill_null(0)).alias(c)
                    for c in value_cols
                ],
            )
        else:
            result = result.select(
                pl.col(date_col),
                *[(pl.col(c) + df[c]).alias(c) for c in value_cols],
            )
    return result


def subtract(x: pl.DataFrame, y: pl.DataFrame, filter: bool = False) -> pl.DataFrame:
    """Element-wise subtraction: x - y.

    Args:
        x: Wide DataFrame with date + symbol columns
        y: Wide DataFrame with date + symbol columns
        filter: If True, treat NaN as 0

    Returns:
        Wide DataFrame with x - y values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    if filter:
        return x.select(
            pl.col(date_col),
            *[
                (pl.col(c).fill_null(0) - y[c].fill_null(0)).alias(c)
                for c in value_cols
            ],
        )
    return x.select(
        pl.col(date_col),
        *[(pl.col(c) - y[c]).alias(c) for c in value_cols],
    )


def multiply(*args: pl.DataFrame | float | int, filter: bool = False) -> pl.DataFrame | float | int:
    """Element-wise multiplication of two or more values.

    Args:
        *args: Two or more values (scalars or DataFrames)
        filter: If True, treat NaN as 1

    Returns:
        Scalar if all inputs are scalars, otherwise DataFrame
    """
    if len(args) < 2:
        raise ValueError("multiply requires at least 2 inputs")

    # Check if all scalars
    if all(isinstance(a, (int, float)) for a in args):
        scalar_result: float | int = 1
        for a in args:
            assert isinstance(a, (int, float))
            scalar_result *= a
        return scalar_result

    # Find first DataFrame to get structure
    first_df = next(a for a in args if isinstance(a, pl.DataFrame))
    date_col = first_df.columns[0]
    value_cols = _get_value_cols(first_df)

    # Start with first arg
    if isinstance(args[0], (int, float)):
        df_result: pl.DataFrame = first_df.select(
            pl.col(date_col),
            *[pl.lit(args[0]).alias(c) for c in value_cols],
        )
    else:
        df_result = args[0]

    for arg in args[1:]:
        if isinstance(arg, (int, float)):
            # Scalar multiplication
            df_result = df_result.select(
                pl.col(date_col),
                *[(pl.col(c) * arg).alias(c) for c in value_cols],
            )
        elif filter:
            df_result = df_result.select(
                pl.col(date_col),
                *[
                    (pl.col(c).fill_null(1) * arg[c].fill_null(1)).alias(c)
                    for c in value_cols
                ],
            )
        else:
            df_result = df_result.select(
                pl.col(date_col),
                *[(pl.col(c) * arg[c]).alias(c) for c in value_cols],
            )
    return df_result


def divide(x: pl.DataFrame, y: pl.DataFrame) -> pl.DataFrame:
    """Safe element-wise division: x / y.

    Division by zero returns null.

    Args:
        x: Wide DataFrame with date + symbol columns (numerator)
        y: Wide DataFrame with date + symbol columns (denominator)

    Returns:
        Wide DataFrame with x / y values (null where y=0)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[
            pl.when(y[c] != 0)
            .then(pl.col(c) / y[c])
            .otherwise(None)
            .alias(c)
            for c in value_cols
        ],
    )


def inverse(x: pl.DataFrame) -> pl.DataFrame:
    """Safe multiplicative inverse: 1/x.

    Division by zero returns null.

    Args:
        x: Wide DataFrame with date + symbol columns

    Returns:
        Wide DataFrame with 1/x values (null where x=0)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[
            pl.when(pl.col(c) != 0)
            .then(1.0 / pl.col(c))
            .otherwise(None)
            .alias(c)
            for c in value_cols
        ],
    )


def log(x: pl.DataFrame) -> pl.DataFrame:
    """Natural logarithm.

    Values <= 0 return null.

    Args:
        x: Wide DataFrame with date + symbol columns

    Returns:
        Wide DataFrame with ln(x) values (null where x<=0)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[
            pl.when(pl.col(c) > 0)
            .then(pl.col(c).log())
            .otherwise(None)
            .alias(c)
            for c in value_cols
        ],
    )


def max(*args: Any, **kwargs: Any) -> Any:
    """Element-wise maximum - compatible with Python built-in.

    When called with DataFrames: element-wise max across inputs.
    Otherwise: delegates to Python's built-in max().

    Args:
        *args: DataFrames for element-wise max, or any args for built-in max
        **kwargs: Passed to built-in max (e.g., key=, default=)

    Returns:
        Wide DataFrame with max values, or result of built-in max

    Examples:
        >>> max(df1, df2)           # Element-wise max of DataFrames
        >>> max([1, 2, 3])          # Python built-in: 3
        >>> max(1, 2, 3)            # Python built-in: 3
        >>> max(items, key=len)     # Python built-in with key
    """
    # If kwargs provided or no DataFrames, use built-in
    if kwargs or not _has_dataframe(*args):
        return builtins.max(*args, **kwargs)

    # Element-wise max for DataFrames
    if len(args) < 2:
        raise ValueError("max requires at least 2 DataFrames for element-wise operation")

    date_col = args[0].columns[0]
    value_cols = _get_value_cols(args[0])

    return args[0].select(
        pl.col(date_col),
        *[
            pl.max_horizontal(*[df[c] for df in args]).alias(c)
            for c in value_cols
        ],
    )


def min(*args: Any, **kwargs: Any) -> Any:
    """Element-wise minimum - compatible with Python built-in.

    When called with DataFrames: element-wise min across inputs.
    Otherwise: delegates to Python's built-in min().

    Args:
        *args: DataFrames for element-wise min, or any args for built-in min
        **kwargs: Passed to built-in min (e.g., key=, default=)

    Returns:
        Wide DataFrame with min values, or result of built-in min

    Examples:
        >>> min(df1, df2)           # Element-wise min of DataFrames
        >>> min([1, 2, 3])          # Python built-in: 1
        >>> min(1, 2, 3)            # Python built-in: 1
        >>> min(items, key=len)     # Python built-in with key
    """
    # If kwargs provided or no DataFrames, use built-in
    if kwargs or not _has_dataframe(*args):
        return builtins.min(*args, **kwargs)

    # Element-wise min for DataFrames
    if len(args) < 2:
        raise ValueError("min requires at least 2 DataFrames for element-wise operation")

    date_col = args[0].columns[0]
    value_cols = _get_value_cols(args[0])

    return args[0].select(
        pl.col(date_col),
        *[
            pl.min_horizontal(*[df[c] for df in args]).alias(c)
            for c in value_cols
        ],
    )


def power(x: pl.DataFrame | float | int, y: pl.DataFrame | float | int) -> pl.DataFrame | float:
    """Element-wise power: x^y.

    Args:
        x: Base - either scalar or DataFrame
        y: Exponent - either scalar or DataFrame

    Returns:
        If both scalar: returns scalar x^y
        Otherwise: Wide DataFrame with x^y values
    """
    # Case 0: both scalars
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x**y

    # Case 1: x is scalar, y is DataFrame
    if isinstance(x, (int, float)):
        assert isinstance(y, pl.DataFrame)
        date_col = y.columns[0]
        value_cols = _get_value_cols(y)
        return y.select(
            pl.col(date_col),
            *[(pl.lit(x).pow(pl.col(c))).alias(c) for c in value_cols],
        )

    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Case 2: x is DataFrame, y is scalar
    if isinstance(y, (int, float)):
        return x.select(
            pl.col(date_col),
            *[(pl.col(c).pow(y)).alias(c) for c in value_cols],
        )

    # Case 3: both are DataFrames
    return x.select(
        pl.col(date_col),
        *[(pl.col(c).pow(y[c])).alias(c) for c in value_cols],
    )


def signed_power(x: pl.DataFrame | float | int, y: pl.DataFrame | float | int) -> pl.DataFrame | float:
    """Signed power: sign(x) * |x|^y.

    Preserves sign of x while raising absolute value to power y.

    Args:
        x: Base - either scalar or DataFrame
        y: Exponent - either scalar or DataFrame

    Returns:
        If both scalar: returns scalar sign(x) * |x|^y
        Otherwise: Wide DataFrame with sign(x) * |x|^y values
    """
    # Both scalars: simple multiplication
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return sign(x) * power(abs(x), y)

    # x scalar, y DataFrame: sign(x) is scalar, power returns DataFrame
    if isinstance(x, (int, float)):
        return multiply(sign(x), power(abs(x), y))

    # x DataFrame, y scalar or DataFrame: use multiply
    return multiply(sign(x), power(abs(x), y))


def sqrt(x: pl.DataFrame) -> pl.DataFrame:
    """Square root.

    Negative values return null.

    Args:
        x: Wide DataFrame with date + symbol columns

    Returns:
        Wide DataFrame with sqrt(x) values (null where x<0)
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[
            pl.when(pl.col(c) >= 0)
            .then(pl.col(c).sqrt())
            .otherwise(None)
            .alias(c)
            for c in value_cols
        ],
    )


def sign(x: pl.DataFrame | float | int) -> pl.DataFrame | int:
    """Sign function: 1 for positive, -1 for negative, 0 for zero.

    Null values remain null.

    Args:
        x: Scalar or Wide DataFrame with date + symbol columns

    Returns:
        Scalar or Wide DataFrame with sign values (1, -1, 0, or null)
    """
    if isinstance(x, (int, float)):
        return 1 if x > 0 else -1 if x < 0 else 0
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[pl.col(c).sign().alias(c) for c in value_cols],
    )


def reverse(x: pl.DataFrame) -> pl.DataFrame:
    """Negation: -x.

    Args:
        x: Wide DataFrame with date + symbol columns

    Returns:
        Wide DataFrame with negated values
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    return x.select(
        pl.col(date_col),
        *[(-pl.col(c)).alias(c) for c in value_cols],
    )


def densify(x: pl.DataFrame) -> pl.DataFrame:
    """Remap unique values to consecutive integers 0..n-1 per row.

    Groups values by unique occurrence within each row and assigns
    sequential indices. Useful for categorical encoding.

    Args:
        x: Wide DataFrame with date + symbol columns

    Returns:
        Wide DataFrame with values remapped to 0..n-1 per row
    """
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)

    # Convert to long format
    long = x.unpivot(
        index=date_col,
        on=value_cols,
        variable_name="symbol",
        value_name="value",
    )

    # Rank unique values per date using dense ranking (ties get same rank)
    ranked = long.with_columns(
        (pl.col("value").rank(method="dense").over(date_col) - 1).alias("value")
    )

    # Pivot back to wide
    wide = ranked.pivot(values="value", index=date_col, on="symbol")

    return wide.select([date_col, *value_cols])
