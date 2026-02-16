"""Vector operators for list-type columns."""

import polars as pl


def _get_value_cols(df: pl.DataFrame) -> list[str]:
    """Get value columns (all except first which is date)."""
    return df.columns[1:]


def vec_avg(x: pl.DataFrame) -> pl.DataFrame:
    """Mean of each vector field."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).list.mean().alias(c) for c in value_cols],
    )


def vec_sum(x: pl.DataFrame) -> pl.DataFrame:
    """Sum of each vector field."""
    date_col = x.columns[0]
    value_cols = _get_value_cols(x)
    return x.select(
        pl.col(date_col),
        *[pl.col(c).list.sum().alias(c) for c in value_cols],
    )
