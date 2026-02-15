"""Alpha expression DSL for composing operators.

Example:
    >>> from quantdl.alpha import Alpha, alpha_eval
    >>> import quantdl.operators as ops
    >>>
    >>> # Using Alpha class with operator overloading
    >>> close = Alpha(close_df)
    >>> volume = Alpha(volume_df)
    >>> alpha = close / Alpha(ops.ts_delay(close.data, 1)) - 1
    >>>
    >>> # Using string DSL (clean syntax for GP/RL)
    >>> result = alpha_eval(
    ...     "rank(-ts_delta(close, 5))",
    ...     {"close": close_df},
    ...     ops=ops,
    ... )
"""

from quantdl.alpha.core import Alpha
from quantdl.alpha.parser import AlphaParseError, alpha_eval
from quantdl.alpha.types import AlphaLike, Scalar
from quantdl.alpha.validation import (
    AlphaError,
    ColumnMismatchError,
    DateMismatchError,
)

__all__ = [
    # Core
    "Alpha",
    "alpha_eval",
    # Types
    "AlphaLike",
    "Scalar",
    # Exceptions
    "AlphaError",
    "AlphaParseError",
    "ColumnMismatchError",
    "DateMismatchError",
]
