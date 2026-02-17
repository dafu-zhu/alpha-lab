"""Alpha expression DSL for composing operators.

Note: For user-friendly API, use `alphalab.api.dsl.compute()` instead.
This module contains internal implementation details.
"""

from alphalab.alpha.core import Alpha
from alphalab.alpha.parser import AlphaParseError
from alphalab.alpha.types import AlphaLike, Scalar
from alphalab.alpha.validation import (
    AlphaError,
    ColumnMismatchError,
    DateMismatchError,
)

__all__ = [
    # Core
    "Alpha",
    # Types
    "AlphaLike",
    "Scalar",
    # Exceptions
    "AlphaError",
    "AlphaParseError",
    "ColumnMismatchError",
    "DateMismatchError",
]
