"""Type definitions for alpha expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from quantdl.alpha.core import Alpha

import polars as pl

# Core type for values that can participate in alpha operations
AlphaLike = Union["Alpha", pl.DataFrame, int, float]

# Scalar types for broadcasting
Scalar = Union[int, float]  # noqa: UP007
