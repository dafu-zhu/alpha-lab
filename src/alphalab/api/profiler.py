"""Operator profiler for development debugging.

Example:
    >>> from alphalab.api.profiler import profile
    >>> with profile():
    ...     result = client.query("rank(-ts_delta(close, 5))")
    # Prints profiling summary on exit
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProfileRecord:
    """Single profiling record for one operator call."""

    operator: str
    duration: float  # seconds
    input_shape: tuple[int, int] | None = None
