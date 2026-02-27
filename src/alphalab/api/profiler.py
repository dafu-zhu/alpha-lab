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


@dataclass
class Profiler:
    """Collects profiling records during a profiling session."""

    records: list[ProfileRecord] = field(default_factory=list)

    def record(
        self,
        operator: str,
        duration: float,
        input_shape: tuple[int, int] | None = None,
    ) -> None:
        """Record an operator call."""
        self.records.append(ProfileRecord(operator, duration, input_shape))

    @property
    def total_time(self) -> float:
        """Total time across all recorded operators."""
        return sum(r.duration for r in self.records)
