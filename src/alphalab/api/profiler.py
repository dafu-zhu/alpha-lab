"""Operator profiler for development debugging.

Example:
    >>> from alphalab.api.profiler import profile
    >>> with profile():
    ...     result = client.query("rank(-ts_delta(close, 5))")
    # Prints profiling summary on exit
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

_local = threading.local()


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

    def summary(self) -> None:
        """Print profiling summary table to stdout."""
        if not self.records:
            print("No profiling records.")
            return

        # Aggregate by operator
        from collections import defaultdict

        agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "total": 0.0, "shape": None})
        for r in self.records:
            agg[r.operator]["calls"] += 1
            agg[r.operator]["total"] += r.duration
            if r.input_shape:
                agg[r.operator]["shape"] = r.input_shape

        total = self.total_time

        # Print table
        print("┌" + "─" * 14 + "┬" + "─" * 7 + "┬" + "─" * 10 + "┬" + "─" * 9 + "┬" + "─" * 13 + "┐")
        print(f"│ {'Operator':<12} │ {'Calls':>5} │ {'Total(s)':>8} │ {'%Total':>7} │ {'Input Shape':>11} │")
        print("├" + "─" * 14 + "┼" + "─" * 7 + "┼" + "─" * 10 + "┼" + "─" * 9 + "┼" + "─" * 13 + "┤")

        # Sort by total time descending
        for op, data in sorted(agg.items(), key=lambda x: -x[1]["total"]):
            pct = (data["total"] / total * 100) if total > 0 else 0
            shape_str = f"{data['shape'][0]}×{data['shape'][1]}" if data["shape"] else ""
            print(f"│ {op:<12} │ {data['calls']:>5} │ {data['total']:>8.3f} │ {pct:>6.1f}% │ {shape_str:>11} │")

        print("├" + "─" * 14 + "┼" + "─" * 7 + "┼" + "─" * 10 + "┼" + "─" * 9 + "┼" + "─" * 13 + "┤")
        print(f"│ {'TOTAL':<12} │ {len(self.records):>5} │ {total:>8.3f} │ {'100.0%':>7} │ {'':<11} │")
        print("└" + "─" * 14 + "┴" + "─" * 7 + "┴" + "─" * 10 + "┴" + "─" * 9 + "┴" + "─" * 13 + "┘")


def _get_profiler() -> Profiler | None:
    """Get the active profiler for this thread, if any."""
    return getattr(_local, "profiler", None)


@contextmanager
def profile() -> Generator[Profiler, None, None]:
    """Context manager to enable operator profiling.

    Example:
        >>> with profile() as p:
        ...     result = client.query("rank(-ts_delta(close, 5))")
        # Prints summary on exit
    """
    p = Profiler()
    _local.profiler = p
    try:
        yield p
    finally:
        _local.profiler = None
        p.summary()
