# Operator Profiler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add operator-level profiling to identify performance bottlenecks in alpha expressions.

**Architecture:** Thread-local profiler activated via context manager. Each operator wrapped with `@profiled` decorator that records timing when profiler is active. Zero meaningful overhead (~50ns) when profiling disabled.

**Tech Stack:** Python stdlib (threading, contextlib, time, dataclasses), polars for DataFrame shape detection

---

## Task 1: Create ProfileRecord and Profiler Core

**Files:**
- Create: `src/alphalab/api/profiler.py`
- Test: `tests/unit/api/test_profiler.py`

**Step 1: Write the failing test for ProfileRecord**

```python
# tests/unit/api/test_profiler.py
"""Tests for operator profiler."""

import pytest


def test_profile_record_stores_data():
    """ProfileRecord stores operator name, duration, and shape."""
    from alphalab.api.profiler import ProfileRecord

    record = ProfileRecord(operator="rank", duration=0.5, input_shape=(100, 50))

    assert record.operator == "rank"
    assert record.duration == 0.5
    assert record.input_shape == (100, 50)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profile_record_stores_data -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/alphalab/api/profiler.py
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profile_record_stores_data -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/profiler.py tests/unit/api/test_profiler.py
git commit -m "feat(profiler): add ProfileRecord dataclass"
```

---

## Task 2: Add Profiler Class with Record Collection

**Files:**
- Modify: `src/alphalab/api/profiler.py`
- Test: `tests/unit/api/test_profiler.py`

**Step 1: Write the failing test**

```python
def test_profiler_collects_records():
    """Profiler collects records and computes total time."""
    from alphalab.api.profiler import Profiler

    p = Profiler()
    p.record("rank", 0.5, (100, 50))
    p.record("ts_delta", 0.2, (100, 50))

    assert len(p.records) == 2
    assert p.total_time == pytest.approx(0.7, rel=1e-6)
    assert p.records[0].operator == "rank"
    assert p.records[1].operator == "ts_delta"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profiler_collects_records -v`
Expected: FAIL with "cannot import name 'Profiler'"

**Step 3: Write minimal implementation**

Add to `src/alphalab/api/profiler.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profiler_collects_records -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/profiler.py tests/unit/api/test_profiler.py
git commit -m "feat(profiler): add Profiler class with record collection"
```

---

## Task 3: Add Summary Printing

**Files:**
- Modify: `src/alphalab/api/profiler.py`
- Test: `tests/unit/api/test_profiler.py`

**Step 1: Write the failing test**

```python
def test_profiler_summary_output(capsys):
    """Profiler prints formatted summary table."""
    from alphalab.api.profiler import Profiler

    p = Profiler()
    p.record("rank", 0.85, (2000, 5000))
    p.record("ts_delta", 0.15, (2000, 5000))
    p.summary()

    captured = capsys.readouterr()
    assert "rank" in captured.out
    assert "ts_delta" in captured.out
    assert "85.0%" in captured.out  # rank is 85% of total
    assert "TOTAL" in captured.out
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profiler_summary_output -v`
Expected: FAIL with "AttributeError: 'Profiler' object has no attribute 'summary'"

**Step 3: Write minimal implementation**

Add to `Profiler` class in `src/alphalab/api/profiler.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profiler_summary_output -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/profiler.py tests/unit/api/test_profiler.py
git commit -m "feat(profiler): add summary table printing"
```

---

## Task 4: Add Thread-Local Context Manager

**Files:**
- Modify: `src/alphalab/api/profiler.py`
- Test: `tests/unit/api/test_profiler.py`

**Step 1: Write the failing test**

```python
def test_profile_context_manager(capsys):
    """profile() context manager activates profiler and prints on exit."""
    from alphalab.api.profiler import profile, _get_profiler

    # Before context: no profiler
    assert _get_profiler() is None

    with profile() as p:
        # Inside context: profiler active
        assert _get_profiler() is p
        p.record("test_op", 0.1, (10, 10))

    # After context: no profiler, summary printed
    assert _get_profiler() is None
    captured = capsys.readouterr()
    assert "test_op" in captured.out
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profile_context_manager -v`
Expected: FAIL with "cannot import name 'profile'"

**Step 3: Write minimal implementation**

Add to `src/alphalab/api/profiler.py`:

```python
import threading
from contextlib import contextmanager
from typing import Generator

_local = threading.local()


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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profile_context_manager -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/profiler.py tests/unit/api/test_profiler.py
git commit -m "feat(profiler): add thread-local profile() context manager"
```

---

## Task 5: Add @profiled Decorator

**Files:**
- Modify: `src/alphalab/api/profiler.py`
- Test: `tests/unit/api/test_profiler.py`

**Step 1: Write the failing tests**

```python
def test_profiled_decorator_records_when_active():
    """@profiled decorator records timing when profiler active."""
    from alphalab.api.profiler import profile, profiled
    import polars as pl

    @profiled
    def my_operator(x: pl.DataFrame) -> pl.DataFrame:
        return x

    df = pl.DataFrame({"Date": [1, 2], "A": [1.0, 2.0], "B": [3.0, 4.0]})

    with profile() as p:
        result = my_operator(df)

    assert len(p.records) == 1
    assert p.records[0].operator == "my_operator"
    assert p.records[0].duration > 0
    assert p.records[0].input_shape == (2, 3)


def test_profiled_decorator_noop_when_inactive():
    """@profiled decorator has no effect when profiler not active."""
    from alphalab.api.profiler import profiled, _get_profiler
    import polars as pl

    call_count = 0

    @profiled
    def my_operator(x: pl.DataFrame) -> pl.DataFrame:
        nonlocal call_count
        call_count += 1
        return x

    df = pl.DataFrame({"Date": [1], "A": [1.0]})

    # No profiler active
    assert _get_profiler() is None
    result = my_operator(df)

    assert call_count == 1
    assert result.equals(df)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profiled_decorator_records_when_active -v`
Expected: FAIL with "cannot import name 'profiled'"

**Step 3: Write minimal implementation**

Add to `src/alphalab/api/profiler.py`:

```python
import functools
import time
from typing import Callable, TypeVar

import polars as pl

F = TypeVar("F", bound=Callable)


def _get_input_shape(args: tuple) -> tuple[int, int] | None:
    """Extract shape from first DataFrame argument."""
    for arg in args:
        if isinstance(arg, pl.DataFrame):
            return (arg.height, arg.width)
    return None


def profiled(fn: F) -> F:
    """Decorator to profile operator function calls.

    When a profiler is active (via `with profile():`), records the operator
    name, execution time, and input shape. When no profiler is active,
    passes through with minimal overhead (~50ns).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        p = _get_profiler()
        if p is None:
            return fn(*args, **kwargs)

        shape = _get_input_shape(args)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration = time.perf_counter() - start
        p.record(fn.__name__, duration, shape)
        return result

    return wrapper  # type: ignore[return-value]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/api/test_profiler.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/alphalab/api/profiler.py tests/unit/api/test_profiler.py
git commit -m "feat(profiler): add @profiled decorator"
```

---

## Task 6: Wrap Operators in __init__.py

**Files:**
- Modify: `src/alphalab/api/operators/__init__.py`

**Step 1: Write the failing integration test**

```python
# tests/unit/api/test_profiler.py (append)

def test_operators_are_profiled():
    """All operators in alphalab.api.operators are wrapped with @profiled."""
    from alphalab.api import operators
    from alphalab.api.profiler import profile
    import polars as pl

    df = pl.DataFrame({
        "Date": [1, 2, 3, 4, 5],
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [5.0, 4.0, 3.0, 2.0, 1.0],
    })

    with profile() as p:
        _ = operators.rank(df)
        _ = operators.ts_mean(df, 2)

    op_names = {r.operator for r in p.records}
    assert "rank" in op_names
    assert "ts_mean" in op_names
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_operators_are_profiled -v`
Expected: FAIL (operators not profiled yet, p.records empty)

**Step 3: Write the implementation**

Replace `src/alphalab/api/operators/__init__.py` with:

```python
"""Alpha operators for wide table transformations.

All operators work on wide DataFrames where:
- First column is the date/timestamp
- Remaining columns are symbol values

Example:
    ```python
    from alphalab.api.operators import ts_mean, rank, zscore

    # Apply 20-day moving average
    ma = ts_mean(prices, 20)

    # Cross-sectional rank
    ranked = rank(ma)

    # Z-score within each date
    standardized = zscore(ma)
    ```
"""

from alphalab.api.profiler import profiled

# Import raw operators with underscore prefix
from alphalab.api.operators.arithmetic import (
    abs as _abs,
    add as _add,
    densify as _densify,
    divide as _divide,
    inverse as _inverse,
    log as _log,
    max as _max,
    min as _min,
    multiply as _multiply,
    power as _power,
    reverse as _reverse,
    sign as _sign,
    signed_power as _signed_power,
    sqrt as _sqrt,
    subtract as _subtract,
)
from alphalab.api.operators.cross_sectional import (
    bucket as _bucket,
    normalize as _normalize,
    quantile as _quantile,
    rank as _rank,
    scale as _scale,
    winsorize as _winsorize,
    zscore as _zscore,
)
from alphalab.api.operators.group import (
    group_backfill as _group_backfill,
    group_mean as _group_mean,
    group_neutralize as _group_neutralize,
    group_rank as _group_rank,
    group_scale as _group_scale,
    group_zscore as _group_zscore,
)
from alphalab.api.operators.logical import (
    and_ as _and_,
    eq as _eq,
    ge as _ge,
    gt as _gt,
    if_else as _if_else,
    is_nan as _is_nan,
    le as _le,
    lt as _lt,
    ne as _ne,
    not_ as _not_,
    or_ as _or_,
)
from alphalab.api.operators.time_series import (
    days_from_last_change as _days_from_last_change,
    hump as _hump,
    kth_element as _kth_element,
    last_diff_value as _last_diff_value,
    ts_arg_max as _ts_arg_max,
    ts_arg_min as _ts_arg_min,
    ts_av_diff as _ts_av_diff,
    ts_backfill as _ts_backfill,
    ts_corr as _ts_corr,
    ts_count_nans as _ts_count_nans,
    ts_covariance as _ts_covariance,
    ts_decay_linear as _ts_decay_linear,
    ts_delay as _ts_delay,
    ts_delta as _ts_delta,
    ts_max as _ts_max,
    ts_mean as _ts_mean,
    ts_min as _ts_min,
    ts_product as _ts_product,
    ts_quantile as _ts_quantile,
    ts_rank as _ts_rank,
    ts_regression as _ts_regression,
    ts_scale as _ts_scale,
    ts_std as _ts_std,
    ts_step as _ts_step,
    ts_sum as _ts_sum,
    ts_zscore as _ts_zscore,
)
from alphalab.api.operators.transformational import trade_when as _trade_when
from alphalab.api.operators.vector import vec_avg as _vec_avg, vec_sum as _vec_sum

# Wrap all operators with profiler
# Arithmetic
abs = profiled(_abs)
add = profiled(_add)
densify = profiled(_densify)
divide = profiled(_divide)
inverse = profiled(_inverse)
log = profiled(_log)
max = profiled(_max)
min = profiled(_min)
multiply = profiled(_multiply)
power = profiled(_power)
reverse = profiled(_reverse)
sign = profiled(_sign)
signed_power = profiled(_signed_power)
sqrt = profiled(_sqrt)
subtract = profiled(_subtract)

# Cross-sectional
bucket = profiled(_bucket)
normalize = profiled(_normalize)
quantile = profiled(_quantile)
rank = profiled(_rank)
scale = profiled(_scale)
winsorize = profiled(_winsorize)
zscore = profiled(_zscore)

# Group
group_backfill = profiled(_group_backfill)
group_mean = profiled(_group_mean)
group_neutralize = profiled(_group_neutralize)
group_rank = profiled(_group_rank)
group_scale = profiled(_group_scale)
group_zscore = profiled(_group_zscore)

# Logical
and_ = profiled(_and_)
eq = profiled(_eq)
ge = profiled(_ge)
gt = profiled(_gt)
if_else = profiled(_if_else)
is_nan = profiled(_is_nan)
le = profiled(_le)
lt = profiled(_lt)
ne = profiled(_ne)
not_ = profiled(_not_)
or_ = profiled(_or_)

# Time-series
days_from_last_change = profiled(_days_from_last_change)
hump = profiled(_hump)
kth_element = profiled(_kth_element)
last_diff_value = profiled(_last_diff_value)
ts_arg_max = profiled(_ts_arg_max)
ts_arg_min = profiled(_ts_arg_min)
ts_av_diff = profiled(_ts_av_diff)
ts_backfill = profiled(_ts_backfill)
ts_corr = profiled(_ts_corr)
ts_count_nans = profiled(_ts_count_nans)
ts_covariance = profiled(_ts_covariance)
ts_decay_linear = profiled(_ts_decay_linear)
ts_delay = profiled(_ts_delay)
ts_delta = profiled(_ts_delta)
ts_max = profiled(_ts_max)
ts_mean = profiled(_ts_mean)
ts_min = profiled(_ts_min)
ts_product = profiled(_ts_product)
ts_quantile = profiled(_ts_quantile)
ts_rank = profiled(_ts_rank)
ts_regression = profiled(_ts_regression)
ts_scale = profiled(_ts_scale)
ts_std = profiled(_ts_std)
ts_step = profiled(_ts_step)
ts_sum = profiled(_ts_sum)
ts_zscore = profiled(_ts_zscore)

# Transformational
trade_when = profiled(_trade_when)

# Vector
vec_avg = profiled(_vec_avg)
vec_sum = profiled(_vec_sum)

__all__ = [
    # Time-series operators (basic)
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_delta",
    "ts_delay",
    # Time-series operators (rolling)
    "ts_product",
    "ts_count_nans",
    "ts_zscore",
    "ts_scale",
    "ts_av_diff",
    "ts_step",
    # Time-series operators (arg)
    "ts_arg_max",
    "ts_arg_min",
    # Time-series operators (lookback)
    "ts_backfill",
    "kth_element",
    "last_diff_value",
    "days_from_last_change",
    # Time-series operators (stateful)
    "hump",
    "ts_decay_linear",
    "ts_rank",
    # Time-series operators (two-variable)
    "ts_corr",
    "ts_covariance",
    "ts_quantile",
    "ts_regression",
    # Cross-sectional operators
    "bucket",
    "rank",
    "zscore",
    "normalize",
    "scale",
    "quantile",
    "winsorize",
    # Group operators
    "group_rank",
    "group_zscore",
    "group_scale",
    "group_neutralize",
    "group_mean",
    "group_backfill",
    # Vector operators
    "vec_avg",
    "vec_sum",
    # Arithmetic operators
    "abs",
    "add",
    "subtract",
    "multiply",
    "divide",
    "inverse",
    "log",
    "max",
    "min",
    "power",
    "signed_power",
    "sqrt",
    "sign",
    "reverse",
    "densify",
    # Logical operators
    "and_",
    "or_",
    "not_",
    "if_else",
    "is_nan",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    # Transformational operators
    "trade_when",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_operators_are_profiled -v`
Expected: PASS

**Step 5: Run all tests to ensure no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/alphalab/api/operators/__init__.py tests/unit/api/test_profiler.py
git commit -m "feat(profiler): wrap all operators with @profiled"
```

---

## Task 7: Add Public Export and Final Integration Test

**Files:**
- Modify: `src/alphalab/api/__init__.py`
- Test: `tests/unit/api/test_profiler.py`

**Step 1: Write the integration test**

```python
def test_profile_with_client_query(capsys):
    """Full integration: profile() works with client.query()."""
    from alphalab.api.profiler import profile
    import polars as pl

    # Create mock data matching client.get() output format
    df = pl.DataFrame({
        "Date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 10), eager=True),
        "AAPL": [100.0 + i for i in range(10)],
        "MSFT": [200.0 + i for i in range(10)],
    })

    # Test using compute() directly (avoids needing full client setup)
    from alphalab.api.dsl import compute

    with profile() as p:
        result = compute("rank(-ts_delta(x, 2))", x=df)

    # Verify profiling captured the operators
    op_names = {r.operator for r in p.records}
    assert "rank" in op_names
    assert "ts_delta" in op_names

    # Verify summary printed
    captured = capsys.readouterr()
    assert "rank" in captured.out
    assert "ts_delta" in captured.out
```

**Step 2: Run test**

Run: `uv run pytest tests/unit/api/test_profiler.py::test_profile_with_client_query -v`
Expected: PASS

**Step 3: Export profile from api module**

Add to `src/alphalab/api/__init__.py`:

```python
from alphalab.api.profiler import profile

__all__ = [..., "profile"]
```

**Step 4: Run all tests**

Run: `uv run pytest tests/ -q`
Expected: All pass

**Step 5: Commit**

```bash
git add src/alphalab/api/__init__.py tests/unit/api/test_profiler.py
git commit -m "feat(profiler): export profile() from api module"
```

---

## Task 8: Manual Verification with Real Data

**Step 1: Test with combined_alpha.py**

Create a test script or modify temporarily:

```python
# examples/scripts/profile_demo.py
from dotenv import load_dotenv
load_dotenv()

import os
from alphalab.api.client import AlphaLabClient
from alphalab.api.profiler import profile

client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])

with profile():
    result = client.query("rank(-ts_delta(close, 5))")

print(f"Result shape: {result.shape}")
```

**Step 2: Run and verify output**

Run: `uv run python examples/scripts/profile_demo.py`

Expected output similar to:
```
┌────────────┬───────┬──────────┬─────────┬─────────────┐
│ Operator   │ Calls │ Total(s) │ %Total  │ Input Shape │
├────────────┼───────┼──────────┼─────────┼─────────────┤
│ rank       │     1 │    4.521 │  89.2%  │ 2000×5372   │
│ ts_delta   │     1 │    0.548 │  10.8%  │ 2000×5372   │
├────────────┼───────┼──────────┼─────────┼─────────────┤
│ TOTAL      │     2 │    5.069 │ 100.0%  │             │
└────────────┴───────┴──────────┴─────────┴─────────────┘
Result shape: (2000, 5372)
```

**Step 3: Commit demo script**

```bash
git add examples/scripts/profile_demo.py
git commit -m "docs: add profiler demo script"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | ProfileRecord dataclass | profiler.py, test_profiler.py |
| 2 | Profiler class | profiler.py, test_profiler.py |
| 3 | Summary printing | profiler.py, test_profiler.py |
| 4 | Thread-local context manager | profiler.py, test_profiler.py |
| 5 | @profiled decorator | profiler.py, test_profiler.py |
| 6 | Wrap all operators | operators/__init__.py |
| 7 | Public export + integration test | api/__init__.py |
| 8 | Manual verification | profile_demo.py |

**Total: 8 tasks, ~8 commits**
