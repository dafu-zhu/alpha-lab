# Operator Profiler Design

**Date:** 2026-02-27
**Status:** Approved
**Purpose:** Development debugging — find slow operators during development

## Overview

Add operator-level profiling to identify performance bottlenecks in alpha expressions. Activated via context manager with zero meaningful overhead when disabled.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User Code                                              │
│  with profile() as p:                                   │
│      client.query("rank(-ts_delta(close, 5))")          │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Profiler Context Manager                               │
│  - Sets thread-local _active_profiler                   │
│  - On exit: prints summary table                        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  @profiled decorator (on each operator)                 │
│  - Checks if profiler active (fast no-op if not)        │
│  - Records: operator name, duration, input shape        │
└─────────────────────────────────────────────────────────┘
```

## API

### Basic Usage

```python
from alphalab.api.profiler import profile

with profile():
    result = client.query("rank(-ts_delta(close, 5))")
# Prints summary table on exit
```

### Programmatic Access

```python
with profile() as p:
    result = client.query("...")

p.total_time      # float: total seconds
p.records         # list[ProfileRecord]: raw data
p.summary()       # prints table again
```

### ProfileRecord

```python
@dataclass
class ProfileRecord:
    operator: str      # "rank", "ts_delta", etc.
    duration: float    # seconds
    input_shape: tuple # (rows, cols) for debugging
```

### Console Output

```
┌────────────┬───────┬──────────┬─────────┬─────────────┐
│ Operator   │ Calls │ Total(s) │ %Total  │ Input Shape │
├────────────┼───────┼──────────┼─────────┼─────────────┤
│ rank       │     2 │    0.045 │  85.2%  │ 2000×5372   │
│ ts_delta   │     1 │    0.008 │  14.8%  │ 2000×5372   │
├────────────┼───────┼──────────┼─────────┼─────────────┤
│ TOTAL      │     3 │    0.053 │ 100.0%  │             │
└────────────┴───────┴──────────┴─────────┴─────────────┘
```

## Implementation

### Thread-local Profiler

```python
import threading
from contextlib import contextmanager

_local = threading.local()

def _get_profiler() -> Profiler | None:
    return getattr(_local, 'profiler', None)

@contextmanager
def profile():
    p = Profiler()
    _local.profiler = p
    try:
        yield p
    finally:
        _local.profiler = None
        p._print_summary()
```

### Profiled Decorator

```python
def profiled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        p = _get_profiler()
        if p is None:
            return fn(*args, **kwargs)  # No overhead path

        shape = _get_input_shape(args)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration = time.perf_counter() - start
        p.record(fn.__name__, duration, shape)
        return result
    return wrapper
```

### Applying to Operators

Auto-wrap in `operators/__init__.py`:

```python
from alphalab.api.profiler import profiled
from alphalab.api.operators.cross_sectional import rank as _rank, ...

rank = profiled(_rank)
ts_delta = profiled(_ts_delta)
```

## Performance Impact

**When profiling OFF:** ~50-100ns per operator call (one `getattr` + `if` check)

**Realistic overhead:** Query with 5 operators adds ~500ns vs operators taking 5-50ms each. Overhead ratio: 0.001% - 0.01%. Negligible.

## Files Changed

- `src/alphalab/api/profiler.py` — new module
- `src/alphalab/api/operators/__init__.py` — wrap operators with `@profiled`

## Testing

### Unit Tests

- `test_profiler_context_manager` — enters/exits cleanly, prints summary
- `test_profiler_records_operator_calls` — records name, duration, shape
- `test_profiler_zero_overhead_when_off` — verify no slowdown without profiling
- `test_profiler_nested_calls` — operators calling operators recorded correctly

### Integration Test

```python
def test_profile_query_end_to_end():
    with profile() as p:
        client.query("rank(-ts_delta(close, 5))")

    assert len(p.records) == 2
    assert {r.operator for r in p.records} == {"rank", "ts_delta"}
    assert p.total_time > 0
```

### Manual Verification

Run `combined_alpha.py` with profiling, confirm `rank` shows ~5s (post-optimization).
