"""Tests for operator profiler."""

import pytest


def test_profile_record_stores_data():
    """ProfileRecord stores operator name, duration, and shape."""
    from alphalab.api.profiler import ProfileRecord

    record = ProfileRecord(operator="rank", duration=0.5, input_shape=(100, 50))

    assert record.operator == "rank"
    assert record.duration == 0.5
    assert record.input_shape == (100, 50)


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
