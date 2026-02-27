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
