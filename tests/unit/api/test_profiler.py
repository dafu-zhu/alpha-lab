"""Tests for operator profiler."""

import pytest


def test_profile_record_stores_data():
    """ProfileRecord stores operator name, duration, and shape."""
    from alphalab.api.profiler import ProfileRecord

    record = ProfileRecord(operator="rank", duration=0.5, input_shape=(100, 50))

    assert record.operator == "rank"
    assert record.duration == 0.5
    assert record.input_shape == (100, 50)
