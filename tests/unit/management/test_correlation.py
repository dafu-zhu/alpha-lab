"""Tests for correlation check logic."""
import polars as pl
import pytest


def test_check_single_alpha_low_corr_passes():
    """corr < 0.7 passes regardless of sharpe improvement."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    new_pnl = pl.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    new_sharpe = 1.0

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=1.5,
            pnl_data=[
                {"date": "2024-01-01", "ret": 0.05},
                {"date": "2024-01-02", "ret": -0.03},
                {"date": "2024-01-03", "ret": 0.02},
                {"date": "2024-01-04", "ret": -0.01},
                {"date": "2024-01-05", "ret": 0.04},
            ],
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is True
    assert len(result.blocking) == 0


def test_check_high_corr_low_improvement_fails():
    """corr >= 0.7 AND improvement < 10% fails."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    # Same data = corr = 1.0
    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 1.05  # 5% improvement < 10%

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=1.0,
            pnl_data=pnl_data,
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is False
    assert len(result.blocking) == 1
    assert result.blocking[0].exist_alpha_id == "exist-v1"


def test_check_zero_sharpe_positive_new_sharpe():
    """Zero existing sharpe with positive new sharpe gives infinite improvement."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 1.0

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=0.0,  # Zero sharpe
            pnl_data=pnl_data,
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    # Should pass: improvement is inf >= 10%
    assert result.passed is True
    assert len(result.details) == 1
    assert result.details[0].improvement == float('inf')


def test_check_zero_sharpe_zero_new_sharpe():
    """Zero existing sharpe with zero new sharpe gives zero improvement."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 0.0

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=0.0,  # Zero sharpe
            pnl_data=pnl_data,
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    # Should fail: corr=1.0 and improvement=0% < 10%
    assert result.passed is False
    assert len(result.details) == 1
    assert result.details[0].improvement == 0.0


def test_check_high_corr_high_improvement_passes():
    """corr >= 0.7 AND improvement >= 10% passes."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 1.2  # 20% improvement >= 10%

    existing = [
        Version(
            id="exist-v1",
            alpha_id="exist",
            version_num=1,
            expression="x",
            sharpe=1.0,
            pnl_data=pnl_data,
        )
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is True


def test_check_no_existing_alphas_passes():
    """No submitted alphas = auto-pass."""
    from alphalab.management.correlation import check_correlation

    new_pnl = pl.Series([0.01, -0.02, 0.03])
    result = check_correlation(new_pnl, 1.5, [])

    assert result.passed is True
    assert len(result.details) == 0


def test_check_multiple_alphas_one_blocks():
    """Multiple existing alphas, one blocks."""
    from alphalab.management.correlation import check_correlation
    from alphalab.management.models import Version

    pnl_data = [
        {"date": "2024-01-01", "ret": 0.01},
        {"date": "2024-01-02", "ret": -0.02},
        {"date": "2024-01-03", "ret": 0.03},
    ]
    new_pnl = pl.Series([d["ret"] for d in pnl_data])
    new_sharpe = 1.05

    existing = [
        Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=pnl_data),  # blocks
        Version(
            "b-v1",
            "b",
            1,
            "y",
            sharpe=2.0,
            pnl_data=[
                {"date": "2024-01-01", "ret": 0.05},
                {"date": "2024-01-02", "ret": 0.01},
                {"date": "2024-01-03", "ret": -0.04},
            ],
        ),  # different data, low corr
    ]

    result = check_correlation(new_pnl, new_sharpe, existing)

    assert result.passed is False
    assert len(result.blocking) == 1
