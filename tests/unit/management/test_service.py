"""Tests for management service."""
import pytest
import polars as pl


@pytest.fixture
def service(tmp_path):
    """Create service with temp database."""
    from alphalab.management.service import AlphaService

    db_path = tmp_path / "alphas.db"
    return AlphaService(db_path)


def test_submit_new_alpha_no_existing(service):
    """First alpha submission auto-passes."""
    pnl = pl.Series([0.01, -0.02, 0.03])
    result = service.submit(
        alpha_id="test_01",
        name="Test",
        version_num=1,
        expression="rank(close)",
        sharpe=1.5,
        pnl=pnl,
    )

    assert result.passed is True
    assert len(result.details) == 0


def test_submit_blocked_then_force(service):
    """Blocked submission can be force submitted."""
    # Use longer series for valid correlation (need >= 2 data points)
    pnl = pl.Series([0.01, -0.02, 0.03, 0.01, -0.01])

    # First alpha
    service.submit("a", "Alpha A", 1, "x", 1.0, pnl)

    # Second alpha - same PnL, low improvement (5% < 10% threshold)
    result = service.submit("b", "Alpha B", 1, "y", 1.05, pnl)

    assert result.passed is False

    # Force submit
    service.force_submit("b")

    alpha = service.repo.get_alpha("b")
    assert alpha.status == "submitted"
