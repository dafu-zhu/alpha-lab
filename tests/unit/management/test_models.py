"""Tests for data models."""
from datetime import datetime


def test_alpha_model_creation():
    """Alpha model stores required fields."""
    from alphalab.management.models import Alpha

    alpha = Alpha(id="momentum_01", name="Momentum Alpha", status="draft")

    assert alpha.id == "momentum_01"
    assert alpha.name == "Momentum Alpha"
    assert alpha.status == "draft"


def test_version_model_creation():
    """Version model stores expression and metrics."""
    from alphalab.management.models import Version

    version = Version(
        id="momentum_01-v1",
        alpha_id="momentum_01",
        version_num=1,
        expression="rank(-ts_delta(close, 5))",
        sharpe=1.5,
    )

    assert version.id == "momentum_01-v1"
    assert version.alpha_id == "momentum_01"
    assert version.sharpe == 1.5


def test_correlation_model_creation():
    """Correlation model stores check results."""
    from alphalab.management.models import Correlation

    corr = Correlation(
        new_alpha_id="momentum_01-v1",
        exist_alpha_id="mean_rev_02-v1",
        corr=0.82,
        new_sharpe=1.6,
        exist_sharpe=1.5,
        improvement=6.67,
        passed=False,
    )

    assert corr.corr == 0.82
    assert corr.passed is False


def test_check_result_model():
    """CheckResult aggregates correlation checks."""
    from alphalab.management.models import Correlation, CheckResult

    results = [
        Correlation("a-v1", "b-v1", 0.45, 1.6, 1.4, 14.3, True),
        Correlation("a-v1", "c-v1", 0.82, 1.6, 1.5, 6.7, False),
    ]

    check = CheckResult(results)

    assert check.passed is False
    assert len(check.blocking) == 1
    assert check.blocking[0].exist_alpha_id == "c-v1"
