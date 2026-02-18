"""Tests for alpha repository."""
import json
import pytest


@pytest.fixture
def repo(tmp_path):
    """Create repository with temp database."""
    from alphalab.management.repository import AlphaRepository

    db_path = tmp_path / "alphas.db"
    return AlphaRepository(db_path)


def test_create_alpha(repo):
    """create_alpha inserts new alpha."""
    from alphalab.management.models import Alpha

    alpha = Alpha(id="test_01", name="Test Alpha")
    repo.create_alpha(alpha)

    result = repo.get_alpha("test_01")
    assert result is not None
    assert result.name == "Test Alpha"
    assert result.status == "draft"


def test_create_version(repo):
    """create_version inserts new version."""
    from alphalab.management.models import Alpha, Version

    repo.create_alpha(Alpha(id="test_01", name="Test"))

    version = Version(
        id="test_01-v1",
        alpha_id="test_01",
        version_num=1,
        expression="rank(close)",
        sharpe=1.5,
        pnl_data=[{"date": "2024-01-01", "ret": 0.01}],
    )
    repo.create_version(version)

    result = repo.get_version("test_01-v1")
    assert result is not None
    assert result.sharpe == 1.5
    assert result.pnl_data == [{"date": "2024-01-01", "ret": 0.01}]


def test_get_submitted_versions(repo):
    """get_submitted returns only submitted alphas."""
    from alphalab.management.models import Alpha, Version

    repo.create_alpha(Alpha(id="a", name="A", status="submitted"))
    repo.create_alpha(Alpha(id="b", name="B", status="draft"))

    repo.create_version(Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=[]))
    repo.create_version(Version("b-v1", "b", 1, "y", sharpe=1.0, pnl_data=[]))

    submitted = repo.get_submitted()

    assert len(submitted) == 1
    assert submitted[0].id == "a-v1"


def test_submit_alpha(repo):
    """submit_alpha changes status and inserts correlations."""
    from alphalab.management.models import Alpha, Version, Correlation

    repo.create_alpha(Alpha(id="a", name="A"))
    repo.create_version(Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=[]))

    correlations = [
        Correlation("a-v1", "exist-v1", 0.5, 1.0, 0.9, 11.1, True),
    ]
    repo.submit_alpha("a", correlations)

    alpha = repo.get_alpha("a")
    assert alpha.status == "submitted"

    corrs = repo.get_correlations("a-v1")
    assert len(corrs) == 1


def test_delete_alpha_cascades(repo):
    """delete_alpha removes alpha, versions, and correlations."""
    from alphalab.management.models import Alpha, Version, Correlation

    repo.create_alpha(Alpha(id="a", name="A", status="submitted"))
    repo.create_version(Version("a-v1", "a", 1, "x", sharpe=1.0, pnl_data=[]))

    repo.create_alpha(Alpha(id="b", name="B", status="submitted"))
    repo.create_version(Version("b-v1", "b", 1, "y", sharpe=1.0, pnl_data=[]))

    # b was submitted after a
    correlations = [Correlation("b-v1", "a-v1", 0.5, 1.0, 0.9, 11.1, True)]
    repo.insert_correlations(correlations)

    # Delete a
    repo.delete_alpha("a")

    assert repo.get_alpha("a") is None
    assert repo.get_version("a-v1") is None
    # Correlation referencing a should be deleted
    assert len(repo.get_correlations("b-v1")) == 0
