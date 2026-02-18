"""Tests for management module exports."""


def test_public_exports():
    """Management module exports key classes."""
    from alphalab.management import (
        Alpha,
        Version,
        Correlation,
        CheckResult,
        AlphaRepository,
        AlphaService,
        check_correlation,
    )

    assert Alpha is not None
    assert AlphaService is not None
