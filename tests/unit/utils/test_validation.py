"""Unit tests for utils.validation module."""

import pytest
from alphalab.utils.validation import validate_year


class TestValidateYear:
    """Test validate_year function"""

    def test_valid_year(self):
        assert validate_year(2024) == 2024
        assert validate_year(1900) == 1900
        assert validate_year(2100) == 2100
        assert validate_year(2000) == 2000

    def test_year_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(1800)
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2101)
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(1899)

    def test_custom_range(self):
        assert validate_year(2020, min_year=2010, max_year=2030) == 2020
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2009, min_year=2010, max_year=2030)
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2031, min_year=2010, max_year=2030)

    def test_non_integer_year(self):
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year("2024")
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2024.5)
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(None)

    def test_boundary_values(self):
        assert validate_year(1900) == 1900
        assert validate_year(2100) == 2100
        assert validate_year(2010, min_year=2010, max_year=2020) == 2010
        assert validate_year(2020, min_year=2010, max_year=2020) == 2020
