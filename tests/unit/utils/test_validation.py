"""
Unit tests for utils.validation module
Tests input validation for SQL injection prevention and data integrity
"""
import pytest
from alphalab.utils.validation import (
    validate_date_string,
    validate_year,
    validate_month
)


class TestValidateDateString:
    """Test validate_date_string function"""

    def test_valid_date(self):
        """Test validation of valid date strings"""
        assert validate_date_string('2024-01-15') == '2024-01-15'
        assert validate_date_string('2024-12-31') == '2024-12-31'
        assert validate_date_string('2020-02-29') == '2020-02-29'  # Leap year

    def test_invalid_format(self):
        """Test rejection of invalid date formats"""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_string('2024/01/15')  # Wrong separator

        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_string('01-15-2024')  # Wrong order

        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_string('2024-1-15')  # Missing zero padding

        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_string('24-01-15')  # 2-digit year

    def test_sql_injection_attempt(self):
        """Test rejection of SQL injection attempts"""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_string("'; DROP TABLE--")

        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_string("2024-01-15; DELETE FROM users")

        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_string("2024-01-15' OR '1'='1")

    def test_invalid_date_values(self):
        """Test rejection of invalid date values"""
        with pytest.raises(ValueError, match="Invalid date value"):
            validate_date_string('2024-13-01')  # Invalid month

        with pytest.raises(ValueError, match="Invalid date value"):
            validate_date_string('2024-00-15')  # Invalid month

        with pytest.raises(ValueError, match="Invalid date value"):
            validate_date_string('2024-01-32')  # Invalid day

        with pytest.raises(ValueError, match="Invalid date value"):
            validate_date_string('2024-02-30')  # Invalid day for February

        with pytest.raises(ValueError, match="Invalid date value"):
            validate_date_string('2023-02-29')  # Not a leap year

    def test_edge_cases(self):
        """Test edge cases for date validation"""
        # Valid edge cases
        assert validate_date_string('2024-01-01') == '2024-01-01'  # First day of year
        assert validate_date_string('2024-12-31') == '2024-12-31'  # Last day of year
        assert validate_date_string('2000-02-29') == '2000-02-29'  # Leap year



class TestValidateYear:
    """Test validate_year function"""

    def test_valid_year(self):
        """Test validation of valid years"""
        assert validate_year(2024) == 2024
        assert validate_year(1900) == 1900
        assert validate_year(2100) == 2100
        assert validate_year(2000) == 2000

    def test_year_out_of_range(self):
        """Test rejection of years out of range"""
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(1800)  # Below minimum

        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2101)  # Above maximum

        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(1899)  # Just below minimum

    def test_custom_range(self):
        """Test validation with custom year range"""
        assert validate_year(2020, min_year=2010, max_year=2030) == 2020

        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2009, min_year=2010, max_year=2030)

        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2031, min_year=2010, max_year=2030)

    def test_non_integer_year(self):
        """Test rejection of non-integer year"""
        with pytest.raises(ValueError, match="Invalid year"):
            validate_year("2024")

        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(2024.5)

        with pytest.raises(ValueError, match="Invalid year"):
            validate_year(None)

    def test_boundary_values(self):
        """Test boundary values for year validation"""
        # Default range boundaries
        assert validate_year(1900) == 1900
        assert validate_year(2100) == 2100

        # Custom range boundaries
        assert validate_year(2010, min_year=2010, max_year=2020) == 2010
        assert validate_year(2020, min_year=2010, max_year=2020) == 2020


class TestValidateMonth:
    """Test validate_month function"""

    def test_valid_month(self):
        """Test validation of valid months"""
        assert validate_month(1) == 1
        assert validate_month(6) == 6
        assert validate_month(12) == 12

    def test_month_out_of_range(self):
        """Test rejection of months out of range"""
        with pytest.raises(ValueError, match="Invalid month"):
            validate_month(0)

        with pytest.raises(ValueError, match="Invalid month"):
            validate_month(13)

        with pytest.raises(ValueError, match="Invalid month"):
            validate_month(-1)

        with pytest.raises(ValueError, match="Invalid month"):
            validate_month(100)

    def test_non_integer_month(self):
        """Test rejection of non-integer month"""
        with pytest.raises(ValueError, match="Invalid month"):
            validate_month("6")

        with pytest.raises(ValueError, match="Invalid month"):
            validate_month(6.5)

        with pytest.raises(ValueError, match="Invalid month"):
            validate_month(None)

    def test_boundary_values(self):
        """Test boundary values for month validation"""
        assert validate_month(1) == 1
        assert validate_month(12) == 12
