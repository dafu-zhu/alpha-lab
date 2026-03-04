"""Input validation utilities for data integrity."""


def validate_year(year: int, min_year: int = 1900, max_year: int = 2100) -> int:
    """Validate year parameter to ensure it's within reasonable bounds.

    :param year: Year to validate
    :param min_year: Minimum valid year (default: 1900)
    :param max_year: Maximum valid year (default: 2100)
    :return: Validated year
    :raises ValueError: If year is invalid
    """
    if not isinstance(year, int) or year < min_year or year > max_year:
        raise ValueError(
            f"Invalid year: {year}. Must be an integer between {min_year} and {max_year}"
        )
    return year
