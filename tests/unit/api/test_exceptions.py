"""Tests for custom exceptions."""

import pytest

from alphalab.api.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    AlphaLabError,
    SecurityNotFoundError,
    StorageError,
)


class TestSecurityNotFoundError:
    """Tests for SecurityNotFoundError."""

    def test_security_not_found_without_as_of(self) -> None:
        """Test SecurityNotFoundError message without as_of date."""
        exc = SecurityNotFoundError("AAPL")
        assert exc.identifier == "AAPL"
        assert exc.as_of is None
        assert str(exc) == "Security not found: AAPL"

    def test_security_not_found_with_as_of(self) -> None:
        """Test SecurityNotFoundError message with as_of date."""
        exc = SecurityNotFoundError("AAPL", as_of="2024-01-01")
        assert exc.identifier == "AAPL"
        assert exc.as_of == "2024-01-01"
        assert str(exc) == "Security not found: AAPL as of 2024-01-01"

    def test_security_not_found_is_alphalab_error(self) -> None:
        """Test SecurityNotFoundError inherits from AlphaLabError."""
        exc = SecurityNotFoundError("AAPL")
        assert isinstance(exc, AlphaLabError)


class TestStorageError:
    """Tests for StorageError."""

    def test_storage_error_without_cause(self) -> None:
        """Test StorageError message without cause."""
        exc = StorageError("read", "/path/to/file.parquet")
        assert exc.operation == "read"
        assert exc.path == "/path/to/file.parquet"
        assert exc.cause is None
        assert str(exc) == "Storage read failed for: /path/to/file.parquet"

    def test_storage_error_with_cause(self) -> None:
        """Test StorageError message with cause."""
        cause = ValueError("connection failed")
        exc = StorageError("scan_parquet", "/data/test.parquet", cause=cause)
        assert exc.operation == "scan_parquet"
        assert exc.path == "/data/test.parquet"
        assert exc.cause is cause
        assert "Storage scan_parquet failed for: /data/test.parquet" in str(exc)
        assert "connection failed" in str(exc)

    def test_storage_error_is_alphalab_error(self) -> None:
        """Test StorageError inherits from AlphaLabError."""
        exc = StorageError("read", "/path")
        assert isinstance(exc, AlphaLabError)


class TestDataNotFoundError:
    """Tests for DataNotFoundError."""

    def test_data_not_found_error(self) -> None:
        """Test DataNotFoundError message."""
        exc = DataNotFoundError("daily", "AAPL")
        assert exc.data_type == "daily"
        assert exc.identifier == "AAPL"
        assert str(exc) == "daily data not found for: AAPL"

    def test_data_not_found_is_alphalab_error(self) -> None:
        """Test DataNotFoundError inherits from AlphaLabError."""
        exc = DataNotFoundError("fundamentals", "MSFT")
        assert isinstance(exc, AlphaLabError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError message."""
        exc = ConfigurationError("Missing credentials")
        assert str(exc) == "Missing credentials"

    def test_configuration_error_is_alphalab_error(self) -> None:
        """Test ConfigurationError inherits from AlphaLabError."""
        exc = ConfigurationError("error")
        assert isinstance(exc, AlphaLabError)


class TestAlphaLabError:
    """Tests for base AlphaLabError."""

    def test_alphalab_error_is_exception(self) -> None:
        """Test AlphaLabError is an Exception."""
        exc = AlphaLabError("base error")
        assert isinstance(exc, Exception)
        assert str(exc) == "base error"

    def test_can_raise_alphalab_error(self) -> None:
        """Test AlphaLabError can be raised and caught."""
        with pytest.raises(AlphaLabError):
            raise AlphaLabError("test error")
