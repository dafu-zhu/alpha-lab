"""Custom exceptions for AlphaLab API."""


class AlphaLabError(Exception):
    """Base exception for AlphaLab errors."""


class SecurityNotFoundError(AlphaLabError):
    """Raised when a security cannot be resolved."""

    def __init__(self, identifier: str, as_of: str | None = None) -> None:
        self.identifier = identifier
        self.as_of = as_of
        msg = f"Security not found: {identifier}"
        if as_of:
            msg += f" as of {as_of}"
        super().__init__(msg)


class DataNotFoundError(AlphaLabError):
    """Raised when requested data is not available."""

    def __init__(self, data_type: str, identifier: str) -> None:
        self.data_type = data_type
        self.identifier = identifier
        super().__init__(f"{data_type} data not found for: {identifier}")


class StorageError(AlphaLabError):
    """Raised when storage operations fail."""

    def __init__(self, operation: str, path: str, cause: Exception | None = None) -> None:
        self.operation = operation
        self.path = path
        self.cause = cause
        msg = f"Storage {operation} failed for: {path}"
        if cause:
            msg += f" ({cause})"
        super().__init__(msg)


class ConfigurationError(AlphaLabError):
    """Raised when configuration is invalid or missing."""


class ValidationError(AlphaLabError):
    """Raised when input validation fails."""
