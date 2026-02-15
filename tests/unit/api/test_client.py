"""Tests for QuantDLClient."""

from pathlib import Path

import pytest

from quantdl.api import QuantDLClient
from quantdl.api.exceptions import DataNotFoundError


@pytest.fixture
def client(test_data_dir: Path) -> QuantDLClient:
    """Create client with local test data."""
    return QuantDLClient(data_path=str(test_data_dir))


class TestClientBasics:
    """Basic client tests."""

    def test_client_creation(self, test_data_dir: Path) -> None:
        """Test client can be created."""
        client = QuantDLClient(data_path=str(test_data_dir))
        assert client is not None

    def test_client_context_manager(self, test_data_dir: Path) -> None:
        """Test client as context manager."""
        with QuantDLClient(data_path=str(test_data_dir)) as client:
            assert client is not None


class TestSecurityLookup:
    """Security lookup tests."""

    def test_lookup_symbol(self, client: QuantDLClient) -> None:
        """Test looking up symbol."""
        info = client.lookup("AAPL")
        assert info is not None
        assert info.symbol == "AAPL"

    def test_lookup_missing(self, client: QuantDLClient) -> None:
        """Test looking up missing symbol."""
        info = client.lookup("INVALID")
        assert info is None


class TestUniverse:
    """Universe loading tests."""

    def test_load_universe(self, client: QuantDLClient) -> None:
        """Test loading universe."""
        symbols = client.universe("top3000")
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols

    def test_invalid_universe(self, client: QuantDLClient) -> None:
        """Test loading invalid universe."""
        with pytest.raises(DataNotFoundError):
            client.universe("invalid_universe")
