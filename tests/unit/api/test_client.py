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


class TestFundamentals:
    """Fundamentals API tests."""

    def test_fundamentals_basic(self, client: QuantDLClient) -> None:
        """Test basic fundamentals fetch."""
        df = client.fundamentals("AAPL", "Revenue", "2024-01-01", "2024-12-31")

        assert "Date" in df.columns
        assert "AAPL" in df.columns
        assert len(df) > 0

    def test_fundamentals_invalid_symbol(self, client: QuantDLClient) -> None:
        """Test fundamentals with invalid symbol."""
        with pytest.raises(DataNotFoundError):
            client.fundamentals("INVALID", "Revenue", "2024-01-01", "2024-12-31")


class TestMetrics:
    """Metrics API tests."""

    def test_metrics_derived_removed(self, client: QuantDLClient) -> None:
        """Test derived metrics raise DataNotFoundError (derived module removed)."""
        with pytest.raises(DataNotFoundError):
            client.metrics("AAPL", "pe_ratio", "2024-01-01", "2024-12-31")

    def test_metrics_invalid_symbol(self, client: QuantDLClient) -> None:
        """Test metrics with invalid symbol."""
        with pytest.raises(DataNotFoundError):
            client.metrics("INVALID", "pe_ratio", "2024-01-01", "2024-12-31")


class TestTicksFieldNotFound:
    """Tests for ticks field not found error."""

    def test_ticks_field_not_found(self, client: QuantDLClient) -> None:
        """Test ticks with invalid field raises DataNotFoundError."""
        with pytest.raises(DataNotFoundError) as exc_info:
            client.ticks("AAPL", field="nonexistent_field", start="2024-01-01", end="2024-01-10")
        assert "field=nonexistent_field" in str(exc_info.value)


class TestFundamentalsConceptNotFound:
    """Tests for fundamentals concept not found error."""

    def test_fundamentals_concept_not_found(self, client: QuantDLClient) -> None:
        """Test fundamentals with invalid concept raises DataNotFoundError."""
        with pytest.raises(DataNotFoundError) as exc_info:
            client.fundamentals("AAPL", concept="NonexistentConcept", start="2024-01-01", end="2024-12-31")
        assert "concept=NonexistentConcept" in str(exc_info.value)


class TestMetricsFieldNotFound:
    """Tests for metrics field not found error."""

    def test_metrics_field_not_found(self, client: QuantDLClient) -> None:
        """Test metrics with invalid metric raises DataNotFoundError."""
        with pytest.raises(DataNotFoundError) as exc_info:
            client.metrics("AAPL", metric="nonexistent_metric", start="2024-01-01", end="2024-12-31")
        assert "metric=nonexistent_metric" in str(exc_info.value)
