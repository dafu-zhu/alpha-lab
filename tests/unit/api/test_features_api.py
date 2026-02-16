"""Tests for features() API method on AlphaLabClient."""

import pytest
import polars as pl
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestFeaturesAPI:
    @pytest.fixture
    def features_dir(self, tmp_path):
        """Create mock feature arrow files."""
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)

        # close.arrow
        df = pl.DataFrame({
            "Date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "100": [150.0, 151.0, 152.0],
            "200": [300.0, 301.0, 302.0],
        })
        df.write_ipc(str(feat_dir / "close.arrow"))

        return tmp_path

    @pytest.fixture
    def client(self, features_dir):
        """Create AlphaLabClient with mock security master."""
        from alphalab.api.types import SecurityInfo
        from alphalab.api.client import AlphaLabClient

        with patch.object(AlphaLabClient, '__init__', lambda self, **kw: None):
            client = AlphaLabClient.__new__(AlphaLabClient)

        # Manually set up internals
        from alphalab.api.backend import StorageBackend
        client._storage = StorageBackend(str(features_dir))

        mock_sm = Mock()

        def resolve_side(ident, as_of=None):
            if ident == "AAPL":
                return SecurityInfo(
                    security_id="100", symbol="AAPL", company="Apple",
                    cik=None,
                    start_date=date(2000, 1, 1), end_date=None,
                )
            if ident == "MSFT":
                return SecurityInfo(
                    security_id="200", symbol="MSFT", company="Microsoft",
                    cik=None,
                    start_date=date(2000, 1, 1), end_date=None,
                )
            return None

        mock_sm.resolve = resolve_side
        client._security_master = mock_sm
        client._calendar_master = Mock()
        client._max_concurrency = 10

        return client

    def test_features_returns_dataframe(self, client):
        result = client.get("close", symbols=["AAPL"])
        assert isinstance(result, pl.DataFrame)
        assert "Date" in result.columns
        assert "AAPL" in result.columns

    def test_features_multiple_symbols(self, client):
        result = client.get("close", symbols=["AAPL", "MSFT"])
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert len(result) == 3

    def test_features_date_filter(self, client):
        result = client.get(
            "close",
            symbols=["AAPL"],
            start="2024-01-03",
            end="2024-01-03",
        )
        assert len(result) == 1
        assert result["Date"][0] == date(2024, 1, 3)

    def test_features_invalid_field_raises(self, client):
        from alphalab.api.exceptions import ValidationError
        with pytest.raises(ValidationError, match="Unknown feature field"):
            client.get("totally_fake_field_xyz")

    def test_features_no_matching_symbols_raises(self, client):
        from alphalab.api.exceptions import DataNotFoundError
        with pytest.raises(DataNotFoundError):
            client.get("close", symbols=["ZZZZ"])

    def test_features_renames_sid_to_symbol(self, client):
        result = client.get("close", symbols=["AAPL"])
        # Column should be "AAPL", not "100"
        assert "AAPL" in result.columns
        assert "100" not in result.columns


class TestQueryAPI:
    @pytest.fixture
    def query_dir(self, tmp_path):
        """Create mock feature arrow files for query tests."""
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)

        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]

        # close.arrow
        close = pl.DataFrame({
            "Date": dates,
            "100": [150.0, 151.0, 152.0],
            "200": [300.0, 301.0, 302.0],
        })
        close.write_ipc(str(feat_dir / "close.arrow"))

        # volume.arrow
        volume = pl.DataFrame({
            "Date": dates,
            "100": [1e6, 1.1e6, 1.2e6],
            "200": [2e6, 2.1e6, 2.2e6],
        })
        volume.write_ipc(str(feat_dir / "volume.arrow"))

        return tmp_path

    @pytest.fixture
    def query_client(self, query_dir):
        """Create AlphaLabClient for query tests."""
        from alphalab.api.types import SecurityInfo
        from alphalab.api.client import AlphaLabClient

        with patch.object(AlphaLabClient, '__init__', lambda self, **kw: None):
            client = AlphaLabClient.__new__(AlphaLabClient)

        from alphalab.api.backend import StorageBackend
        client._storage = StorageBackend(str(query_dir))

        mock_sm = Mock()

        def resolve_side(ident, as_of=None):
            if ident == "AAPL":
                return SecurityInfo(
                    security_id="100", symbol="AAPL", company="Apple",
                    cik=None,
                    start_date=date(2000, 1, 1), end_date=None,
                )
            if ident == "MSFT":
                return SecurityInfo(
                    security_id="200", symbol="MSFT", company="Microsoft",
                    cik=None,
                    start_date=date(2000, 1, 1), end_date=None,
                )
            return None

        mock_sm.resolve = resolve_side
        client._security_master = mock_sm
        client._calendar_master = Mock()
        client._max_concurrency = 10

        return client

    def test_query_simple_expression(self, query_client):
        result = query_client.query("close + 1", symbols=["AAPL"])
        assert isinstance(result, pl.DataFrame)
        assert result["AAPL"][0] == 151.0  # 150 + 1

    def test_query_auto_loads_fields(self, query_client):
        result = query_client.query("close + volume", symbols=["AAPL"])
        assert result["AAPL"][0] == 150.0 + 1e6

    def test_query_with_operators(self, query_client):
        result = query_client.query(
            "rank(close)",
            symbols=["AAPL", "MSFT"],
        )
        # AAPL < MSFT so AAPL rank = 0, MSFT = 1
        assert result["AAPL"][0] == 0.0
        assert result["MSFT"][0] == 1.0

    def test_query_multi_statement(self, query_client):
        result = query_client.query(
            "x = close + 1; x + 1",
            symbols=["AAPL"],
        )
        assert result["AAPL"][0] == 152.0  # 150 + 1 + 1

    def test_query_variable_not_loaded_as_field(self, query_client):
        """Assigned variables should not be treated as field names."""
        # 'x' is assigned, not a field — should not try to load 'x.arrow'
        result = query_client.query(
            "x = close * 2; x + 1",
            symbols=["AAPL"],
        )
        assert result["AAPL"][0] == 301.0  # 150*2 + 1


class TestStorageBackendIPC:
    def test_read_ipc(self, tmp_path):
        from alphalab.api.backend import StorageBackend

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.arrow"
        df.write_ipc(str(path))

        backend = StorageBackend(str(tmp_path))
        result = backend.read_ipc("test.arrow")
        assert result.shape == (3, 2)

    def test_read_ipc_with_columns(self, tmp_path):
        from alphalab.api.backend import StorageBackend

        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        path = tmp_path / "test.arrow"
        df.write_ipc(str(path))

        backend = StorageBackend(str(tmp_path))
        result = backend.read_ipc("test.arrow", columns=["a", "c"])
        assert result.columns == ["a", "c"]

    def test_read_ipc_missing_file_raises(self, tmp_path):
        from alphalab.api.backend import StorageBackend
        from alphalab.api.exceptions import StorageError

        backend = StorageBackend(str(tmp_path))
        with pytest.raises(StorageError):
            backend.read_ipc("nonexistent.arrow")

    def test_scan_ipc(self, tmp_path):
        from alphalab.api.backend import StorageBackend

        df = pl.DataFrame({"x": [10, 20]})
        path = tmp_path / "scan.arrow"
        df.write_ipc(str(path))

        backend = StorageBackend(str(tmp_path))
        lf = backend.scan_ipc("scan.arrow")
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().shape == (2, 1)
