"""Tests for features() API method on QuantDLClient."""

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
        """Create QuantDLClient with mock security master."""
        from quantdl.api.types import SecurityInfo
        from quantdl.api.client import QuantDLClient

        with patch.object(QuantDLClient, '__init__', lambda self, **kw: None):
            client = QuantDLClient.__new__(QuantDLClient)

        # Manually set up internals
        from quantdl.api.backend import StorageBackend
        client._storage = StorageBackend(str(features_dir))

        mock_sm = Mock()

        def resolve_side(ident, as_of=None):
            if ident == "AAPL":
                return SecurityInfo(
                    security_id="100", symbol="AAPL", company="Apple",
                    cik=None, cusip=None,
                    start_date=date(2000, 1, 1), end_date=None,
                )
            if ident == "MSFT":
                return SecurityInfo(
                    security_id="200", symbol="MSFT", company="Microsoft",
                    cik=None, cusip=None,
                    start_date=date(2000, 1, 1), end_date=None,
                )
            return None

        mock_sm.resolve = resolve_side
        client._security_master = mock_sm
        client._calendar_master = Mock()
        client._max_concurrency = 10

        from concurrent.futures import ThreadPoolExecutor
        client._executor = ThreadPoolExecutor(max_workers=2)

        return client

    def test_features_returns_dataframe(self, client):
        result = client.features("close", symbols=["AAPL"])
        assert isinstance(result, pl.DataFrame)
        assert "Date" in result.columns
        assert "AAPL" in result.columns

    def test_features_multiple_symbols(self, client):
        result = client.features("close", symbols=["AAPL", "MSFT"])
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert len(result) == 3

    def test_features_date_filter(self, client):
        result = client.features(
            "close",
            symbols=["AAPL"],
            start="2024-01-03",
            end="2024-01-03",
        )
        assert len(result) == 1
        assert result["Date"][0] == date(2024, 1, 3)

    def test_features_invalid_field_raises(self, client):
        from quantdl.api.exceptions import ValidationError
        with pytest.raises(ValidationError, match="Unknown feature field"):
            client.features("totally_fake_field_xyz")

    def test_features_no_matching_symbols_raises(self, client):
        from quantdl.api.exceptions import DataNotFoundError
        with pytest.raises(DataNotFoundError):
            client.features("close", symbols=["ZZZZ"])

    def test_features_renames_sid_to_symbol(self, client):
        result = client.features("close", symbols=["AAPL"])
        # Column should be "AAPL", not "100"
        assert "AAPL" in result.columns
        assert "100" not in result.columns


class TestStorageBackendIPC:
    def test_read_ipc(self, tmp_path):
        from quantdl.api.backend import StorageBackend

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.arrow"
        df.write_ipc(str(path))

        backend = StorageBackend(str(tmp_path))
        result = backend.read_ipc("test.arrow")
        assert result.shape == (3, 2)

    def test_read_ipc_with_columns(self, tmp_path):
        from quantdl.api.backend import StorageBackend

        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        path = tmp_path / "test.arrow"
        df.write_ipc(str(path))

        backend = StorageBackend(str(tmp_path))
        result = backend.read_ipc("test.arrow", columns=["a", "c"])
        assert result.columns == ["a", "c"]

    def test_read_ipc_missing_file_raises(self, tmp_path):
        from quantdl.api.backend import StorageBackend
        from quantdl.api.exceptions import StorageError

        backend = StorageBackend(str(tmp_path))
        with pytest.raises(StorageError):
            backend.read_ipc("nonexistent.arrow")

    def test_scan_ipc(self, tmp_path):
        from quantdl.api.backend import StorageBackend

        df = pl.DataFrame({"x": [10, 20]})
        path = tmp_path / "scan.arrow"
        df.write_ipc(str(path))

        backend = StorageBackend(str(tmp_path))
        lf = backend.scan_ipc("scan.arrow")
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().shape == (2, 1)
