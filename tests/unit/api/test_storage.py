"""Tests for StorageBackend."""

from pathlib import Path

import polars as pl
import pytest

from quantdl.api.exceptions import StorageError
from quantdl.api.backend import StorageBackend


@pytest.fixture
def local_storage(test_data_dir: Path) -> StorageBackend:
    """Create storage backend with local test data."""
    return StorageBackend(data_path=test_data_dir)


class TestStorageBasics:
    """Basic storage tests."""

    def test_local_mode_read(self, local_storage: StorageBackend) -> None:
        """Test reading parquet in local mode."""
        df = local_storage.read_parquet("data/meta/master/security_master.parquet")
        assert len(df) > 0
        assert "security_id" in df.columns

    def test_local_mode_scan(self, local_storage: StorageBackend) -> None:
        """Test scanning parquet in local mode."""
        lf = local_storage.scan_parquet("data/meta/master/security_master.parquet")
        df = lf.collect()
        assert len(df) > 0

    def test_exists_local_true(self, local_storage: StorageBackend) -> None:
        """Test exists returns True for existing local file."""
        result = local_storage.exists("data/meta/master/security_master.parquet")
        assert result is True

    def test_exists_local_false(self, local_storage: StorageBackend) -> None:
        """Test exists returns False for non-existing local file."""
        result = local_storage.exists("data/meta/master/nonexistent.parquet")
        assert result is False


class TestStorageErrors:
    """Tests for storage error handling."""

    def test_read_parquet_error_wrapping(self, local_storage: StorageBackend) -> None:
        """Test read_parquet wraps errors in StorageError."""
        with pytest.raises(StorageError) as exc_info:
            local_storage.read_parquet("nonexistent/path.parquet")
        assert "nonexistent/path.parquet" in str(exc_info.value)

    def test_read_parquet_error_has_cause(self, local_storage: StorageBackend) -> None:
        """Test StorageError includes original cause."""
        with pytest.raises(StorageError) as exc_info:
            local_storage.read_parquet("nonexistent/path.parquet")
        assert exc_info.value.cause is not None


class TestPathResolution:
    """Tests for path resolution."""

    def test_resolve_path_local(self, test_data_dir: Path) -> None:
        """Test path resolution in local mode."""
        storage = StorageBackend(data_path=test_data_dir)
        resolved = storage._resolve_path("/data/test.parquet")
        assert str(test_data_dir) in resolved
        assert "data/test.parquet" in resolved.replace("\\", "/")


class TestColumnSelection:
    """Tests for column selection."""

    def test_scan_parquet_with_columns(self, local_storage: StorageBackend) -> None:
        """Test scanning with column selection."""
        lf = local_storage.scan_parquet(
            "data/meta/master/security_master.parquet",
            columns=["security_id", "symbol"],
        )
        df = lf.collect()
        assert df.columns == ["security_id", "symbol"]

    def test_read_parquet_with_columns(self, local_storage: StorageBackend) -> None:
        """Test reading with column selection."""
        df = local_storage.read_parquet(
            "data/meta/master/security_master.parquet",
            columns=["security_id", "symbol"],
        )
        assert df.columns == ["security_id", "symbol"]

    def test_read_parquet_with_filters(self, local_storage: StorageBackend) -> None:
        """Test reading with filters."""
        df = local_storage.read_parquet(
            "data/meta/master/security_master.parquet",
            filters=[pl.col("symbol") == "AAPL"],
        )
        assert len(df) == 1
        assert df["symbol"][0] == "AAPL"


class TestIPCMethods:
    """Tests for Arrow IPC read/scan methods."""

    def test_read_ipc_nonexistent_raises(self, local_storage: StorageBackend) -> None:
        """Test read_ipc wraps errors in StorageError."""
        with pytest.raises(StorageError) as exc_info:
            local_storage.read_ipc("nonexistent.arrow")
        assert "nonexistent.arrow" in str(exc_info.value)

    def test_read_ipc_with_valid_file(self, test_data_dir: Path) -> None:
        """Test read_ipc works with a valid IPC file."""
        # Write a test IPC file
        ipc_path = test_data_dir / "test.arrow"
        df = pl.DataFrame({"a": [1, 2, 3]})
        df.write_ipc(str(ipc_path))

        storage = StorageBackend(data_path=test_data_dir)
        result = storage.read_ipc("test.arrow")
        assert result.equals(df)

    def test_read_ipc_with_columns(self, test_data_dir: Path) -> None:
        """Test read_ipc column selection."""
        ipc_path = test_data_dir / "test2.arrow"
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.write_ipc(str(ipc_path))

        storage = StorageBackend(data_path=test_data_dir)
        result = storage.read_ipc("test2.arrow", columns=["a"])
        assert result.columns == ["a"]
