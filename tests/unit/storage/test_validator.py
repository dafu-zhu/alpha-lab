"""
Unit tests for storage.validation.Validator class
Tests data validation and existence checking (local-only storage)
"""
import os
import pytest
from unittest.mock import Mock, patch
import polars as pl

from alphalab.storage.clients import LocalStorageClient


def _write_ticks_parquet(path, year=2024, rows=5):
    """Helper to create a real ticks parquet file with timestamp data."""
    dates = [f"{year}-01-{i+1:02d}" for i in range(rows)]
    df = pl.DataFrame({
        "timestamp": dates,
        "open": [100.0] * rows,
        "high": [101.0] * rows,
        "low": [99.0] * rows,
        "close": [100.5] * rows,
        "volume": [1000] * rows,
    })
    df.write_parquet(path)


class TestValidatorLocal:
    """Test Validator in local storage mode."""

    def test_data_exists_ticks_local(self, tmp_path):
        """Test data_exists for ticks in local mode."""
        from alphalab.storage.pipeline import Validator
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "12345"
        ticks_dir.mkdir(parents=True)
        _write_ticks_parquet(ticks_dir / "ticks.parquet", year=2024)

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        assert validator.data_exists("AAPL", "ticks", year=2024, security_id=12345) is True
        assert validator.data_exists("AAPL", "ticks", year=2024, security_id=99999) is False

    def test_data_exists_fundamental_local(self, tmp_path):
        """Test data_exists for fundamental in local mode."""
        from alphalab.storage.pipeline import Validator
        fnd_dir = tmp_path / "data" / "raw" / "fundamental" / "12345"
        fnd_dir.mkdir(parents=True)
        (fnd_dir / "fundamental.parquet").write_text("dummy")

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        assert validator.data_exists("AAPL", "fundamental", security_id=12345) is True

    def test_top_3000_exists_local(self, tmp_path):
        """Test top_3000_exists in local mode."""
        from alphalab.storage.pipeline import Validator
        uni_dir = tmp_path / "data" / "meta" / "universe" / "2024" / "06"
        uni_dir.mkdir(parents=True)
        (uni_dir / "top3000.txt").write_text("AAPL\nMSFT")

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        assert validator.top_3000_exists(2024, 6) is True
        assert validator.top_3000_exists(2024, 7) is False

    def test_list_files_local(self, tmp_path):
        """Test list_files_under_prefix in local mode."""
        from alphalab.storage.pipeline import Validator
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "SEC001"
        ticks_dir.mkdir(parents=True)
        (ticks_dir / "ticks.parquet").write_text("dummy")

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        files = validator.list_files_under_prefix("data/raw/ticks")
        assert len(files) >= 1
        assert any("ticks.parquet" in f for f in files)

    def test_list_files_local_empty(self, tmp_path):
        """Test list_files_under_prefix for nonexistent prefix."""
        from alphalab.storage.pipeline import Validator

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        files = validator.list_files_under_prefix("data/nonexistent")
        assert files == []

    def test_data_exists_ticks_with_security_id(self, tmp_path):
        """Test data_exists for ticks uses security_id-based single file path."""
        from alphalab.storage.pipeline import Validator

        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "12345"
        ticks_dir.mkdir(parents=True)
        _write_ticks_parquet(ticks_dir / "ticks.parquet", year=2024)

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        result = validator.data_exists("AAPL", "ticks", year=2024, security_id=12345)
        assert result is True

    def test_data_exists_ticks_without_security_id(self, tmp_path):
        """Test data_exists for ticks uses symbol as fallback identifier."""
        from alphalab.storage.pipeline import Validator

        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "AAPL"
        ticks_dir.mkdir(parents=True)
        _write_ticks_parquet(ticks_dir / "ticks.parquet", year=2024)

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        result = validator.data_exists("AAPL", "ticks", year=2024)
        assert result is True

    def test_data_exists_ticks_year_check(self, tmp_path):
        """Test data_exists checks for specific year within ticks file."""
        from alphalab.storage.pipeline import Validator

        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "12345"
        ticks_dir.mkdir(parents=True)
        _write_ticks_parquet(ticks_dir / "ticks.parquet", year=2024)

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        # Year 2024 exists in file
        assert validator.data_exists("AAPL", "ticks", year=2024, security_id=12345) is True
        # Year 2023 does NOT exist in file
        assert validator.data_exists("AAPL", "ticks", year=2023, security_id=12345) is False

    def test_data_exists_fundamental_with_security_id(self, tmp_path):
        """Test data_exists for fundamental uses security_id-based path."""
        from alphalab.storage.pipeline import Validator

        fnd_dir = tmp_path / "data" / "raw" / "fundamental" / "12345"
        fnd_dir.mkdir(parents=True)
        (fnd_dir / "fundamental.parquet").write_text("dummy")

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        result = validator.data_exists("AAPL", "fundamental", security_id=12345)
        assert result is True

    def test_data_exists_fundamental_with_cik(self, tmp_path):
        """Test data_exists for fundamental uses CIK as fallback."""
        from alphalab.storage.pipeline import Validator

        fnd_dir = tmp_path / "data" / "raw" / "fundamental" / "0000320193"
        fnd_dir.mkdir(parents=True)
        (fnd_dir / "fundamental.parquet").write_text("dummy")

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        result = validator.data_exists("AAPL", "fundamental", cik="0000320193")
        assert result is True

    def test_data_exists_not_found(self, tmp_path):
        """Test data_exists returns False when file not found."""
        from alphalab.storage.pipeline import Validator

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        result = validator.data_exists("AAPL", "ticks", year=2024, security_id=12345)
        assert result is False

    def test_data_exists_invalid_type_raises(self, tmp_path):
        """Test data_exists raises ValueError for invalid data_type."""
        from alphalab.storage.pipeline import Validator

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)

        with pytest.raises(ValueError, match="Expected data_type"):
            validator.data_exists("AAPL", "invalid_type")

    def test_top_3000_exists_true(self, tmp_path):
        """Test top_3000_exists returns True when file exists."""
        from alphalab.storage.pipeline import Validator

        uni_dir = tmp_path / "data" / "meta" / "universe" / "2024" / "06"
        uni_dir.mkdir(parents=True)
        (uni_dir / "top3000.txt").write_text("AAPL\nMSFT")

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        result = validator.top_3000_exists(2024, 6)
        assert result is True

    def test_top_3000_exists_false(self, tmp_path):
        """Test top_3000_exists returns False when file missing."""
        from alphalab.storage.pipeline import Validator

        client = LocalStorageClient(str(tmp_path))
        validator = Validator(storage_client=client)
        result = validator.top_3000_exists(2024, 6)
        assert result is False

    def test_list_files_env_fallback(self, tmp_path):
        """Test list_files_under_prefix falls back to LOCAL_STORAGE_PATH env var."""
        from alphalab.storage.pipeline import Validator
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "SEC001"
        ticks_dir.mkdir(parents=True)
        (ticks_dir / "ticks.parquet").write_text("dummy")

        with patch.dict(os.environ, {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            validator = Validator()
            files = validator.list_files_under_prefix("data/raw/ticks")
            assert len(files) >= 1
            assert any("ticks.parquet" in f for f in files)
