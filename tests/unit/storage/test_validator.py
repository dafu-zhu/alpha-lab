"""
Unit tests for storage.validation.Validator class
Tests data validation and existence checking
"""
import os
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError


@patch.dict(os.environ, {'STORAGE_BACKEND': 'local'})
class TestValidatorLocal:
    """Test Validator in local storage mode."""

    def test_data_exists_ticks_local(self, tmp_path):
        """Test data_exists for ticks in local mode."""
        from quantdl.storage.pipeline import Validator
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "12345"
        ticks_dir.mkdir(parents=True)
        (ticks_dir / "ticks.parquet").write_text("dummy")

        with patch.dict(os.environ, {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            validator = Validator(s3_client=Mock(), bucket_name="test-bucket")
            assert validator.data_exists("AAPL", "ticks", year=2024, security_id=12345) is True
            assert validator.data_exists("AAPL", "ticks", year=2024, security_id=99999) is False

    def test_data_exists_fundamental_local(self, tmp_path):
        """Test data_exists for fundamental in local mode."""
        from quantdl.storage.pipeline import Validator
        fnd_dir = tmp_path / "data" / "raw" / "fundamental" / "12345"
        fnd_dir.mkdir(parents=True)
        (fnd_dir / "fundamental.parquet").write_text("dummy")

        with patch.dict(os.environ, {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            validator = Validator(s3_client=Mock(), bucket_name="test-bucket")
            assert validator.data_exists("AAPL", "fundamental", security_id=12345) is True

    def test_top_3000_exists_local(self, tmp_path):
        """Test top_3000_exists in local mode."""
        from quantdl.storage.pipeline import Validator
        uni_dir = tmp_path / "data" / "meta" / "universe" / "2024" / "06"
        uni_dir.mkdir(parents=True)
        (uni_dir / "top3000.txt").write_text("AAPL\nMSFT")

        with patch.dict(os.environ, {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            validator = Validator(s3_client=Mock(), bucket_name="test-bucket")
            assert validator.top_3000_exists(2024, 6) is True
            assert validator.top_3000_exists(2024, 7) is False

    def test_list_files_local(self, tmp_path):
        """Test list_files_under_prefix in local mode."""
        from quantdl.storage.pipeline import Validator
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "SEC001"
        ticks_dir.mkdir(parents=True)
        (ticks_dir / "ticks.parquet").write_text("dummy")

        with patch.dict(os.environ, {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            validator = Validator(s3_client=Mock(), bucket_name="test-bucket")
            files = validator.list_files_under_prefix("data/raw/ticks")
            assert len(files) >= 1
            assert any("ticks.parquet" in f for f in files)

    def test_list_files_local_empty(self, tmp_path):
        """Test list_files_under_prefix for nonexistent prefix."""
        from quantdl.storage.pipeline import Validator
        with patch.dict(os.environ, {'LOCAL_STORAGE_PATH': str(tmp_path)}):
            validator = Validator(s3_client=Mock(), bucket_name="test-bucket")
            files = validator.list_files_under_prefix("data/nonexistent")
            assert files == []

    def test_data_exists_error_non_404_s3(self):
        """Test data_exists returns False and logs on non-404 S3 error."""
        from quantdl.storage.pipeline import Validator
        mock_s3 = Mock()
        error_response = {'Error': {'Code': '403', 'Message': 'Forbidden'}}
        mock_s3.head_object.side_effect = ClientError(error_response, 'HeadObject')

        with patch.dict(os.environ, {'STORAGE_BACKEND': 's3'}):
            with patch('quantdl.storage.pipeline.validation.setup_logger', return_value=Mock()):
                validator = Validator(s3_client=mock_s3, bucket_name="test-bucket")
                result = validator.data_exists("AAPL", "ticks", year=2024, security_id=12345)
                assert result is False

    def test_list_files_s3(self):
        """Test list_files_under_prefix in S3 mode."""
        from quantdl.storage.pipeline import Validator
        mock_s3 = Mock()
        mock_s3.list_objects_v2.return_value = {
            'Contents': [{'Key': 'data/raw/ticks/daily/SEC001/ticks.parquet'}],
            'IsTruncated': False,
        }

        with patch.dict(os.environ, {'STORAGE_BACKEND': 's3'}):
            with patch('quantdl.storage.pipeline.validation.setup_logger', return_value=Mock()):
                validator = Validator(s3_client=mock_s3, bucket_name="test-bucket")
                files = validator.list_files_under_prefix("data/raw/ticks")
                assert len(files) == 1

    def test_list_files_s3_paginated(self):
        """Test list_files_under_prefix handles S3 pagination."""
        from quantdl.storage.pipeline import Validator
        mock_s3 = Mock()
        mock_s3.list_objects_v2.side_effect = [
            {
                'Contents': [{'Key': 'file1.parquet'}],
                'IsTruncated': True,
                'NextContinuationToken': 'token123',
            },
            {
                'Contents': [{'Key': 'file2.parquet'}],
                'IsTruncated': False,
            },
        ]

        with patch.dict(os.environ, {'STORAGE_BACKEND': 's3'}):
            with patch('quantdl.storage.pipeline.validation.setup_logger', return_value=Mock()):
                validator = Validator(s3_client=mock_s3, bucket_name="test-bucket")
                files = validator.list_files_under_prefix("prefix")
                assert len(files) == 2


@patch.dict(os.environ, {'STORAGE_BACKEND': 's3'})
class TestValidator:
    """Test Validator class"""

    def test_data_exists_ticks_with_security_id(self):
        """Test data_exists for ticks uses security_id-based single file path."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024, security_id=12345)

        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/daily/12345/ticks.parquet'
        )
        assert result is True

    def test_data_exists_ticks_without_security_id(self):
        """Test data_exists for ticks uses symbol as fallback identifier."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024)

        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/daily/AAPL/ticks.parquet'
        )
        assert result is True

    def test_data_exists_fundamental_with_security_id(self):
        """Test data_exists for fundamental uses security_id-based path."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "fundamental", security_id=12345)

        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/fundamental/12345/fundamental.parquet'
        )
        assert result is True

    def test_data_exists_fundamental_with_cik(self):
        """Test data_exists for fundamental uses CIK as fallback."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "fundamental", cik="0000320193")

        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/fundamental/0000320193/fundamental.parquet'
        )
        assert result is True

    def test_data_exists_not_found(self):
        """Test data_exists returns False when object not found (404)."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024, security_id=12345)

        assert result is False

    def test_data_exists_invalid_type_raises(self):
        """Test data_exists raises ValueError for invalid data_type."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")

        with pytest.raises(ValueError, match="Expected data_type"):
            validator.data_exists("AAPL", "invalid_type")

    def test_top_3000_exists_true(self):
        """Test top_3000_exists returns True when file exists."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.top_3000_exists(2024, 6)

        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/meta/universe/2024/06/top3000.txt'
        )
        assert result is True

    def test_top_3000_exists_error_logging(self):
        """Test top_3000_exists logs error on non-404 errors."""
        from quantdl.storage.pipeline import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()

        # Mock non-404 error
        error_response = {'Error': {'Code': '403', 'Message': 'Forbidden'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        # Patch setup_logger to return our mock logger
        with patch('quantdl.storage.pipeline.validation.setup_logger', return_value=mock_logger):
            validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
            result = validator.top_3000_exists(2024, 6)

        assert result is False
        mock_logger.error.assert_called()
        error_call = str(mock_logger.error.call_args)
        assert "data/meta/universe/2024/06/top3000.txt" in error_call
