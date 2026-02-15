"""
Unit tests for storage.validation.Validator class
Tests data validation and existence checking
"""
import os
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError


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
