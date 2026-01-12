"""
Unit tests for storage.validation.Validator class
Tests data validation and existence checking
"""
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError


class TestValidator:
    """Test Validator class"""

    def test_data_exists_monthly_partition(self):
        """Test data_exists with monthly partition - covers line 145."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024, month=6)

        # Should construct monthly partition path
        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/daily/AAPL/2024/06/ticks.parquet'
        )
        assert result is True

    def test_top_3000_exists_error_logging(self):
        """Test top_3000_exists logs error on non-404 errors - covers lines 191-192."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()

        # Mock non-404 error
        error_response = {'Error': {'Code': '403', 'Message': 'Forbidden'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        # Patch setup_logger to return our mock logger
        with patch('quantdl.storage.validation.setup_logger', return_value=mock_logger):
            validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
            result = validator.top_3000_exists(2024, 6)

        assert result is False
        mock_logger.error.assert_called()
        # Check that the error message contains the key
        error_call = str(mock_logger.error.call_args)
        assert "data/symbols/2024/06/top3000.txt" in error_call
