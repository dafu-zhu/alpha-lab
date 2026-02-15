"""
Unit tests for storage.data_publishers module
Tests DataPublishers functionality with dependency injection
"""
import os
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import polars as pl
import requests


def _make_publisher():
    from quantdl.storage.pipeline import DataPublishers

    s3_client = Mock()
    upload_config = SimpleNamespace(transfer={})
    logger = Mock()
    data_collectors = Mock()
    security_master = Mock()
    security_master.get_security_id = Mock(return_value=12345)

    publisher = DataPublishers(
        s3_client=s3_client,
        upload_config=upload_config,
        logger=logger,
        data_collectors=data_collectors,
        security_master=security_master,
        bucket_name="test-bucket"
    )

    return publisher, s3_client, data_collectors


@patch.dict(os.environ, {'STORAGE_BACKEND': 's3'})
class TestDataPublishers:
    def test_initialization(self):
        publisher, s3_client, data_collectors = _make_publisher()

        assert publisher.s3_client == s3_client
        assert publisher.data_collectors == data_collectors
        assert publisher.bucket_name == "test-bucket"

    def test_publish_daily_ticks_success(self):
        """Test publishing daily ticks to single file per security_id."""
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()

        # Mock get_object to raise NoSuchKey (no existing file)
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df)

        assert result == {"symbol": "AAPL", "status": "success", "error": None}
        s3_client.upload_fileobj.assert_called_once()

        # Verify security_id-based path
        call_args = s3_client.upload_fileobj.call_args
        assert call_args.kwargs["Key"] == "data/raw/ticks/daily/12345/ticks.parquet"

    def test_publish_daily_ticks_empty(self):
        """Test that empty dataframes are skipped."""
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df)
        assert result["status"] == "skipped"
        s3_client.upload_fileobj.assert_not_called()

    def test_publish_daily_ticks_value_error_skips(self):
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=ValueError("not active on"))

        # Mock get_object to raise NoSuchKey (no existing file)
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df)

        assert result["status"] == "skipped"

    def test_publish_daily_ticks_value_error_other(self):
        """Test ValueError that doesn't contain 'not active on' returns failed status."""
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=ValueError("some other error"))

        # Mock get_object to raise NoSuchKey
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df)

        assert result["status"] == "failed"
        assert result["error"] == "some other error"
        publisher.logger.error.assert_called()

    def test_publish_daily_ticks_request_exception(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=requests.RequestException("boom"))

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df)

        assert result["status"] == "failed"

    def test_publish_daily_ticks_unexpected_error(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=RuntimeError("boom"))

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df)

        assert result["status"] == "failed"

    def test_publish_daily_ticks_reraises_non_nosuchkey_error(self):
        """Test that non-NoSuchKey ClientError results in failed status."""
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()

        # Mock get_object to raise a different error (e.g., AccessDenied)
        error_response = {'Error': {'Code': 'AccessDenied'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df)

        assert result["status"] == "failed"

    def test_publish_fundamental_skips_without_cik(self):
        publisher, _, data_collectors = _make_publisher()

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik=None,
            security_id=12345,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"
        data_collectors.collect_fundamental_long.assert_not_called()

    def test_publish_fundamental_empty(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.return_value = pl.DataFrame()

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            security_id=12345,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"

    def test_publish_fundamental_success(self):
        publisher, _, data_collectors = _make_publisher()
        publisher.upload_fileobj = Mock()
        data_collectors._load_concepts.return_value = ["sales", "income"]
        data_collectors.collect_fundamental_long.return_value = pl.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "as_of_date": ["2024-06-30", "2024-06-30"],
            "accn": ["1", "1"],
            "form": ["10-Q", "10-Q"],
            "concept": ["sales", "income"],
            "value": [100.0, 10.0],
            "start": ["2024-04-01", "2024-04-01"],
            "end": ["2024-06-30", "2024-06-30"],
            "frame": ["CY2024Q2", "CY2024Q2"],
            "is_instant": [False, False]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            security_id=12345,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_fundamental_uses_security_id_path(self):
        """Test that publish_fundamental uses security_id-based path."""
        publisher, s3_client, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["sales"]
        data_collectors.collect_fundamental_long.return_value = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "accn": ["1"],
            "form": ["10-Q"],
            "concept": ["sales"],
            "value": [100.0],
            "start": ["2024-04-01"],
            "end": ["2024-06-30"],
            "frame": ["CY2024Q2"],
            "is_instant": [False]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            security_id=12345,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"

        # Verify security_id-based path is used
        call_args = s3_client.upload_fileobj.call_args
        s3_key = call_args.kwargs["Key"]
        assert s3_key == "data/raw/fundamental/12345/fundamental.parquet"

        # Verify metadata includes both symbol and CIK
        metadata = call_args.kwargs["ExtraArgs"]["Metadata"]
        assert metadata["symbol"] == "AAPL"
        assert metadata["cik"] == "0000320193"

    def test_publish_fundamental_request_exception(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.side_effect = requests.RequestException("boom")

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            security_id=12345,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_fundamental_value_error(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.side_effect = ValueError("bad data")

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            security_id=12345,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_fundamental_unexpected_exception(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.side_effect = RuntimeError("boom")

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            security_id=12345,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_top_3000_success(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock()

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=["AAPL", "MSFT"],
            source="test"
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_top_3000_upload_error(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=RuntimeError("boom"))

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=["AAPL", "MSFT"],
            source="test"
        )

        assert result["status"] == "failed"

    def test_publish_top_3000_empty(self):
        publisher, _, _ = _make_publisher()

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=[],
            source="test"
        )

        assert result["status"] == "skipped"

    def test_get_fundamental_metadata_exists(self):
        """Test getting metadata for existing fundamental data."""
        publisher, s3_client, _ = _make_publisher()

        # Mock S3 head_object response with metadata
        s3_client.head_object.return_value = {
            'Metadata': {
                'symbol': 'AAPL',
                'cik': '0000320193',
                'latest_filing_date': '2024-01-31',
                'latest_accn': '0000320193-24-000010'
            }
        }

        metadata = publisher.get_fundamental_metadata(12345)

        assert metadata is not None
        assert metadata['symbol'] == 'AAPL'
        assert metadata['latest_filing_date'] == '2024-01-31'
        assert metadata['latest_accn'] == '0000320193-24-000010'
        s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/raw/fundamental/12345/fundamental.parquet'
        )

    def test_get_fundamental_metadata_not_exists(self):
        """Test getting metadata when file doesn't exist."""
        publisher, s3_client, _ = _make_publisher()

        # Mock S3 head_object to raise exception (file not found)
        s3_client.head_object.side_effect = Exception("404")

        metadata = publisher.get_fundamental_metadata(12345)

        assert metadata is None

    def test_publish_fundamental_includes_metadata_tracking(self):
        """Test that publish_fundamental includes latest_filing_date and latest_accn in metadata."""
        publisher, s3_client, data_collectors = _make_publisher()

        # Mock data
        mock_df = pl.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'as_of_date': ['2024-01-31', '2024-10-31'],
            'accn': ['0000320193-24-000010', '0000320193-24-000078'],
            'form': ['10-Q', '10-K'],
            'concept': ['Assets', 'Assets'],
            'value': [1000000, 1100000],
            'start': [None, None],
            'end': ['2024-01-31', '2024-10-31'],
            'frame': ['CY2024Q1', 'CY2024Q4'],
            'is_instant': [True, True]
        })

        data_collectors.collect_fundamental_long.return_value = mock_df
        data_collectors._load_concepts.return_value = ['Assets']

        # Mock rate limiter
        rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym='AAPL',
            start_date='2024-01-01',
            end_date='2024-12-31',
            cik='0000320193',
            security_id=12345,
            sec_rate_limiter=rate_limiter
        )

        # Verify upload was called
        assert s3_client.upload_fileobj.called

        # Get the metadata argument from the upload call
        call_args = s3_client.upload_fileobj.call_args
        metadata = call_args[1]['ExtraArgs']['Metadata']

        # Verify latest_filing_date and latest_accn are in metadata
        assert 'latest_filing_date' in metadata
        assert 'latest_accn' in metadata
        assert metadata['latest_filing_date'] == '2024-10-31'  # Latest date
        assert metadata['latest_accn'] == '0000320193-24-000078'  # Accn for latest date
        assert result['status'] == 'success'
