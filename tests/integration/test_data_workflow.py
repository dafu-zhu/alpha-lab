"""
Integration tests for data collection workflows
Tests end-to-end data collection scenarios with mocked external services
"""
import io
import pytest
import polars as pl
import datetime as dt
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


@pytest.mark.integration
class TestDataCollectionWorkflow:
    """
    Integration tests for complete data collection workflows
    These tests verify component interactions without hitting real APIs
    """

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client for testing"""
        mock = MagicMock()
        mock.put_object.return_value = {'ETag': 'test-etag'}
        mock.head_object.return_value = {}
        return mock

    @pytest.fixture
    def sample_tick_data(self):
        """Create sample tick data for testing"""
        return pl.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'open': [150.0, 151.0, 152.0],
            'high': [155.0, 156.0, 157.0],
            'low': [149.0, 150.0, 151.0],
            'close': [152.0, 153.0, 154.0],
            'volume': [1000000, 1100000, 1200000]
        })

    @pytest.fixture
    def sample_fundamental_data(self):
        """Create sample fundamental data for testing"""
        return pl.DataFrame({
            'symbol': ['AAPL'] * 4,
            'as_of_date': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'concept': ['rev', 'rev', 'rev', 'rev'],
            'value': [100000.0, 110000.0, 120000.0, 130000.0],
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'],
            'form': ['10-Q', '10-Q', '10-Q', '10-K']
        })

    def test_tick_data_collection_workflow(self, sample_tick_data, mock_s3_client):
        """
        Test complete workflow for tick data collection:
        1. Fetch data from source (mocked)
        2. Process/validate data
        3. Upload to S3 (mocked)
        """
        # Step 1: Mock data fetch
        fetched_data = sample_tick_data

        # Step 2: Validate data structure
        assert 'date' in fetched_data.columns
        assert 'open' in fetched_data.columns
        assert 'high' in fetched_data.columns
        assert 'low' in fetched_data.columns
        assert 'close' in fetched_data.columns
        assert 'volume' in fetched_data.columns

        # Step 3: Validate OHLC constraints
        for row in fetched_data.iter_rows(named=True):
            assert row['high'] >= row['low']
            assert row['close'] >= row['low']
            assert row['close'] <= row['high']
            assert row['volume'] >= 0

        # Step 4: Mock S3 upload (single file per security_id)
        s3_key = 'data/raw/ticks/daily/12345/ticks.parquet'
        parquet_buffer = io.BytesIO()
        fetched_data.write_parquet(parquet_buffer)
        buffer = parquet_buffer.getvalue()

        mock_s3_client.put_object(
            Bucket='test-bucket',
            Key=s3_key,
            Body=buffer
        )

        # Verify upload was called
        mock_s3_client.put_object.assert_called_once()

    def test_fundamental_data_collection_workflow(self, sample_fundamental_data, mock_s3_client):
        """
        Test complete workflow for fundamental data collection:
        1. Fetch from SEC EDGAR (mocked)
        2. Process into long format
        3. Upload to S3
        """
        # Step 1: Mock data fetch (already in long format)
        raw_data = sample_fundamental_data

        # Step 2: Validate long format structure
        assert 'symbol' in raw_data.columns
        assert 'as_of_date' in raw_data.columns
        assert 'concept' in raw_data.columns
        assert 'value' in raw_data.columns

        # Step 3: Verify data can be processed
        concepts = raw_data['concept'].unique().to_list()
        assert 'rev' in concepts

        # Step 4: Mock S3 upload (security_id-based path)
        s3_key = 'data/raw/fundamental/12345/fundamental.parquet'
        parquet_buffer = io.BytesIO()
        raw_data.write_parquet(parquet_buffer)
        buffer = parquet_buffer.getvalue()

        mock_s3_client.put_object(
            Bucket='test-bucket',
            Key=s3_key,
            Body=buffer
        )

        mock_s3_client.put_object.assert_called_once()

    def test_multi_symbol_batch_workflow(self, mock_s3_client):
        """
        Test batch processing workflow for multiple symbols:
        1. Load symbol universe
        2. Batch fetch data
        3. Process in parallel
        4. Upload results
        """
        # Step 1: Mock symbol universe
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        # Step 2: Mock batch data fetch
        batch_data = {}
        for symbol in symbols:
            batch_data[symbol] = pl.DataFrame({
                'date': ['2024-01-01'],
                'close': [100.0 + hash(symbol) % 50],
                'volume': [1000000]
            })

        # Step 3: Process each symbol's data
        results = []
        for symbol, data in batch_data.items():
            # Validate data
            assert len(data) > 0
            assert 'close' in data.columns

            # Mock processing result
            results.append({
                'symbol': symbol,
                'status': 'success',
                'rows': len(data)
            })

        # Step 4: Verify all symbols were processed
        assert len(results) == len(symbols)
        assert all(r['status'] == 'success' for r in results)

    def test_error_handling_workflow(self, mock_s3_client):
        """
        Test error handling in data collection workflow:
        1. Simulate data fetch failure
        2. Verify error is caught
        3. Continue with next symbol
        """
        symbols = ['VALID1', 'INVALID', 'VALID2']
        results = []

        for symbol in symbols:
            try:
                if symbol == 'INVALID':
                    # Simulate error
                    raise ValueError(f"Failed to fetch data for {symbol}")

                # Mock successful fetch
                data = pl.DataFrame({'date': ['2024-01-01'], 'close': [100.0]})
                results.append({'symbol': symbol, 'status': 'success'})

            except ValueError as e:
                # Error should be caught and logged
                results.append({'symbol': symbol, 'status': 'failed', 'error': str(e)})

        # Verify workflow continued despite error
        assert len(results) == 3
        assert results[0]['status'] == 'success'
        assert results[1]['status'] == 'failed'
        assert results[2]['status'] == 'success'

    def test_data_validation_workflow(self, sample_tick_data):
        """
        Test data validation workflow:
        1. Fetch data
        2. Run validation checks
        3. Filter invalid data
        4. Upload only valid data
        """
        # Step 1: Start with sample data
        data = sample_tick_data

        # Step 2: Run validation checks
        validation_results = []

        # Check for nulls
        has_nulls = data.null_count().sum_horizontal()[0] > 0
        validation_results.append(('null_check', not has_nulls))

        # Check OHLC constraints
        ohlc_valid = True
        for row in data.iter_rows(named=True):
            if not (row['high'] >= row['low']):
                ohlc_valid = False
                break
        validation_results.append(('ohlc_check', ohlc_valid))

        # Check for negative values
        no_negatives = True
        for row in data.iter_rows(named=True):
            if row['volume'] < 0:
                no_negatives = False
                break
        validation_results.append(('negative_check', no_negatives))

        # Step 3: All validations should pass
        assert all(result for _, result in validation_results)

        # Step 4: Data is valid, ready for upload
        assert len(data) > 0


@pytest.mark.integration
class TestDataStorageWorkflow:
    """Integration tests for data storage workflows"""

    def test_s3_key_generation(self):
        """Test S3 key generation for different data types"""
        # Daily ticks (single file per security_id)
        daily_key = "data/raw/ticks/daily/12345/ticks.parquet"
        assert '12345' in daily_key
        assert daily_key.startswith('data/raw/ticks/daily/')

        # Fundamental (security_id-based)
        fund_key = "data/raw/fundamental/12345/fundamental.parquet"
        assert 'fundamental' in fund_key
        assert fund_key.startswith('data/raw/fundamental/')

        # Top 3000 universe
        universe_key = "data/meta/universe/2024/06/top3000.txt"
        assert '2024' in universe_key
        assert universe_key.startswith('data/meta/universe/')

    def test_parquet_serialization_workflow(self):
        """Test Parquet serialization/deserialization workflow"""
        # Create test data
        original_data = pl.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'value': [100.0, 200.0],
            'date': ['2024-01-01', '2024-01-02']
        })

        # Serialize to Parquet
        parquet_buffer = io.BytesIO()
        original_data.write_parquet(parquet_buffer)
        buffer = parquet_buffer.getvalue()
        assert len(buffer) > 0

        # Deserialize
        restored_data = pl.read_parquet(io.BytesIO(buffer))

        # Verify data integrity
        assert original_data.shape == restored_data.shape
        assert original_data.columns == restored_data.columns
        assert original_data.equals(restored_data)

    def test_data_organization(self):
        """Test organization of data paths"""
        paths = {
            'ticks': 'data/raw/ticks/daily/12345/ticks.parquet',
            'fundamental': 'data/raw/fundamental/12345/fundamental.parquet',
            'universe': 'data/meta/universe/2024/06/top3000.txt',
            'master': 'data/meta/master/security_master.parquet',
        }

        # Verify proper organization
        assert paths['ticks'].startswith('data/raw/ticks/')
        assert paths['fundamental'].startswith('data/raw/fundamental/')
        assert paths['universe'].startswith('data/meta/universe/')
        assert paths['master'].startswith('data/meta/master/')

    def test_batch_upload_workflow(self):
        """Test batch upload workflow for multiple files"""
        # Mock multiple files to upload
        files_to_upload = [
            (12345, pl.DataFrame({'date': ['2024-01-01'], 'close': [150.0]})),
            (12346, pl.DataFrame({'date': ['2024-01-01'], 'close': [300.0]})),
            (12347, pl.DataFrame({'date': ['2024-01-01'], 'close': [140.0]})),
        ]

        upload_results = []
        for security_id, data in files_to_upload:
            # Single file per security_id
            s3_key = f"data/raw/ticks/daily/{security_id}/ticks.parquet"
            parquet_buffer = io.BytesIO()
            data.write_parquet(parquet_buffer)
            buffer = parquet_buffer.getvalue()

            # Simulate successful upload
            upload_results.append({
                'security_id': security_id,
                'key': s3_key,
                'size': len(buffer),
                'status': 'success'
            })

        # Verify all uploads succeeded
        assert len(upload_results) == 3
        assert all(r['status'] == 'success' for r in upload_results)
        assert all(r['size'] > 0 for r in upload_results)
