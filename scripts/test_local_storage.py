#!/usr/bin/env python
"""
Test script to verify local filesystem storage backend works with all data types.

Usage:
    uv run python scripts/test_local_storage.py

This script:
1. Sets up local storage backend
2. Tests daily ticks, minute ticks, and fundamental data for a few symbols
3. Verifies data can be written and read back
"""

import os
import sys
import tempfile
import datetime as dt
from pathlib import Path

# Set environment for local storage BEFORE importing quantdl modules
TEST_STORAGE_PATH = tempfile.mkdtemp(prefix="quantdl_test_")
os.environ['STORAGE_BACKEND'] = 'local'
os.environ['LOCAL_STORAGE_PATH'] = TEST_STORAGE_PATH

print(f"Using local storage at: {TEST_STORAGE_PATH}")

import polars as pl
from quantdl.storage.s3_client import S3Client
from quantdl.storage.local_client import LocalStorageClient
from quantdl.storage.exceptions import NoSuchKeyError


def test_basic_operations():
    """Test basic put/get/list operations."""
    print("\n=== Testing basic operations ===")

    s3 = S3Client()
    client = s3.client

    # Verify we got LocalStorageClient
    assert isinstance(client, LocalStorageClient), f"Expected LocalStorageClient, got {type(client)}"
    assert s3.is_local is True
    print("✓ S3Client returns LocalStorageClient when STORAGE_BACKEND=local")

    # Test put/get
    test_data = b"Hello, local storage!"
    client.put_object(
        Bucket='us-equity-datalake',
        Key='test/hello.txt',
        Body=test_data,
        ContentType='text/plain'
    )

    response = client.get_object(Bucket='us-equity-datalake', Key='test/hello.txt')
    assert response['Body'].read() == test_data
    print("✓ put_object and get_object work")

    # Test head_object
    head = client.head_object(Bucket='us-equity-datalake', Key='test/hello.txt')
    assert head['ContentLength'] == len(test_data)
    print("✓ head_object works")

    # Test list_objects_v2
    listing = client.list_objects_v2(Bucket='us-equity-datalake', Prefix='test/')
    assert any(obj['Key'] == 'test/hello.txt' for obj in listing['Contents'])
    print("✓ list_objects_v2 works")

    # Test delete_object
    client.delete_object(Bucket='us-equity-datalake', Key='test/hello.txt')
    try:
        client.get_object(Bucket='us-equity-datalake', Key='test/hello.txt')
        assert False, "Should have raised NoSuchKeyError"
    except NoSuchKeyError:
        pass
    print("✓ delete_object works")

    print("\n✓ All basic operations passed!")


def test_parquet_roundtrip():
    """Test writing and reading parquet files (simulates ticks data)."""
    print("\n=== Testing Parquet roundtrip ===")

    client = S3Client().client

    # Create sample daily ticks data
    df = pl.DataFrame({
        'timestamp': ['2025-01-15', '2025-01-16', '2025-01-17'],
        'open': [150.0, 151.5, 152.0],
        'high': [152.0, 153.0, 154.5],
        'low': [149.5, 150.5, 151.0],
        'close': [151.5, 152.0, 153.5],
        'volume': [1000000, 1100000, 950000]
    })

    # Write parquet
    import io
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)

    security_id = 12345
    s3_key = f"data/raw/ticks/daily/{security_id}/2025/01/ticks.parquet"

    client.put_object(
        Bucket='us-equity-datalake',
        Key=s3_key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet'
    )
    print(f"✓ Wrote parquet to {s3_key}")

    # Read parquet back
    response = client.get_object(Bucket='us-equity-datalake', Key=s3_key)
    df_read = pl.read_parquet(response['Body'])

    assert df_read.shape == df.shape
    assert df_read['timestamp'].to_list() == df['timestamp'].to_list()
    print(f"✓ Read parquet back: {df_read.shape[0]} rows")

    # Verify file exists on disk
    expected_path = Path(TEST_STORAGE_PATH) / s3_key
    assert expected_path.exists(), f"File not found at {expected_path}"
    print(f"✓ File exists at: {expected_path}")

    print("\n✓ Parquet roundtrip passed!")


def test_fundamental_data():
    """Test writing fundamental data structure."""
    print("\n=== Testing fundamental data structure ===")

    client = S3Client().client

    # Create sample fundamental data (long format)
    df = pl.DataFrame({
        'symbol': ['AAPL'] * 4,
        'as_of_date': ['2024-10-30'] * 4,
        'accn': ['0000320193-24-000123'] * 4,
        'form': ['10-K'] * 4,
        'concept': ['rev', 'net_inc', 'assets', 'equity'],
        'value': [394328000000.0, 96995000000.0, 352583000000.0, 62146000000.0],
        'start': ['2023-10-01', '2023-10-01', None, None],
        'end': ['2024-09-28', '2024-09-28', '2024-09-28', '2024-09-28'],
        'frame': ['CY2024'] * 4,
        'is_instant': [False, False, True, True]
    })

    import io
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)

    cik = '0000320193'
    s3_key = f"data/raw/fundamental/{cik}/fundamental.parquet"

    client.put_object(
        Bucket='us-equity-datalake',
        Key=s3_key,
        Body=buffer.getvalue(),
        ContentType='application/x-parquet',
        Metadata={'cik': cik, 'symbol': 'AAPL'}
    )
    print(f"✓ Wrote fundamental data to {s3_key}")

    # Read back and verify
    response = client.get_object(Bucket='us-equity-datalake', Key=s3_key)
    df_read = pl.read_parquet(response['Body'])
    assert df_read.shape == df.shape
    print(f"✓ Read fundamental data: {df_read.shape[0]} rows, {df_read['concept'].n_unique()} concepts")

    # Verify metadata
    head = client.head_object(Bucket='us-equity-datalake', Key=s3_key)
    assert head['Metadata']['cik'] == cik
    print("✓ Metadata preserved")

    print("\n✓ Fundamental data test passed!")


def test_list_with_delimiter():
    """Test directory listing (used by validation)."""
    print("\n=== Testing list with delimiter ===")

    client = S3Client().client

    # Create hierarchical structure
    for month in ['01', '02', '03']:
        for day in ['01', '15']:
            key = f"data/raw/ticks/minute/99999/2025/{month}/{day}/ticks.parquet"
            client.put_object(
                Bucket='us-equity-datalake',
                Key=key,
                Body=b"dummy"
            )

    # List with delimiter (get "directories")
    response = client.list_objects_v2(
        Bucket='us-equity-datalake',
        Prefix='data/raw/ticks/minute/99999/2025/01/',
        Delimiter='/'
    )

    prefixes = [p['Prefix'] for p in response['CommonPrefixes']]
    assert 'data/raw/ticks/minute/99999/2025/01/01/' in prefixes
    assert 'data/raw/ticks/minute/99999/2025/01/15/' in prefixes
    print(f"✓ Found day folders: {prefixes}")

    print("\n✓ List with delimiter test passed!")


def print_storage_summary():
    """Print summary of what was written to local storage."""
    print("\n=== Storage Summary ===")

    storage_path = Path(TEST_STORAGE_PATH)

    def count_files(path, pattern="**/*"):
        return sum(1 for f in path.glob(pattern) if f.is_file() and not f.name.startswith('.'))

    print(f"Storage root: {storage_path}")
    print(f"Total files: {count_files(storage_path)}")

    # List top-level structure
    for item in sorted(storage_path.rglob('*')):
        if item.is_file() and not item.name.startswith('.'):
            rel_path = item.relative_to(storage_path)
            size = item.stat().st_size
            print(f"  {rel_path} ({size} bytes)")


def cleanup():
    """Clean up test storage."""
    import shutil
    print(f"\nCleaning up: {TEST_STORAGE_PATH}")
    shutil.rmtree(TEST_STORAGE_PATH, ignore_errors=True)


def main():
    print("=" * 60)
    print("Local Storage Backend Test")
    print("=" * 60)

    try:
        test_basic_operations()
        test_parquet_roundtrip()
        test_fundamental_data()
        test_list_with_delimiter()
        print_storage_summary()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == '__main__':
    main()
