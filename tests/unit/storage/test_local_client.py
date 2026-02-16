"""
Unit tests for storage.local_client.LocalStorageClient class
Tests local filesystem storage backend with boto3-compatible API
"""
import io
import json
import pytest

from alphalab.storage.clients import LocalStorageClient, StreamingBody
from alphalab.storage.utils import NoSuchKeyError


class TestStreamingBody:
    """Test StreamingBody class"""

    def test_read_all(self):
        """Test read() returns all data"""
        body = StreamingBody(b"hello world")
        assert body.read() == b"hello world"

    def test_read_partial(self):
        """Test read(amt) returns partial data"""
        body = StreamingBody(b"hello world")
        assert body.read(5) == b"hello"
        assert body.read(6) == b" world"

    def test_context_manager(self):
        """Test StreamingBody works as context manager"""
        body = StreamingBody(b"data")
        with body as b:
            assert b.read() == b"data"

    def test_seek_and_tell(self):
        """Test seek() and tell() for polars compatibility"""
        body = StreamingBody(b"hello world")
        assert body.tell() == 0
        body.read(5)
        assert body.tell() == 5
        body.seek(0)
        assert body.tell() == 0
        assert body.read() == b"hello world"

    def test_seek_from_end(self):
        """Test seek with whence=2 (from end)"""
        body = StreamingBody(b"hello world")
        body.seek(-5, 2)  # 5 bytes from end
        assert body.read() == b"world"


class TestLocalStorageClient:
    """Test LocalStorageClient class"""

    def test_init_creates_directory(self, tmp_path):
        """Test __init__ creates base directory if not exists"""
        new_dir = tmp_path / "new_storage"
        client = LocalStorageClient(str(new_dir))
        assert new_dir.exists()

    def test_put_and_get_object_bytes(self, tmp_path):
        """Test put_object and get_object with bytes"""
        client = LocalStorageClient(str(tmp_path))

        client.put_object(
            Bucket='test-bucket',
            Key='data/test.txt',
            Body=b'hello world'
        )

        response = client.get_object(Bucket='test-bucket', Key='data/test.txt')

        assert response['Body'].read() == b'hello world'
        assert response['ContentLength'] == 11
        assert 'LastModified' in response
        assert 'ETag' in response

    def test_put_and_get_object_bytesio(self, tmp_path):
        """Test put_object and get_object with BytesIO"""
        client = LocalStorageClient(str(tmp_path))

        buffer = io.BytesIO(b'test content')
        client.put_object(
            Bucket='test-bucket',
            Key='data/file.bin',
            Body=buffer
        )

        response = client.get_object(Bucket='test-bucket', Key='data/file.bin')
        assert response['Body'].read() == b'test content'

    def test_put_object_with_metadata(self, tmp_path):
        """Test put_object stores metadata"""
        client = LocalStorageClient(str(tmp_path))

        client.put_object(
            Bucket='test-bucket',
            Key='data/meta.txt',
            Body=b'data',
            ContentType='text/plain',
            Metadata={'custom_key': 'custom_value'}
        )

        response = client.get_object(Bucket='test-bucket', Key='data/meta.txt')
        assert response['ContentType'] == 'text/plain'
        assert response['Metadata'] == {'custom_key': 'custom_value'}

    def test_get_object_not_found(self, tmp_path):
        """Test get_object raises NoSuchKeyError for missing key"""
        client = LocalStorageClient(str(tmp_path))

        with pytest.raises(NoSuchKeyError) as exc_info:
            client.get_object(Bucket='test-bucket', Key='nonexistent.txt')

        assert exc_info.value.response['Error']['Code'] == 'NoSuchKey'
        assert 'nonexistent.txt' in exc_info.value.response['Error']['Message']

    def test_head_object(self, tmp_path):
        """Test head_object returns metadata without body"""
        client = LocalStorageClient(str(tmp_path))

        client.put_object(
            Bucket='bucket',
            Key='file.txt',
            Body=b'content',
            ContentType='text/plain',
            Metadata={'key': 'value'}
        )

        response = client.head_object(Bucket='bucket', Key='file.txt')

        assert response['ContentLength'] == 7
        assert response['ContentType'] == 'text/plain'
        assert response['Metadata'] == {'key': 'value'}
        assert 'LastModified' in response
        assert 'ETag' in response
        assert 'Body' not in response

    def test_head_object_not_found(self, tmp_path):
        """Test head_object raises NoSuchKeyError for missing key"""
        client = LocalStorageClient(str(tmp_path))

        with pytest.raises(NoSuchKeyError):
            client.head_object(Bucket='bucket', Key='missing.txt')

    def test_delete_object(self, tmp_path):
        """Test delete_object removes file and metadata"""
        client = LocalStorageClient(str(tmp_path))

        # Create file
        client.put_object(
            Bucket='bucket',
            Key='to_delete.txt',
            Body=b'delete me',
            Metadata={'key': 'value'}
        )

        # Verify exists
        client.head_object(Bucket='bucket', Key='to_delete.txt')

        # Delete
        result = client.delete_object(Bucket='bucket', Key='to_delete.txt')
        assert result == {}

        # Verify deleted
        with pytest.raises(NoSuchKeyError):
            client.get_object(Bucket='bucket', Key='to_delete.txt')

        # Verify metadata file also deleted
        metadata_path = tmp_path / '.to_delete.txt.metadata.json'
        assert not metadata_path.exists()

    def test_delete_object_not_found_silent(self, tmp_path):
        """Test delete_object silently succeeds for missing key"""
        client = LocalStorageClient(str(tmp_path))

        # Should not raise
        result = client.delete_object(Bucket='bucket', Key='nonexistent.txt')
        assert result == {}

    def test_list_objects_v2_basic(self, tmp_path):
        """Test list_objects_v2 returns all objects under prefix"""
        client = LocalStorageClient(str(tmp_path))

        # Create test files
        client.put_object(Bucket='b', Key='data/a.txt', Body=b'a')
        client.put_object(Bucket='b', Key='data/b.txt', Body=b'b')
        client.put_object(Bucket='b', Key='other/c.txt', Body=b'c')

        response = client.list_objects_v2(Bucket='b', Prefix='data/')

        assert response['KeyCount'] == 2
        keys = [obj['Key'] for obj in response['Contents']]
        assert 'data/a.txt' in keys
        assert 'data/b.txt' in keys
        assert 'other/c.txt' not in keys

    def test_list_objects_v2_with_delimiter(self, tmp_path):
        """Test list_objects_v2 with delimiter returns CommonPrefixes"""
        client = LocalStorageClient(str(tmp_path))

        # Create hierarchical structure
        client.put_object(Bucket='b', Key='data/2024/01/file1.txt', Body=b'1')
        client.put_object(Bucket='b', Key='data/2024/02/file2.txt', Body=b'2')
        client.put_object(Bucket='b', Key='data/root.txt', Body=b'root')

        response = client.list_objects_v2(
            Bucket='b',
            Prefix='data/',
            Delimiter='/'
        )

        # Should have one direct file in Contents
        assert any(obj['Key'] == 'data/root.txt' for obj in response['Contents'])

        # Should have CommonPrefixes for subdirectories
        prefixes = [p['Prefix'] for p in response['CommonPrefixes']]
        assert 'data/2024/' in prefixes

    def test_list_objects_v2_empty_prefix(self, tmp_path):
        """Test list_objects_v2 with empty prefix returns all files"""
        client = LocalStorageClient(str(tmp_path))

        client.put_object(Bucket='b', Key='file1.txt', Body=b'1')
        client.put_object(Bucket='b', Key='dir/file2.txt', Body=b'2')

        response = client.list_objects_v2(Bucket='b', Prefix='')

        assert response['KeyCount'] == 2

    def test_list_objects_v2_nonexistent_prefix(self, tmp_path):
        """Test list_objects_v2 with nonexistent prefix returns empty"""
        client = LocalStorageClient(str(tmp_path))

        response = client.list_objects_v2(Bucket='b', Prefix='nonexistent/')

        assert response['Contents'] == []
        assert response['KeyCount'] == 0
        assert not response['IsTruncated']

    def test_list_objects_v2_excludes_metadata_files(self, tmp_path):
        """Test list_objects_v2 excludes .metadata.json files"""
        client = LocalStorageClient(str(tmp_path))

        client.put_object(
            Bucket='b',
            Key='data/file.txt',
            Body=b'content',
            Metadata={'key': 'value'}
        )

        response = client.list_objects_v2(Bucket='b', Prefix='data/')

        keys = [obj['Key'] for obj in response['Contents']]
        assert 'data/file.txt' in keys
        assert not any('.metadata.json' in k for k in keys)

    def test_upload_fileobj(self, tmp_path):
        """Test upload_fileobj with BytesIO"""
        client = LocalStorageClient(str(tmp_path))

        buffer = io.BytesIO(b'uploaded content')
        client.upload_fileobj(
            Fileobj=buffer,
            Bucket='bucket',
            Key='uploaded.txt',
            ExtraArgs={
                'ContentType': 'text/plain',
                'Metadata': {'source': 'test'}
            }
        )

        response = client.get_object(Bucket='bucket', Key='uploaded.txt')
        assert response['Body'].read() == b'uploaded content'
        assert response['ContentType'] == 'text/plain'
        assert response['Metadata'] == {'source': 'test'}

    def test_upload_fileobj_no_extra_args(self, tmp_path):
        """Test upload_fileobj without ExtraArgs"""
        client = LocalStorageClient(str(tmp_path))

        buffer = io.BytesIO(b'data')
        client.upload_fileobj(
            Fileobj=buffer,
            Bucket='bucket',
            Key='simple.bin'
        )

        response = client.get_object(Bucket='bucket', Key='simple.bin')
        assert response['Body'].read() == b'data'

    def test_nested_directory_creation(self, tmp_path):
        """Test put_object creates nested directories"""
        client = LocalStorageClient(str(tmp_path))

        client.put_object(
            Bucket='b',
            Key='a/b/c/d/e/file.txt',
            Body=b'deep'
        )

        assert (tmp_path / 'a' / 'b' / 'c' / 'd' / 'e' / 'file.txt').exists()

    def test_overwrite_existing_file(self, tmp_path):
        """Test put_object overwrites existing file"""
        client = LocalStorageClient(str(tmp_path))

        client.put_object(Bucket='b', Key='file.txt', Body=b'original')
        client.put_object(Bucket='b', Key='file.txt', Body=b'updated')

        response = client.get_object(Bucket='b', Key='file.txt')
        assert response['Body'].read() == b'updated'

    def test_binary_data(self, tmp_path):
        """Test put/get with binary data"""
        client = LocalStorageClient(str(tmp_path))

        # Binary data with null bytes
        binary_data = bytes(range(256))
        client.put_object(Bucket='b', Key='binary.bin', Body=binary_data)

        response = client.get_object(Bucket='b', Key='binary.bin')
        assert response['Body'].read() == binary_data

    def test_large_file(self, tmp_path):
        """Test put/get with larger file"""
        client = LocalStorageClient(str(tmp_path))

        # 1MB of data
        large_data = b'x' * (1024 * 1024)
        client.put_object(Bucket='b', Key='large.bin', Body=large_data)

        response = client.get_object(Bucket='b', Key='large.bin')
        assert response['Body'].read() == large_data
        assert response['ContentLength'] == 1024 * 1024


class TestNoSuchKeyErrorCompatibility:
    """Test NoSuchKeyError boto3 compatibility"""

    def test_error_response_structure(self):
        """Test error has boto3-compatible response structure"""
        error = NoSuchKeyError('my-bucket', 'my-key')

        assert error.response['Error']['Code'] == 'NoSuchKey'
        assert 'my-key' in error.response['Error']['Message']
        assert error.response['Error']['Key'] == 'my-key'
        assert error.response['Error']['BucketName'] == 'my-bucket'

    def test_can_check_error_code(self):
        """Test error code can be checked like boto3 ClientError"""
        error = NoSuchKeyError('bucket', 'key')

        # This is how code typically checks boto3 errors
        assert error.response.get('Error', {}).get('Code') == 'NoSuchKey'
