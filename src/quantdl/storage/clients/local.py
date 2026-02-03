"""
Local Filesystem Storage Client

Duck-typed replacement for boto3 S3 client that stores data on local filesystem.
Implements the same interface as boto3.client('s3') for seamless switching.
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from quantdl.storage.utils.exceptions import NoSuchKeyError


class StreamingBody:
    """
    Mimics boto3's StreamingBody for local file reads.

    Provides read() and seek() methods, compatible with how
    S3 responses are consumed (including by polars.read_parquet).
    """

    def __init__(self, data: bytes):
        self._data = io.BytesIO(data)

    def read(self, amt: Optional[int] = None) -> bytes:
        """Read bytes from the stream."""
        return self._data.read(amt)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position in stream."""
        return self._data.seek(offset, whence)

    def tell(self) -> int:
        """Return current position in stream."""
        return self._data.tell()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._data.close()


class LocalStorageClient:
    """
    Local filesystem storage client with boto3 S3 API compatibility.

    Maps S3 operations to local filesystem:
    - Bucket name is ignored (all data stored under base_path)
    - Key becomes relative path under base_path
    - Metadata stored in sidecar .metadata.json files

    Usage:
        client = LocalStorageClient('/path/to/storage')
        client.put_object(Bucket='ignored', Key='data/file.txt', Body=b'content')
        response = client.get_object(Bucket='ignored', Key='data/file.txt')
        content = response['Body'].read()
    """

    def __init__(self, base_path: str):
        """
        Initialize local storage client.

        :param base_path: Root directory for all storage operations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert S3 key to local filesystem path."""
        return self.base_path / key

    def _metadata_path(self, file_path: Path) -> Path:
        """Get path to metadata sidecar file."""
        return file_path.parent / f".{file_path.name}.metadata.json"

    def _save_metadata(self, file_path: Path, metadata: Dict[str, str]):
        """Save metadata to sidecar JSON file."""
        meta_path = self._metadata_path(file_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, file_path: Path) -> Dict[str, str]:
        """Load metadata from sidecar JSON file."""
        meta_path = self._metadata_path(file_path)
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, OSError):
                # Return empty metadata if file is corrupted
                return {}
        return {}

    def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        Retrieve an object from local storage.

        :param Bucket: Bucket name (ignored for local storage)
        :param Key: Object key (becomes relative path)
        :return: Dict with 'Body' (StreamingBody), 'ContentLength', 'ContentType', 'Metadata'
        :raises NoSuchKeyError: If key does not exist
        """
        file_path = self._key_to_path(Key)

        if not file_path.exists():
            raise NoSuchKeyError(Bucket, Key)

        with open(file_path, 'rb') as f:
            data = f.read()

        metadata = self._load_metadata(file_path)
        stat = file_path.stat()

        return {
            'Body': StreamingBody(data),
            'ContentLength': len(data),
            'ContentType': metadata.get('ContentType', 'application/octet-stream'),
            'LastModified': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            'Metadata': metadata.get('UserMetadata', {}),
            'ETag': f'"{stat.st_mtime}-{stat.st_size}"'
        }

    def put_object(
        self,
        Bucket: str,
        Key: str,
        Body: Union[bytes, io.BytesIO],
        ContentType: str = 'application/octet-stream',
        Metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Store an object in local storage.

        :param Bucket: Bucket name (ignored for local storage)
        :param Key: Object key (becomes relative path)
        :param Body: Object data (bytes or BytesIO)
        :param ContentType: MIME type
        :param Metadata: User metadata dict
        :return: Dict with 'ETag'
        """
        file_path = self._key_to_path(Key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle both bytes and BytesIO
        if isinstance(Body, io.BytesIO):
            data = Body.getvalue()
        elif hasattr(Body, 'read'):
            data = Body.read()
        else:
            data = Body

        with open(file_path, 'wb') as f:
            f.write(data)

        # Save metadata
        meta = {
            'ContentType': ContentType,
            'UserMetadata': Metadata or {}
        }
        self._save_metadata(file_path, meta)

        stat = file_path.stat()
        return {
            'ETag': f'"{stat.st_mtime}-{stat.st_size}"'
        }

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        Get object metadata without retrieving the object.

        :param Bucket: Bucket name (ignored)
        :param Key: Object key
        :return: Dict with 'ContentLength', 'ContentType', 'LastModified', 'Metadata'
        :raises NoSuchKeyError: If key does not exist
        """
        file_path = self._key_to_path(Key)

        if not file_path.exists():
            raise NoSuchKeyError(Bucket, Key)

        stat = file_path.stat()
        metadata = self._load_metadata(file_path)

        return {
            'ContentLength': stat.st_size,
            'ContentType': metadata.get('ContentType', 'application/octet-stream'),
            'LastModified': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            'Metadata': metadata.get('UserMetadata', {}),
            'ETag': f'"{stat.st_mtime}-{stat.st_size}"'
        }

    def delete_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        """
        Delete an object from local storage.

        :param Bucket: Bucket name (ignored)
        :param Key: Object key
        :return: Empty dict (matches S3 behavior)
        """
        file_path = self._key_to_path(Key)
        meta_path = self._metadata_path(file_path)

        if file_path.exists():
            file_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        return {}

    def list_objects_v2(
        self,
        Bucket: str,
        Prefix: str = '',
        ContinuationToken: Optional[str] = None,
        Delimiter: Optional[str] = None,
        MaxKeys: int = 1000
    ) -> Dict[str, Any]:
        """
        List objects under a prefix.

        :param Bucket: Bucket name (ignored)
        :param Prefix: Key prefix to filter by
        :param ContinuationToken: Token for pagination (not fully implemented)
        :param Delimiter: Delimiter for grouping (e.g., '/' for directory-like listing)
        :param MaxKeys: Maximum number of keys to return
        :return: Dict with 'Contents', 'CommonPrefixes', 'IsTruncated'
        """
        prefix_path = self._key_to_path(Prefix)

        # Handle case where prefix is a directory
        if prefix_path.is_dir():
            search_path = prefix_path
            prefix_filter = ''
        else:
            search_path = prefix_path.parent
            prefix_filter = prefix_path.name

        contents: List[Dict[str, Any]] = []
        common_prefixes: List[Dict[str, str]] = []
        seen_prefixes: set = set()

        if not search_path.exists():
            return {
                'Contents': [],
                'CommonPrefixes': [],
                'IsTruncated': False,
                'KeyCount': 0
            }

        # Walk directory tree
        for item in search_path.rglob('*'):
            # Skip metadata files
            if item.name.startswith('.') and item.name.endswith('.metadata.json'):
                continue

            # Skip directories
            if item.is_dir():
                continue

            # Get relative key
            rel_path = item.relative_to(self.base_path)
            key = str(rel_path).replace('\\', '/')

            # Check prefix filter
            if Prefix and not key.startswith(Prefix):
                continue

            # Handle delimiter (directory grouping)
            if Delimiter:
                # Get the part after the prefix
                suffix = key[len(Prefix):] if Prefix else key

                if Delimiter in suffix:
                    # Extract common prefix (up to and including delimiter)
                    prefix_end = suffix.index(Delimiter) + len(Delimiter)
                    common_prefix = Prefix + suffix[:prefix_end]

                    if common_prefix not in seen_prefixes:
                        seen_prefixes.add(common_prefix)
                        common_prefixes.append({'Prefix': common_prefix})
                    continue

            stat = item.stat()
            contents.append({
                'Key': key,
                'Size': stat.st_size,
                'LastModified': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                'ETag': f'"{stat.st_mtime}-{stat.st_size}"'
            })

        # Sort by key
        contents.sort(key=lambda x: x['Key'])
        common_prefixes.sort(key=lambda x: x['Prefix'])

        # Apply MaxKeys limit
        is_truncated = len(contents) > MaxKeys
        if is_truncated:
            contents = contents[:MaxKeys]

        return {
            'Contents': contents,
            'CommonPrefixes': common_prefixes,
            'IsTruncated': is_truncated,
            'KeyCount': len(contents),
            'Prefix': Prefix
        }

    def upload_fileobj(
        self,
        Fileobj: io.BytesIO,
        Bucket: str,
        Key: str,
        Config: Any = None,
        ExtraArgs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Upload a file-like object to local storage.

        This is the streaming upload method used by boto3's S3 transfer.

        :param Fileobj: File-like object to upload
        :param Bucket: Bucket name (ignored)
        :param Key: Object key
        :param Config: Transfer config (ignored for local storage)
        :param ExtraArgs: Extra arguments including ContentType, Metadata
        """
        extra = ExtraArgs or {}
        content_type = extra.get('ContentType', 'application/octet-stream')
        metadata = extra.get('Metadata', {})

        # Read file object
        Fileobj.seek(0)
        data = Fileobj.read()

        self.put_object(
            Bucket=Bucket,
            Key=Key,
            Body=data,
            ContentType=content_type,
            Metadata=metadata
        )
