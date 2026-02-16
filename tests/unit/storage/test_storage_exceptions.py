"""Tests for storage.utils.exceptions module."""

from alphalab.storage.utils.exceptions import NoSuchBucketError, NoSuchKeyError


class TestNoSuchKeyError:
    def test_creates_response(self):
        exc = NoSuchKeyError("my-bucket", "my/key.parquet")
        assert exc.bucket == "my-bucket"
        assert exc.key == "my/key.parquet"
        assert exc.response["Error"]["Code"] == "NoSuchKey"
        assert "my/key.parquet" in exc.response["Error"]["Message"]
        assert str(exc) == "NoSuchKey: my-bucket/my/key.parquet"


class TestNoSuchBucketError:
    def test_creates_response(self):
        exc = NoSuchBucketError("missing-bucket")
        assert exc.bucket == "missing-bucket"
        assert exc.response["Error"]["Code"] == "NoSuchBucket"
        assert "missing-bucket" in exc.response["Error"]["Message"]
        assert str(exc) == "NoSuchBucket: missing-bucket"

    def test_is_exception(self):
        exc = NoSuchBucketError("bucket")
        assert isinstance(exc, Exception)
