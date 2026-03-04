"""Tests for data._utils.exceptions module."""

from alphalab.data._utils.exceptions import NoSuchKeyError


class TestNoSuchKeyError:
    def test_creates_response(self):
        exc = NoSuchKeyError("my-bucket", "my/key.parquet")
        assert exc.bucket == "my-bucket"
        assert exc.key == "my/key.parquet"
        assert exc.response["Error"]["Code"] == "NoSuchKey"
        assert "my/key.parquet" in exc.response["Error"]["Message"]
        assert str(exc) == "NoSuchKey: my-bucket/my/key.parquet"
