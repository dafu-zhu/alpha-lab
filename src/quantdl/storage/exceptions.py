"""
Custom exceptions for storage backends.

Provides boto3-compatible exceptions for use with LocalStorageClient.
"""


class NoSuchKeyError(Exception):
    """
    Raised when a requested object key does not exist.

    Compatible with boto3's ClientError interface by providing a .response
    attribute with the same structure as boto3 errors.
    """

    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key
        self.response = {
            'Error': {
                'Code': 'NoSuchKey',
                'Message': f'The specified key does not exist: {key}',
                'Key': key,
                'BucketName': bucket
            }
        }
        super().__init__(f"NoSuchKey: {bucket}/{key}")


class NoSuchBucketError(Exception):
    """
    Raised when a requested bucket (base directory) does not exist.

    Compatible with boto3's ClientError interface.
    """

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.response = {
            'Error': {
                'Code': 'NoSuchBucket',
                'Message': f'The specified bucket does not exist: {bucket}',
                'BucketName': bucket
            }
        }
        super().__init__(f"NoSuchBucket: {bucket}")
