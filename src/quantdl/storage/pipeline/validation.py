import datetime as dt
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from quantdl.storage.clients import S3Client
from typing import Optional
from quantdl.utils.logger import setup_logger

load_dotenv()


class Validator:
    def __init__(self, s3_client=None, bucket_name: Optional[str] = None):
        self.s3_client = s3_client or S3Client().client
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'us-equity-datalake')
        self.log_dir = Path("data/logs/validation")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            name='validation',
            log_dir=self.log_dir,
            level=logging.WARNING
        )

    def list_files_under_prefix(self, prefix: str) -> list[str]:
        """
        List all files (object keys) under a given prefix.

        :param prefix: Prefix/directory to list (e.g., 'data/fundamental')
        :return: List of full object keys under the prefix
        """
        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')

        if storage_backend == 'local':
            return self._list_local_files(prefix, local_path)
        else:
            return self._list_s3_files(prefix)

    def _list_local_files(self, prefix: str, local_path: str) -> list[str]:
        """List all files under a local directory prefix."""
        files = []
        local_dir = Path(local_path) / prefix
        if local_dir.exists():
            for file_path in local_dir.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    rel_path = file_path.relative_to(Path(local_path))
                    files.append(str(rel_path).replace('\\', '/'))
        return files

    def _list_s3_files(self, prefix: str) -> list[str]:
        """List all files under an S3 prefix."""
        files = []
        continuation_token = None

        while True:
            params = {
                'Bucket': self.bucket_name,
                'Prefix': prefix
            }
            if continuation_token:
                params['ContinuationToken'] = continuation_token

            response = self.s3_client.list_objects_v2(**params)

            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])

            if response.get('IsTruncated'):
                continuation_token = response['NextContinuationToken']
            else:
                break

        return files

    def data_exists(
            self,
            symbol,
            data_type: str,
            year: Optional[int] = None,
            cik: Optional[str] = None,
            security_id: Optional[int] = None
        ) -> bool:
        """
        Check if data exists for a given symbol/identifier.

        :param symbol: Stock symbol (used as fallback identifier)
        :param data_type: "ticks" or "fundamental"
        :param year: Year for ticks data (used to check year overlap in single file)
        :param cik: CIK string for fundamental data
        :param security_id: Security ID for ticks and fundamental data
        """
        if data_type == 'ticks':
            identifier = str(security_id) if security_id is not None else symbol
            s3_key = f'data/raw/ticks/daily/{identifier}/ticks.parquet'
        elif data_type == 'fundamental':
            identifier = str(security_id) if security_id is not None else (cik if cik else symbol)
            s3_key = f'data/raw/fundamental/{identifier}/fundamental.parquet'
        else:
            raise ValueError(f'Expected data_type is ticks or fundamental, got {data_type} instead')

        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')

        if storage_backend == 'local':
            local_file = Path(local_path) / s3_key
            return local_file.exists()
        else:
            try:
                self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                return True
            except ClientError as error:
                if error.response.get('Error', {}).get('Code') == '404':
                    return False
                else:
                    self.logger.error(f'Error checking {s3_key}: {error}')
                    return False

    def top_3000_exists(self, year: int, month: int) -> bool:
        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')
        key = f"data/meta/universe/{year}/{month:02d}/top3000.txt"

        if storage_backend == 'local':
            local_file = Path(local_path) / key
            return local_file.exists()
        else:
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                return True
            except ClientError as error:
                if error.response.get('Error', {}).get('Code') == '404':
                    return False
                self.logger.error(f"Error checking {key}: {error}")
                return False
