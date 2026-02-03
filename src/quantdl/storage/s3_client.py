import os
from typing import Dict, Any
from dotenv import load_dotenv
import boto3
from botocore.config import Config
from quantdl.storage.config_loader import UploadConfig

load_dotenv()


class S3Client:
    """
    Storage client factory that returns either boto3 S3 client or LocalStorageClient
    based on STORAGE_BACKEND environment variable.

    Environment variables:
        STORAGE_BACKEND: 'local' for filesystem, 's3' (default) for AWS S3
        LOCAL_STORAGE_PATH: Required when STORAGE_BACKEND=local, path to storage root
    """
    def __init__(self, config_path: str = "configs/storage.yaml"):
        self.config = UploadConfig(config_path)

        # Check storage backend
        self._backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        self._local_path = os.getenv('LOCAL_STORAGE_PATH')

        if self._backend == 'local':
            if not self._local_path:
                raise ValueError(
                    "LOCAL_STORAGE_PATH environment variable required when "
                    "STORAGE_BACKEND=local"
                )
            self._local_client = None
        else:
            self.boto_config = self._create_boto_config()
            # Load key and secrets from .env
            self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    def _create_boto_config(self) -> Config:
        """
        Create boto3 Config object from loaded configuration
        """
        client_cfg = self.config.client

        if not client_cfg:
            raise ValueError("Client configuration is empty or missing")

        # Build with required params
        config_kwargs: Dict[str, Any] = {
            'region_name': client_cfg.get('region_name', 'us-east-2'),
            'max_pool_connections': client_cfg.get('max_pool_connections', 50),
        }

        # Add optional params
        if 'connect_timeout' in client_cfg:
            config_kwargs['connect_timeout'] = client_cfg['connect_timeout']

        if 'read_timeout' in client_cfg:
            config_kwargs['read_timeout'] = client_cfg['read_timeout']

        if 'retries' in client_cfg:
            retries_cfg = client_cfg['retries']
            if isinstance(retries_cfg, dict):
                config_kwargs['retries'] = {
                    'mode': retries_cfg.get('mode', 'standard'),
                    'total_max_attempts': retries_cfg.get('total_max_attempts', 5)
                }

        if 's3' in client_cfg:
            s3_cfg = client_cfg['s3']
            if isinstance(s3_cfg, dict):
                config_kwargs['s3'] = {}

                if 'addressing_style' in s3_cfg:
                    config_kwargs['s3']['addressing_style'] = s3_cfg['addressing_style']

                if 'payload_signing_enabled' in s3_cfg:
                    config_kwargs['s3']['payload_signing_enabled'] = s3_cfg['payload_signing_enabled']

                if 'us_east_1_regional_endpoint' in s3_cfg:
                    config_kwargs['s3']['us_east_1_regional_endpoint'] = s3_cfg['us_east_1_regional_endpoint']

        if 'tcp_keepalive' in client_cfg:
            config_kwargs['tcp_keepalive'] = client_cfg['tcp_keepalive']

        if 'request_min_compression_size_bytes' in client_cfg:
            config_kwargs['request_min_compression_size_bytes'] = client_cfg['request_min_compression_size_bytes']

        if 'disable_request_compression' in client_cfg:
            config_kwargs['disable_request_compression'] = client_cfg['disable_request_compression']

        if 'request_checksum_calculation' in client_cfg:
            config_kwargs['request_checksum_calculation'] = client_cfg['request_checksum_calculation']

        if 'response_checksum_validation' in client_cfg:
            config_kwargs['response_checksum_validation'] = client_cfg['response_checksum_validation']

        return Config(**config_kwargs)
    
    @property
    def client(self):
        """
        Return storage client based on configured backend.

        For STORAGE_BACKEND=local: Returns LocalStorageClient
        For STORAGE_BACKEND=s3 (default): Returns boto3 S3 client
        """
        if self._backend == 'local':
            if self._local_client is None:
                from quantdl.storage.local_client import LocalStorageClient
                self._local_client = LocalStorageClient(self._local_path)
            return self._local_client

        return boto3.client(
            's3',
            config=self.boto_config,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

    @property
    def is_local(self) -> bool:
        """Return True if using local filesystem storage backend."""
        return self._backend == 'local'