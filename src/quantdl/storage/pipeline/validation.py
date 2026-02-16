import datetime as dt
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
from typing import Optional
from quantdl.utils.logger import setup_logger

load_dotenv()


class Validator:
    def __init__(self, storage_client=None):
        self.storage_client = storage_client
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
        if self.storage_client is None:
            return self._list_local_files_from_env(prefix)
        return self._list_local_files(prefix)

    def _list_local_files_from_env(self, prefix: str) -> list[str]:
        """List all files under a local directory prefix using LOCAL_STORAGE_PATH."""
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')
        files = []
        local_dir = Path(local_path) / prefix
        if local_dir.exists():
            for file_path in local_dir.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    rel_path = file_path.relative_to(Path(local_path))
                    files.append(str(rel_path).replace('\\', '/'))
        return files

    def _list_local_files(self, prefix: str) -> list[str]:
        """List all files under a local directory prefix using storage client."""
        files = []
        local_dir = self.storage_client.base_path / prefix
        if local_dir.exists():
            for file_path in local_dir.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    rel_path = file_path.relative_to(self.storage_client.base_path)
                    files.append(str(rel_path).replace('\\', '/'))
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
            key = f'data/raw/ticks/daily/{identifier}/ticks.parquet'
        elif data_type == 'fundamental':
            identifier = str(security_id) if security_id is not None else (cik if cik else symbol)
            key = f'data/raw/fundamental/{identifier}/fundamental.parquet'
        else:
            raise ValueError(f'Expected data_type is ticks or fundamental, got {data_type} instead')

        if self.storage_client is not None:
            local_file = self.storage_client.base_path / key
        else:
            local_path = os.getenv('LOCAL_STORAGE_PATH', '')
            local_file = Path(local_path) / key

        return local_file.exists()

    def top_3000_exists(self, year: int, month: int) -> bool:
        key = f"data/meta/universe/{year}/{month:02d}/top3000.txt"

        if self.storage_client is not None:
            local_file = self.storage_client.base_path / key
        else:
            local_path = os.getenv('LOCAL_STORAGE_PATH', '')
            local_file = Path(local_path) / key

        return local_file.exists()
