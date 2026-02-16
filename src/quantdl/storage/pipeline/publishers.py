"""
Data publishing functionality for uploading market data to local storage.

This module handles publishing collected data to local storage:
- Daily ticks (Parquet files per security_id)
- Fundamental data (Parquet files per security_id)
"""

import io
import json
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import polars as pl
from quantdl.storage.utils import NoSuchKeyError
from dotenv import load_dotenv

load_dotenv()


class DataPublishers:
    """
    Handles publishing market data to local storage.
    """

    def __init__(
        self,
        storage_client,
        logger: logging.Logger,
        data_collectors,
        security_master,
        alpaca_start_year: int = 2025
    ):
        """
        Initialize data publishers.

        :param storage_client: LocalStorageClient instance
        :param logger: Logger instance
        :param data_collectors: Data collectors instance
        :param security_master: SecurityMaster instance for symbol→security_id resolution
        """
        self.storage_client = storage_client
        self.logger = logger
        self.data_collectors = data_collectors
        self.security_master = security_master
        self.alpaca_start_year = alpaca_start_year

    def upload_fileobj(
        self,
        data: io.BytesIO,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Upload file object to local storage."""
        local_path = self.storage_client.base_path
        file_path = Path(local_path) / key
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data file
        data.seek(0)
        file_path.write_bytes(data.read())

        # Write metadata as hidden file
        if metadata:
            metadata_file = file_path.parent / f'.{file_path.name}.metadata.json'
            metadata_file.write_text(json.dumps(metadata, indent=2))

    def publish_daily_ticks(
        self,
        sym: str,
        security_id: int,
        df: pl.DataFrame,
        start_year: int,
        end_year: int,
    ) -> Dict[str, Optional[str]]:
        """
        Publish daily ticks for a single symbol to local storage.

        Single file per security_id with append-merge logic:
        read existing → remove range overlap → concat → write.

        Storage: data/raw/ticks/daily/{security_id}/ticks.parquet

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param security_id: Security ID from SecurityMaster
        :param df: Polars DataFrame with daily ticks data
        :param start_year: Start year of the range being published
        :param end_year: End year of the range being published
        :return: Dict with status info
        """
        try:
            if len(df) == 0:
                return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

            s3_key = f"data/raw/ticks/daily/{security_id}/ticks.parquet"

            # Read existing data and append-merge
            try:
                response = self.storage_client.get_object(
                    Bucket='',
                    Key=s3_key
                )
                existing_df = pl.read_parquet(response['Body'])

                # Remove data within the requested range and replace with new
                existing_df = existing_df.filter(
                    (pl.col('timestamp') < f"{start_year}-01-01") |
                    (pl.col('timestamp') > f"{end_year}-12-31")
                )
                combined_df = pl.concat([existing_df, df]).sort('timestamp')
            except NoSuchKeyError:
                combined_df = df

            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_metadata = {
                'security_id': str(security_id),
                'symbols': [sym],
                'data_type': 'daily_ticks',
                'source': 'alpaca',
                'trading_days': str(len(combined_df)),
            }

            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except ValueError as e:
            if "not active on" in str(e):
                self.logger.info(f'Skipping {sym}: {e}')
                return {'symbol': sym, 'status': 'skipped', 'error': str(e)}
            else:
                self.logger.error(f'ValueError for {sym}: {e}')
                return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except requests.RequestException as e:
            self.logger.warning(f'Failed to publish daily ticks for {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error publishing {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def get_fundamental_metadata(self, security_id: int) -> Optional[Dict[str, str]]:
        """
        Get metadata for existing fundamental data.

        :param security_id: Security ID
        :return: Metadata dict or None if file doesn't exist
        """
        s3_key = f"data/raw/fundamental/{security_id}/fundamental.parquet"
        try:
            response = self.storage_client.head_object(
                Bucket='',
                Key=s3_key
            )
            return response.get('Metadata', {})
        except Exception:
            return None

    def publish_fundamental(
        self,
        sym: str,
        start_date: str,
        end_date: str,
        cik: Optional[str],
        security_id: int,
        sec_rate_limiter,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        """
        Publish fundamental data for a single symbol for a date range to local storage.

        Storage: data/raw/fundamental/{security_id}/fundamental.parquet

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param start_date: Start date (YYYY-MM-DD) for filing date filter
        :param end_date: End date (YYYY-MM-DD) for filing date filter
        :param cik: CIK string (needed for SEC EDGAR API calls)
        :param security_id: Security ID (used for storage path)
        :param sec_rate_limiter: Rate limiter for SEC API
        :param concepts: Optional list of concepts to fetch
        :param config_path: Optional path to sec_mapping.yaml
        :return: Dict with status info
        """
        try:
            if cik is None:
                self.logger.warning(f'Skipping {sym}: No CIK found')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No CIK found for {sym} from {start_date} to {end_date}',
                    'cik': None
                }

            # Rate limit before making SEC API request
            sec_rate_limiter.acquire()

            combined_df = self.data_collectors.collect_fundamental_long(
                cik=cik,
                start_date=start_date,
                end_date=end_date,
                symbol=sym,
                concepts=concepts,
                config_path=config_path
            )

            if len(combined_df) == 0:
                self.logger.warning(
                    f'No fundamental data found for {sym} (CIK {cik}) '
                    f"from {start_date} to {end_date}"
                )
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No fundamental data available for {sym} from {start_date} to {end_date}',
                    'cik': cik
                }

            concepts_total = len(self.data_collectors._load_concepts(concepts, config_path))

            # Calculate latest filing info for metadata tracking
            latest_date = combined_df.select(pl.col("as_of_date").max()).item()
            latest_accn = combined_df.filter(pl.col("as_of_date") == latest_date).select("accn").unique().item()

            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_key = f"data/raw/fundamental/{security_id}/fundamental.parquet"
            concepts_found = combined_df.select(pl.col("concept").n_unique()).item()
            s3_metadata = {
                'symbol': sym,
                'security_id': str(security_id),
                'cik': cik,
                'data_type': 'fundamental',
                'rows': str(len(combined_df)),
                'concepts_found': str(concepts_found),
                'concepts_total': str(concepts_total),
                'start_date': start_date,
                'end_date': end_date,
                'latest_filing_date': str(latest_date),
                'latest_accn': str(latest_accn)
            }
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None, 'cik': cik}

        except requests.RequestException as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.warning(f'Failed to fetch data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e), 'cik': cik}
        except ValueError as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Invalid data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e), 'cik': cik}
        except Exception as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Unexpected error for {sym}{cik_str}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e), 'cik': cik}

    def publish_top_3000(
        self,
        year: int,
        month: int,
        as_of: str,
        symbols: List[str],
        source: str
    ) -> Dict[str, Optional[str]]:
        """
        Publish monthly top 3000 symbols as a newline-delimited text file.

        Storage: data/meta/universe/{YYYY}/{MM}/top3000.txt
        """
        try:
            if not symbols:
                return {
                    'status': 'skipped',
                    'error': 'No symbols provided',
                    'year': str(year),
                    'month': f"{month:02d}"
                }

            content = "\n".join(symbols) + "\n"
            buffer = io.BytesIO(content.encode("utf-8"))
            buffer.seek(0)

            s3_key = f"data/meta/universe/{year}/{month:02d}/top3000.txt"
            s3_metadata = {
                'year': str(year),
                'month': f"{month:02d}",
                'as_of': as_of,
                'data_type': 'top3000',
                'count': str(len(symbols)),
                'source': source
            }
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

            return {
                'status': 'success',
                'error': None,
                'year': str(year),
                'month': f"{month:02d}"
            }
        except Exception as e:
            self.logger.error(
                f"Unexpected error publishing top3000 for {year}-{month:02d}: {e}",
                exc_info=True
            )
            return {
                'status': 'failed',
                'error': str(e),
                'year': str(year),
                'month': f"{month:02d}"
            }
