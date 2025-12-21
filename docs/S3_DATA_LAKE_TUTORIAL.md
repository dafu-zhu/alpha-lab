# Building a Data Lake with AWS S3 and Boto3

## Introduction

This tutorial teaches you how to build the US Equity Data Lake on AWS S3 using Boto3. We focus **only** on the core methods and parameters you'll actually use in this project.

### What You'll Learn

- **S3 Client Configuration**: Optimized settings for data lake operations
- **Upload Operations**: Single files, parallel uploads, from memory
- **Download Operations**: Single files, parallel downloads, to memory
- **Batch Operations**: Uploading/downloading thousands of files efficiently
- **Transfer Optimization**: Multipart uploads and concurrency tuning

### Prerequisites

- Python 3.8+
- AWS credentials configured (see [S3_CONFIGURATION.md](S3_CONFIGURATION.md))
- Boto3 installed: `pip install boto3`

---

## Table of Contents

1. [Core S3 Client Methods](#1-core-s3-client-methods)
2. [Upload Operations](#2-upload-operations)
3. [Download Operations](#3-download-operations)
4. [Listing and Checking Objects](#4-listing-and-checking-objects)
5. [Transfer Optimization](#5-transfer-optimization)
6. [Batch Operations](#6-batch-operations)
7. [Production Patterns](#7-production-patterns)
8. [Complete Implementation](#8-complete-implementation)

---

## 1. Core S3 Client Methods

### Methods Used in Data Lake Project

```python
import boto3
from botocore.config import Config

# Initialize S3 client (covered in S3_CONFIGURATION.md)
s3_client = boto3.client('s3', config=config)

# Core methods we use:
s3_client.upload_file(Filename, Bucket, Key, Config, ExtraArgs, Callback)
s3_client.download_file(Bucket, Key, Filename, Config, Callback)
s3_client.upload_fileobj(Fileobj, Bucket, Key, Config, ExtraArgs, Callback)
s3_client.download_fileobj(Bucket, Key, Fileobj, Config, Callback)
s3_client.list_objects_v2(Bucket, Prefix, MaxKeys, ContinuationToken)
s3_client.head_object(Bucket, Key)
```

### Why These Methods?

| Method | Use Case in Data Lake | Why Necessary |
|--------|----------------------|---------------|
| `upload_file` | Upload Parquet files from disk to S3 | Main method for daily/minute tick uploads |
| `download_file` | Download Parquet files from S3 to disk | Query API downloads data for analysis |
| `upload_fileobj` | Upload JSON data directly from memory | Avoid disk I/O for small fundamental data |
| `download_fileobj` | Read JSON data directly into memory | Query API returns data without disk writes |
| `list_objects_v2` | List all tick files for a symbol | Query API needs to find all available dates |
| `head_object` | Check if file exists before download | Avoid errors when checking data availability |

---

## 2. Upload Operations

### 2.1 upload_file() - Upload from Disk

**Purpose:** Upload a file from local disk to S3.

**All Parameters Explained:**

```python
s3_client.upload_file(
    Filename='local/path/to/file.parquet',  # REQUIRED: Local file path
    Bucket='us-equity-datalake',             # REQUIRED: S3 bucket name
    Key='data/ticks/daily/AAPL/2024/ticks.parquet',  # REQUIRED: S3 object key

    Config=TransferConfig(...),              # OPTIONAL: Transfer optimization settings

    ExtraArgs={                              # OPTIONAL: Additional S3 parameters
        'Metadata': {                        # Custom metadata (key-value pairs)
            'symbol': 'AAPL',
            'collection-date': '2024-12-19',
            'data-type': 'daily-ticks'
        },
        'ContentType': 'application/octet-stream'  # MIME type
    },

    Callback=ProgressPercentage(filename)    # OPTIONAL: Progress tracking function
)
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `Filename` | str | ✅ Yes | Path to local Parquet/JSON file to upload |
| `Bucket` | str | ✅ Yes | Always `'us-equity-datalake'` in our project |
| `Key` | str | ✅ Yes | S3 path following our structure: `data/ticks/daily/{symbol}/{year}/ticks.parquet` |
| `Config` | TransferConfig | ⚠️ Recommended | Controls multipart upload, concurrency (see section 5) |
| `ExtraArgs` | dict | ❌ Optional | We use `Metadata` to track data lineage (source, date, symbol) |
| `Callback` | callable | ❌ Optional | We use for progress bars during large uploads |

**Example Usage in Data Lake:**

```python
import boto3
from botocore.exceptions import ClientError
import logging

s3_client = boto3.client('s3')

def upload_daily_ticks(symbol, year, local_file_path):
    """
    Upload daily tick data for a symbol/year.

    Example: Upload AAPL 2024 daily ticks
    Local:  data/AAPL_2024.parquet
    S3:     data/ticks/daily/AAPL/2024/ticks.parquet
    """
    bucket = 'us-equity-datalake'
    s3_key = f'data/ticks/daily/{symbol}/{year}/ticks.parquet'

    try:
        s3_client.upload_file(
            Filename=local_file_path,
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs={
                'Metadata': {
                    'symbol': symbol,
                    'year': str(year),
                    'data-type': 'daily-ticks'
                }
            }
        )
        logging.info(f"Uploaded {symbol} {year} to {s3_key}")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False

# Usage
upload_daily_ticks('AAPL', 2024, 'data/AAPL_2024.parquet')
```

### 2.2 upload_fileobj() - Upload from Memory

**Purpose:** Upload data directly from memory without creating a temporary file.

**All Parameters Explained:**

```python
import io

s3_client.upload_fileobj(
    Fileobj=io.BytesIO(data_bytes),          # REQUIRED: File-like object
    Bucket='us-equity-datalake',             # REQUIRED: S3 bucket name
    Key='data/fundamental/AAPL/2024/Q4.json', # REQUIRED: S3 object key

    Config=TransferConfig(...),              # OPTIONAL: Transfer settings
    ExtraArgs={'Metadata': {...}},           # OPTIONAL: Metadata
    Callback=progress_callback               # OPTIONAL: Progress tracking
)
```

**When to Use in Data Lake:**
- ✅ Uploading small fundamental data (JSON) - no need to write to disk first
- ✅ Uploading reference data (ticker metadata, index constituents)
- ❌ NOT for large Parquet files - use `upload_file` instead

**Example Usage:**

```python
import io
import json

def upload_fundamental_data(symbol, year, quarter, data_dict):
    """
    Upload fundamental data directly from Python dict to S3.

    No temporary file created - saves disk I/O.
    """
    s3_client = boto3.client('s3')

    # Convert dict to JSON bytes
    json_bytes = json.dumps(data_dict, indent=2).encode('utf-8')
    file_obj = io.BytesIO(json_bytes)

    s3_key = f'data/fundamental/{symbol}/{year}/Q{quarter}.json'

    try:
        s3_client.upload_fileobj(
            Fileobj=file_obj,
            Bucket='us-equity-datalake',
            Key=s3_key,
            ExtraArgs={
                'Metadata': {
                    'symbol': symbol,
                    'year': str(year),
                    'quarter': str(quarter),
                    'source': 'sec-edgar'
                },
                'ContentType': 'application/json'
            }
        )
        logging.info(f"Uploaded {symbol} {year} Q{quarter} fundamental data")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False

# Usage
fundamental_data = {
    'symbol': 'AAPL',
    'revenue': 123456789,
    'net_income': 23456789,
    # ... more metrics
}
upload_fundamental_data('AAPL', 2024, 4, fundamental_data)
```

---

## 3. Download Operations

### 3.1 download_file() - Download to Disk

**Purpose:** Download a file from S3 to local disk.

**All Parameters Explained:**

```python
s3_client.download_file(
    Bucket='us-equity-datalake',             # REQUIRED: S3 bucket name
    Key='data/ticks/daily/AAPL/2024/ticks.parquet',  # REQUIRED: S3 object key
    Filename='local/downloads/AAPL_2024.parquet',    # REQUIRED: Local destination path

    Config=TransferConfig(...),              # OPTIONAL: Transfer settings
    Callback=progress_callback               # OPTIONAL: Progress tracking
)
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `Bucket` | str | ✅ Yes | Always `'us-equity-datalake'` |
| `Key` | str | ✅ Yes | S3 path to the file we want to download |
| `Filename` | str | ✅ Yes | Where to save the downloaded file locally |
| `Config` | TransferConfig | ⚠️ Recommended | Controls download concurrency for large files |
| `Callback` | callable | ❌ Optional | Progress tracking for user feedback |

**Example Usage in Data Lake:**

```python
def download_daily_ticks(symbol, year, local_dir='downloads'):
    """
    Download daily tick data for analysis.

    S3:     data/ticks/daily/AAPL/2024/ticks.parquet
    Local:  downloads/AAPL_2024.parquet
    """
    import os
    from pathlib import Path

    s3_client = boto3.client('s3')

    # Create download directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    s3_key = f'data/ticks/daily/{symbol}/{year}/ticks.parquet'
    local_path = os.path.join(local_dir, f'{symbol}_{year}.parquet')

    try:
        s3_client.download_file(
            Bucket='us-equity-datalake',
            Key=s3_key,
            Filename=local_path
        )
        logging.info(f"Downloaded {symbol} {year} to {local_path}")
        return local_path
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logging.error(f"File not found: {s3_key}")
        else:
            logging.error(f"Download failed: {e}")
        return None

# Usage
local_file = download_daily_ticks('AAPL', 2024)
if local_file:
    import pandas as pd
    df = pd.read_parquet(local_file)
    print(df.head())
```

### 3.2 download_fileobj() - Download to Memory

**Purpose:** Download file directly into memory (no disk I/O).

**All Parameters Explained:**

```python
import io

file_obj = io.BytesIO()

s3_client.download_fileobj(
    Bucket='us-equity-datalake',             # REQUIRED: S3 bucket name
    Key='data/fundamental/AAPL/2024/Q4.json', # REQUIRED: S3 object key
    Fileobj=file_obj,                        # REQUIRED: File-like object to write to

    Config=TransferConfig(...),              # OPTIONAL: Transfer settings
    Callback=progress_callback               # OPTIONAL: Progress tracking
)

file_obj.seek(0)  # Reset to beginning
data = file_obj.read()  # Read bytes
```

**When to Use in Data Lake:**
- ✅ Query API returning data without saving to disk
- ✅ Small files (fundamentals, reference data) that fit in memory
- ❌ NOT for large Parquet files (hundreds of MB) - use `download_file`

**Example Usage:**

```python
import io
import json

def get_fundamental_data(symbol, year, quarter):
    """
    Download fundamental data directly into memory and return as dict.

    No temporary file created - faster for small JSON files.
    """
    s3_client = boto3.client('s3')
    s3_key = f'data/fundamental/{symbol}/{year}/Q{quarter}.json'

    try:
        # Download to memory
        file_obj = io.BytesIO()
        s3_client.download_fileobj(
            Bucket='us-equity-datalake',
            Key=s3_key,
            Fileobj=file_obj
        )

        # Parse JSON
        file_obj.seek(0)
        data = json.loads(file_obj.read().decode('utf-8'))

        logging.info(f"Retrieved {symbol} {year} Q{quarter} fundamental data")
        return data

    except ClientError as e:
        logging.error(f"Download failed: {e}")
        return None

# Usage
data = get_fundamental_data('AAPL', 2024, 4)
if data:
    print(f"Revenue: ${data['revenue']:,}")
    print(f"Net Income: ${data['net_income']:,}")
```

---

## 4. Listing and Checking Objects

### 4.1 list_objects_v2() - List Files in S3

**Purpose:** List all objects with a given prefix (folder path).

**All Parameters Explained:**

```python
response = s3_client.list_objects_v2(
    Bucket='us-equity-datalake',             # REQUIRED: S3 bucket name

    Prefix='data/ticks/daily/AAPL/',         # OPTIONAL: Filter by prefix
    MaxKeys=1000,                            # OPTIONAL: Max results per page (default: 1000)
    ContinuationToken='...',                 # OPTIONAL: For pagination
    StartAfter='data/ticks/daily/AAPL/2020/' # OPTIONAL: Start listing after this key
)

# Response structure:
{
    'Contents': [
        {'Key': 'data/ticks/daily/AAPL/2024/ticks.parquet', 'Size': 12345, 'LastModified': ...},
        {'Key': 'data/ticks/daily/AAPL/2023/ticks.parquet', 'Size': 11234, 'LastModified': ...},
    ],
    'IsTruncated': False,
    'NextContinuationToken': '...'  # Use for next page if IsTruncated=True
}
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `Bucket` | str | ✅ Yes | Our data lake bucket |
| `Prefix` | str | ⚠️ Recommended | Filter files: `'data/ticks/daily/AAPL/'` returns only AAPL files |
| `MaxKeys` | int | ❌ Optional | Limit results (default 1000). We rarely need to change this. |
| `ContinuationToken` | str | ❌ Auto | Used for pagination when results > 1000 |
| `StartAfter` | str | ❌ Optional | We don't use this (Prefix is sufficient) |

**Example Usage in Data Lake:**

```python
def list_available_years(symbol):
    """
    List all years we have daily tick data for a symbol.

    Example: For AAPL, return [2024, 2023, 2022, ...]
    """
    s3_client = boto3.client('s3')
    prefix = f'data/ticks/daily/{symbol}/'

    years = []
    continuation_token = None

    while True:
        # Build request parameters
        params = {
            'Bucket': 'us-equity-datalake',
            'Prefix': prefix
        }
        if continuation_token:
            params['ContinuationToken'] = continuation_token

        # List objects
        response = s3_client.list_objects_v2(**params)

        # Extract years from object keys
        if 'Contents' in response:
            for obj in response['Contents']:
                # Key format: data/ticks/daily/AAPL/2024/ticks.parquet
                parts = obj['Key'].split('/')
                if len(parts) >= 5:
                    year = int(parts[4])
                    if year not in years:
                        years.append(year)

        # Check for more pages
        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    return sorted(years, reverse=True)

# Usage
years = list_available_years('AAPL')
print(f"AAPL data available for: {years}")
# Output: AAPL data available for: [2024, 2023, 2022, 2021, ...]
```

### 4.2 head_object() - Check if File Exists

**Purpose:** Check if an object exists and get its metadata without downloading it.

**All Parameters Explained:**

```python
response = s3_client.head_object(
    Bucket='us-equity-datalake',             # REQUIRED: S3 bucket name
    Key='data/ticks/daily/AAPL/2024/ticks.parquet'  # REQUIRED: S3 object key
)

# Response structure:
{
    'ContentLength': 12345,
    'ContentType': 'application/octet-stream',
    'LastModified': datetime(2024, 12, 19, ...),
    'Metadata': {
        'symbol': 'AAPL',
        'year': '2024',
        'data-type': 'daily-ticks'
    }
}
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `Bucket` | str | ✅ Yes | Our data lake bucket |
| `Key` | str | ✅ Yes | S3 path to check |

**Example Usage in Data Lake:**

```python
def data_exists(symbol, year):
    """
    Check if we have daily tick data for a symbol/year.

    Returns True if file exists, False otherwise.
    """
    s3_client = boto3.client('s3')
    s3_key = f'data/ticks/daily/{symbol}/{year}/ticks.parquet'

    try:
        s3_client.head_object(
            Bucket='us-equity-datalake',
            Key=s3_key
        )
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Other error (permissions, etc.)
            logging.error(f"Error checking {s3_key}: {e}")
            return False

# Usage in Query API
def get_daily_ticks(symbol, year):
    """Download data only if it exists."""
    if not data_exists(symbol, year):
        logging.warning(f"No data for {symbol} {year}")
        return None

    return download_daily_ticks(symbol, year)

# Usage
df = get_daily_ticks('AAPL', 2024)
```

---

## 5. Transfer Optimization

### TransferConfig Class

**Purpose:** Control how boto3 handles file transfers (multipart uploads, concurrency, threading).

**All Parameters Explained:**

```python
from boto3.s3.transfer import TransferConfig

config = TransferConfig(
    multipart_threshold=10 * 1024 * 1024,    # 10 MB - when to use multipart upload
    max_concurrency=20,                      # Number of parallel uploads
    multipart_chunksize=10 * 1024 * 1024,    # 10 MB - size of each part
    num_download_attempts=5,                 # Retry attempts per chunk
    max_io_queue=100,                        # Internal queue size
    io_chunksize=262144,                     # 256 KB - read/write chunk size
    use_threads=True                         # Enable threading
)
```

**Parameter Details for Data Lake:**

| Parameter | Default | Our Setting | Why? |
|-----------|---------|-------------|------|
| `multipart_threshold` | 8 MB | **10 MB** | Files > 10 MB split into parts. Our Parquet files are 10-500 MB. |
| `max_concurrency` | 10 | **20** | Upload 20 parts simultaneously. We have good bandwidth, want speed. |
| `multipart_chunksize` | 8 MB | **10 MB** | Each part is 10 MB. Balance: fewer HTTP requests vs. retry cost. |
| `num_download_attempts` | 5 | **5** (keep default) | Retry 5 times per chunk. Good for unreliable networks. |
| `max_io_queue` | 100 | **100** (keep default) | Internal queue size. No need to change. |
| `io_chunksize` | 256 KB | **256 KB** (keep default) | Read/write chunk size. No need to change. |
| `use_threads` | True | **True** (keep default) | Enable threading for parallel uploads. Always keep True. |

**Visual: How Multipart Upload Works**

```
Without multipart (file < 10 MB):
File (8 MB) ────────────────────────> S3
Single HTTP request, ~2 seconds

With multipart (file > 10 MB):
File (100 MB) split into 10 parts (10 MB each)
Part 1 (10 MB) ──────> S3
Part 2 (10 MB) ──────> S3  } 20 concurrent uploads
Part 3 (10 MB) ──────> S3  }
...                         } Much faster!
Part 10 (10 MB) ─────> S3

Upload time: ~5 seconds instead of ~20 seconds
```

**Example Usage in Data Lake:**

```python
from boto3.s3.transfer import TransferConfig

# Configure for our data lake (optimized for 10-500 MB Parquet files)
transfer_config = TransferConfig(
    multipart_threshold=10 * 1024 * 1024,   # 10 MB
    max_concurrency=20,                      # 20 parallel parts
    multipart_chunksize=10 * 1024 * 1024,   # 10 MB parts
    use_threads=True
)

def upload_daily_ticks_optimized(symbol, year, local_file_path):
    """Upload with optimized transfer settings."""
    s3_client = boto3.client('s3')
    s3_key = f'data/ticks/daily/{symbol}/{year}/ticks.parquet'

    try:
        s3_client.upload_file(
            Filename=local_file_path,
            Bucket='us-equity-datalake',
            Key=s3_key,
            Config=transfer_config  # Use our optimized config
        )
        logging.info(f"Uploaded {symbol} {year} (optimized)")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False

# Usage
upload_daily_ticks_optimized('AAPL', 2024, 'data/AAPL_2024.parquet')
```

---

## 6. Batch Operations

### Parallel Upload with ThreadPoolExecutor

**Purpose:** Upload thousands of files efficiently using parallel threads.

**Why Necessary for Data Lake:**
- Daily backfill: Upload 5,000 symbols × 15 years = 75,000 files
- Sequential upload: ~75,000 seconds (~21 hours)
- Parallel upload (20 workers): ~3,750 seconds (~1 hour)

**Example Implementation:**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging

def upload_directory_parallel(local_dir, s3_prefix, max_workers=20):
    """
    Upload entire directory to S3 in parallel.

    Args:
        local_dir: Local directory (e.g., 'data/ticks/daily')
        s3_prefix: S3 prefix (e.g., 'data/ticks/daily')
        max_workers: Number of parallel uploads (default: 20)

    Returns:
        Dictionary with success/failure counts
    """
    s3_client = boto3.client('s3')
    local_dir = Path(local_dir)

    # Find all Parquet files to upload
    files_to_upload = []
    for file_path in local_dir.rglob('*.parquet'):
        relative_path = file_path.relative_to(local_dir)
        s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
        files_to_upload.append((str(file_path), s3_key))

    total = len(files_to_upload)
    logging.info(f"Found {total} files to upload")

    # Upload function for each file
    def upload_single_file(file_info):
        local_path, s3_key = file_info
        try:
            s3_client.upload_file(
                Filename=local_path,
                Bucket='us-equity-datalake',
                Key=s3_key
            )
            return True, s3_key
        except Exception as e:
            logging.error(f"Failed {s3_key}: {e}")
            return False, s3_key

    # Upload in parallel
    results = {'success': 0, 'failed': 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(upload_single_file, f): f for f in files_to_upload}

        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            success, s3_key = future.result()
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1

            # Progress update
            print(f"\rProgress: {i}/{total} ({i/total*100:.1f}%)", end='')

    print()  # New line
    logging.info(f"Upload complete: {results['success']} success, {results['failed']} failed")
    return results

# Usage: Daily ticks backfill
results = upload_directory_parallel(
    local_dir='data/ticks/daily',
    s3_prefix='data/ticks/daily',
    max_workers=20  # 20 concurrent uploads
)
```

### Parallel Download

```python
def download_directory_parallel(s3_prefix, local_dir, max_workers=20):
    """
    Download all files with a given prefix in parallel.

    Example: Download all AAPL daily ticks (all years)
    """
    s3_client = boto3.client('s3')
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # List all objects with prefix
    objects = []
    continuation_token = None

    while True:
        params = {'Bucket': 'us-equity-datalake', 'Prefix': s3_prefix}
        if continuation_token:
            params['ContinuationToken'] = continuation_token

        response = s3_client.list_objects_v2(**params)

        if 'Contents' in response:
            objects.extend([obj['Key'] for obj in response['Contents']])

        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    logging.info(f"Found {len(objects)} files to download")

    # Download function
    def download_single_file(s3_key):
        relative_path = s3_key.replace(s3_prefix, '').lstrip('/')
        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3_client.download_file(
                Bucket='us-equity-datalake',
                Key=s3_key,
                Filename=str(local_path)
            )
            return True, s3_key
        except Exception as e:
            logging.error(f"Failed {s3_key}: {e}")
            return False, s3_key

    # Download in parallel
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_file, key): key for key in objects}

        for future in as_completed(futures):
            success, s3_key = future.result()
            if success:
                downloaded += 1
            print(f"\rDownloaded: {downloaded}/{len(objects)}", end='')

    print()
    logging.info(f"Download complete: {downloaded} files")
    return downloaded

# Usage: Download all AAPL data
download_directory_parallel(
    s3_prefix='data/ticks/daily/AAPL',
    local_dir='downloads/AAPL',
    max_workers=20
)
```

---

## 7. Production Patterns

### Error Handling with Retries

```python
import time
from botocore.exceptions import ClientError

def upload_with_retry(local_path, s3_key, max_retries=3):
    """
    Upload with automatic retry on transient errors.

    Implements exponential backoff: 1s, 2s, 4s between retries.
    """
    s3_client = boto3.client('s3')

    for attempt in range(max_retries):
        try:
            s3_client.upload_file(
                Filename=local_path,
                Bucket='us-equity-datalake',
                Key=s3_key
            )
            logging.info(f"Upload succeeded on attempt {attempt + 1}")
            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']

            # Permanent errors - don't retry
            if error_code in ['NoSuchBucket', 'AccessDenied']:
                logging.error(f"Permanent error: {error_code}")
                return False

            # Transient errors - retry with backoff
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"Failed after {max_retries} attempts")
                return False

    return False

# Usage
upload_with_retry('data/AAPL_2024.parquet', 'data/ticks/daily/AAPL/2024/ticks.parquet')
```

### Logging and Monitoring

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_lake_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)

def upload_with_logging(local_path, s3_key):
    """Upload with detailed logging and metrics."""
    import time
    from pathlib import Path

    s3_client = boto3.client('s3')

    # Log upload start
    file_size = Path(local_path).stat().st_size
    logging.info(f"Starting upload: {local_path} ({file_size / 1024 / 1024:.2f} MB)")

    start = time.time()

    try:
        s3_client.upload_file(
            Filename=local_path,
            Bucket='us-equity-datalake',
            Key=s3_key
        )

        # Log success with metrics
        duration = time.time() - start
        speed = (file_size / 1024 / 1024) / duration if duration > 0 else 0

        logging.info(
            f"Upload complete: {s3_key} | "
            f"Duration: {duration:.2f}s | "
            f"Speed: {speed:.2f} MB/s"
        )
        return True

    except ClientError as e:
        logging.error(f"Upload failed: {s3_key} | Error: {e}")
        return False
```

---

## 8. Complete Implementation

### Production-Ready DataLakeUploader Class

This class combines all the methods above into a production-ready uploader:

```python
"""
US Equity Data Lake Uploader

Production-ready S3 uploader with:
- Parallel uploads
- Error handling and retries
- Logging and monitoring
- Transfer optimization
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
import logging
import time


@dataclass
class UploadResult:
    """Result of a single file upload."""
    success: bool
    local_path: str
    s3_key: str
    duration: float
    error: Optional[str] = None


class DataLakeUploader:
    """
    Production-ready S3 data lake uploader.

    Features:
    - Optimized transfer configuration
    - Parallel uploads
    - Automatic retry with exponential backoff
    - Progress tracking and logging
    """

    def __init__(
        self,
        bucket_name: str = 'us-equity-datalake',
        region: str = 'us-east-2',
        max_workers: int = 20,
        max_retries: int = 3
    ):
        """
        Initialize uploader.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            max_workers: Max parallel uploads
            max_retries: Max retry attempts per file
        """
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        self.max_retries = max_retries

        # Configure S3 client (from S3_CONFIGURATION.md)
        s3_config = Config(
            region_name=region,
            max_pool_connections=max_workers,
            retries={'mode': 'standard', 'total_max_attempts': max_retries},
            tcp_keepalive=True
        )
        self.s3_client = boto3.client('s3', config=s3_config)

        # Configure transfer settings
        self.transfer_config = TransferConfig(
            multipart_threshold=10 * 1024 * 1024,
            max_concurrency=20,
            multipart_chunksize=10 * 1024 * 1024,
            use_threads=True
        )

        self.logger = logging.getLogger(__name__)

    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> UploadResult:
        """
        Upload single file with retry logic.

        Args:
            local_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata

        Returns:
            UploadResult object
        """
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                extra_args = {}
                if metadata:
                    extra_args['Metadata'] = metadata

                self.s3_client.upload_file(
                    Filename=local_path,
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Config=self.transfer_config,
                    ExtraArgs=extra_args if extra_args else None
                )

                duration = time.time() - start_time
                self.logger.info(f"✓ Uploaded {s3_key} in {duration:.2f}s")

                return UploadResult(
                    success=True,
                    local_path=local_path,
                    s3_key=s3_key,
                    duration=duration
                )

            except ClientError as e:
                error_code = e.response['Error']['Code']

                # Permanent errors
                if error_code in ['NoSuchBucket', 'AccessDenied']:
                    self.logger.error(f"✗ Permanent error for {s3_key}: {error_code}")
                    return UploadResult(
                        success=False,
                        local_path=local_path,
                        s3_key=s3_key,
                        duration=time.time() - start_time,
                        error=error_code
                    )

                # Retry transient errors
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    self.logger.warning(
                        f"Upload failed for {s3_key} (attempt {attempt + 1}), "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    self.logger.error(f"✗ Failed {s3_key} after {self.max_retries} attempts")
                    return UploadResult(
                        success=False,
                        local_path=local_path,
                        s3_key=s3_key,
                        duration=time.time() - start_time,
                        error=str(e)
                    )

        return UploadResult(
            success=False,
            local_path=local_path,
            s3_key=s3_key,
            duration=time.time() - start_time,
            error="Max retries exceeded"
        )

    def upload_directory(
        self,
        local_dir: str,
        s3_prefix: str,
        file_pattern: str = '*.parquet'
    ) -> Dict[str, any]:
        """
        Upload entire directory in parallel.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix (folder)
            file_pattern: Glob pattern (default: '*.parquet')

        Returns:
            Dictionary with upload statistics
        """
        local_dir = Path(local_dir)

        # Find all files matching pattern
        files_to_upload = []
        for file_path in local_dir.rglob(file_pattern):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                files_to_upload.append((str(file_path), s3_key))

        total_files = len(files_to_upload)
        self.logger.info(f"Found {total_files} files to upload")

        # Upload in parallel
        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.upload_file, local_path, s3_key): (local_path, s3_key)
                for local_path, s3_key in files_to_upload
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Progress update
                completed = len(results)
                percent = (completed / total_files) * 100
                print(f"\rProgress: {completed}/{total_files} ({percent:.1f}%)", end='')

        print()

        # Calculate statistics
        total_duration = time.time() - start_time
        successful = sum(1 for r in results if r.success)

        stats = {
            'total_files': total_files,
            'successful': successful,
            'failed': total_files - successful,
            'total_duration': total_duration,
            'average_duration': total_duration / total_files if total_files > 0 else 0
        }

        self.logger.info(
            f"Upload complete: {successful}/{total_files} succeeded in {total_duration:.2f}s"
        )

        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    uploader = DataLakeUploader(
        bucket_name='us-equity-datalake',
        region='us-east-2',
        max_workers=20
    )

    # Upload daily ticks directory
    stats = uploader.upload_directory(
        local_dir='data/ticks/daily',
        s3_prefix='data/ticks/daily',
        file_pattern='*.parquet'
    )

    print(f"\nStatistics:")
    print(f"  Total: {stats['total_files']}")
    print(f"  Success: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Duration: {stats['total_duration']:.2f}s")
```

---

## Summary

### Core S3 Methods Used in Data Lake

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `upload_file()` | Upload Parquet from disk | Daily/minute tick collection |
| `upload_fileobj()` | Upload JSON from memory | Fundamental data, reference data |
| `download_file()` | Download to disk | Query API large datasets |
| `download_fileobj()` | Download to memory | Query API small datasets |
| `list_objects_v2()` | List available files | Query API discovery |
| `head_object()` | Check file existence | Query API validation |

### Key Parameters

- **TransferConfig**: `multipart_threshold=10MB`, `max_concurrency=20` for optimal speed
- **ThreadPoolExecutor**: `max_workers=20` for parallel batch operations
- **Retries**: `max_retries=3` with exponential backoff for reliability

### Next Steps

1. Review [S3_CONFIGURATION.md](S3_CONFIGURATION.md) for client configuration
2. Review [LAMBDA_TUTORIAL.md](LAMBDA_TUTORIAL.md) for automated collection
3. Review [EVENTBRIDGE_TUTORIAL.md](EVENTBRIDGE_TUTORIAL.md) for scheduling
4. Implement the `DataLakeUploader` class in your project
