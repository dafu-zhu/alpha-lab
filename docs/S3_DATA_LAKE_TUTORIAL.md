# Building a Data Lake with AWS S3 and Boto3

## Introduction

This comprehensive tutorial teaches you how to build a production-ready data lake on AWS S3 using the Boto3 Python SDK. The tutorial is structured around the US Equity Data Lake project, but the concepts apply to any large-scale data storage system.

### What You'll Learn

- **Bucket Management**: Creating and organizing S3 buckets for data lake architecture
- **Data Upload Strategies**: Efficient file uploads with progress tracking and error handling
- **Data Download Patterns**: Retrieving data from your lake efficiently
- **Transfer Optimization**: Configuring multipart transfers, concurrency, and threading
- **Security & Access Control**: Presigned URLs for secure temporary access
- **Production Best Practices**: Error handling, retry logic, and monitoring

### Prerequisites

- Python 3.8+
- AWS Account with S3 access
- AWS credentials configured (see [S3_CONFIGURATION.md](S3_CONFIGURATION.md))
- Basic understanding of Python and AWS S3 concepts

---

## Table of Contents

1. [Understanding Data Lake Architecture on S3](#1-understanding-data-lake-architecture-on-s3)
2. [Setting Up Your S3 Bucket](#2-setting-up-your-s3-bucket)
3. [Uploading Data to Your Data Lake](#3-uploading-data-to-your-data-lake)
4. [Downloading Data from Your Data Lake](#4-downloading-data-from-your-data-lake)
5. [Optimizing File Transfers](#5-optimizing-file-transfers)
6. [Batch Operations and Parallelization](#6-batch-operations-and-parallelization)
7. [Managing Access with Presigned URLs](#7-managing-access-with-presigned-urls)
8. [Production Patterns and Best Practices](#8-production-patterns-and-best-practices)
9. [Complete Implementation Example](#9-complete-implementation-example)

---

## 1. Understanding Data Lake Architecture on S3

### What is a Data Lake?

A data lake is a centralized repository that stores all structured and unstructured data at any scale. Unlike traditional databases, data lakes:

- Store raw data in native formats (JSON, Parquet, CSV)
- Support massive scale (petabytes of data)
- Enable schema-on-read (define structure when reading, not when writing)
- Provide cost-effective long-term storage

### Why S3 for Data Lakes?

Amazon S3 is ideal for data lakes because it provides:

✅ **Infinite scalability** - Store unlimited data
✅ **Durability** - 99.999999999% (11 9's) durability
✅ **Cost-effective** - Pay only for what you use (~$0.023/GB/month)
✅ **High availability** - 99.99% availability SLA
✅ **Integration** - Works with AWS analytics services (Athena, Glue, EMR)

### Our Data Lake Structure

For the US Equity Data Lake, we organize data by type and time partitions:

```
s3://us-equity-datalake/
├── ticks/
│   ├── daily/
│   │   ├── AAPL/
│   │   │   ├── 2024/
│   │   │   │   └── ticks.json
│   │   │   ├── 2023/
│   │   │   │   └── ticks.json
│   ├── minute/
│   │   ├── AAPL/
│   │   │   ├── 2024/
│   │   │   │   ├── 01/
│   │   │   │   │   ├── 15/
│   │   │   │   │   │   └── ticks.parquet
├── fundamental/
│   ├── AAPL/
│   │   ├── 2024/
│   │   │   └── fundamental.json
├── reference/
│   ├── ticker_metadata.parquet
│   └── index_constituents.parquet
```

**Key Design Principles:**

1. **Hierarchical partitioning** - Organize by symbol → year → month → day
2. **Predictable paths** - Easy to construct object keys programmatically
3. **Format optimization** - JSON for small files, Parquet for large datasets
4. **Separation of concerns** - Different data types in different prefixes

---

## 2. Setting Up Your S3 Bucket

### Step 1: Create Your Data Lake Bucket

```python
import logging
import boto3
from botocore.exceptions import ClientError

def create_data_lake_bucket(bucket_name, region='us-east-2'):
    """
    Create an S3 bucket for data lake storage.

    Args:
        bucket_name: Unique bucket name (must be globally unique)
        region: AWS region (default: us-east-2)

    Returns:
        True if bucket created successfully, False otherwise
    """
    try:
        # Create S3 client in the target region
        s3_client = boto3.client('s3', region_name=region)

        # Configure bucket creation
        bucket_config = {}
        if region != 'us-east-1':
            # LocationConstraint required for all regions except us-east-1
            bucket_config = {
                'CreateBucketConfiguration': {
                    'LocationConstraint': region
                }
            }

        # Create the bucket
        s3_client.create_bucket(
            Bucket=bucket_name,
            **bucket_config
        )

        logging.info(f"Successfully created bucket: {bucket_name} in {region}")
        return True

    except ClientError as e:
        error_code = e.response['Error']['Code']

        if error_code == 'BucketAlreadyOwnedByYou':
            logging.info(f"Bucket {bucket_name} already exists and is owned by you")
            return True
        elif error_code == 'BucketAlreadyExists':
            logging.error(f"Bucket {bucket_name} already exists globally (owned by someone else)")
        else:
            logging.error(f"Error creating bucket: {e}")

        return False

# Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create bucket for US equity data lake
    create_data_lake_bucket('us-equity-datalake-YOUR-UNIQUE-ID', region='us-east-2')
```

**Important Notes:**

- **Bucket names must be globally unique** across all AWS accounts
- **Bucket names are permanent** - choose carefully
- **Use DNS-compliant names** - lowercase letters, numbers, hyphens only
- **Consider naming convention** - `{project}-{environment}-{purpose}-{unique-id}`

### Step 2: Verify Bucket Creation

```python
def list_buckets():
    """List all S3 buckets in your AWS account."""
    s3_client = boto3.client('s3')

    try:
        response = s3_client.list_buckets()

        print('Available S3 buckets:')
        for bucket in response['Buckets']:
            print(f"  - {bucket['Name']} (created: {bucket['CreationDate']})")

    except ClientError as e:
        logging.error(f"Error listing buckets: {e}")

# Usage
list_buckets()
```

### Step 3: Configure Bucket Lifecycle Policies (Optional)

For cost optimization, configure lifecycle policies to transition older data to cheaper storage classes:

```python
def configure_lifecycle_policy(bucket_name):
    """
    Configure S3 lifecycle policy for cost optimization.

    Transitions:
    - After 30 days: Standard → Standard-IA (Infrequent Access)
    - After 90 days: Standard-IA → Glacier
    - After 365 days: Glacier → Deep Archive
    """
    s3_client = boto3.client('s3')

    lifecycle_policy = {
        'Rules': [
            {
                'Id': 'archive-old-data',
                'Status': 'Enabled',
                'Prefix': '',  # Apply to all objects
                'Transitions': [
                    {
                        'Days': 30,
                        'StorageClass': 'STANDARD_IA'
                    },
                    {
                        'Days': 90,
                        'StorageClass': 'GLACIER'
                    },
                    {
                        'Days': 365,
                        'StorageClass': 'DEEP_ARCHIVE'
                    }
                ]
            }
        ]
    }

    try:
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
        logging.info(f"Lifecycle policy configured for {bucket_name}")
    except ClientError as e:
        logging.error(f"Error configuring lifecycle policy: {e}")

# Usage (optional - only if you want automatic archival)
# configure_lifecycle_policy('us-equity-datalake-YOUR-UNIQUE-ID')
```

---

## 3. Uploading Data to Your Data Lake

### Basic Upload: Single File

The simplest way to upload a file to S3:

```python
import boto3
from botocore.exceptions import ClientError
import logging

def upload_file_basic(local_file_path, bucket_name, s3_object_key):
    """
    Upload a single file to S3.

    Args:
        local_file_path: Path to local file (e.g., 'data/AAPL_2024.json')
        bucket_name: S3 bucket name
        s3_object_key: Destination path in S3 (e.g., 'ticks/daily/AAPL/2024/ticks.json')

    Returns:
        True if successful, False otherwise
    """
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_object_key)
        logging.info(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_object_key}")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False
    except FileNotFoundError:
        logging.error(f"File not found: {local_file_path}")
        return False

# Usage
upload_file_basic(
    local_file_path='data/ticks/AAPL_2024.json',
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/ticks.json'
)
```

### Upload with Metadata

Attach custom metadata to track data lineage and properties:

```python
def upload_with_metadata(local_file_path, bucket_name, s3_object_key, metadata=None):
    """
    Upload file with custom metadata.

    Example metadata:
    {
        'source': 'yfinance',
        'collection-date': '2024-01-15',
        'symbol': 'AAPL',
        'data-type': 'daily-ticks'
    }
    """
    s3_client = boto3.client('s3')

    # Default metadata if none provided
    if metadata is None:
        metadata = {}

    try:
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_object_key,
            ExtraArgs={
                'Metadata': metadata,
                'ContentType': 'application/json'  # Set appropriate content type
            }
        )
        logging.info(f"Uploaded with metadata: {s3_object_key}")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False

# Usage
upload_with_metadata(
    local_file_path='data/ticks/AAPL_2024.json',
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/ticks.json',
    metadata={
        'source': 'yfinance',
        'collection-date': '2024-01-15',
        'symbol': 'AAPL',
        'data-type': 'daily-ticks',
        'record-count': '252'
    }
)
```

### Upload with Progress Tracking

For large files, track upload progress:

```python
import os
import sys
import threading

class ProgressPercentage:
    """
    Progress callback for S3 uploads/downloads.

    Displays progress as: filename  123 MB / 456 MB  (27.00%)
    """

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # Called by boto3 during transfer
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100

            sys.stdout.write(
                f"\r{self._filename}  {self._seen_so_far / 1024 / 1024:.2f} MB / "
                f"{self._size / 1024 / 1024:.2f} MB  ({percentage:.2f}%)"
            )
            sys.stdout.flush()

def upload_with_progress(local_file_path, bucket_name, s3_object_key):
    """Upload file with progress bar."""
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_object_key,
            Callback=ProgressPercentage(local_file_path)
        )
        print()  # New line after progress
        logging.info(f"Upload complete: {s3_object_key}")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False

# Usage
upload_with_progress(
    local_file_path='data/large_dataset.parquet',
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/minute/AAPL/2024/01/data.parquet'
)
```

### Upload from File-Like Object

Upload data directly from memory without saving to disk:

```python
import io
import json

def upload_from_memory(data, bucket_name, s3_object_key):
    """
    Upload data from memory (without creating local file).

    Useful for:
    - Small datasets that fit in memory
    - Transformed data (avoid intermediate file creation)
    - Streaming data processing
    """
    s3_client = boto3.client('s3')

    # Convert data to bytes
    if isinstance(data, dict) or isinstance(data, list):
        # JSON data
        data_bytes = json.dumps(data).encode('utf-8')
    elif isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data

    # Create file-like object
    file_obj = io.BytesIO(data_bytes)

    try:
        s3_client.upload_fileobj(file_obj, bucket_name, s3_object_key)
        logging.info(f"Uploaded from memory: {s3_object_key}")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False

# Usage
data = {
    'symbol': 'AAPL',
    'date': '2024-01-15',
    'close': 185.92,
    'volume': 54213000
}

upload_from_memory(
    data=data,
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/01-15.json'
)
```

---

## 4. Downloading Data from Your Data Lake

### Basic Download: Single File

Download a file from S3 to local disk:

```python
def download_file_basic(bucket_name, s3_object_key, local_file_path):
    """
    Download a single file from S3.

    Args:
        bucket_name: S3 bucket name
        s3_object_key: Source path in S3
        local_file_path: Destination path on local disk

    Returns:
        True if successful, False otherwise
    """
    s3_client = boto3.client('s3')

    try:
        s3_client.download_file(bucket_name, s3_object_key, local_file_path)
        logging.info(f"Downloaded s3://{bucket_name}/{s3_object_key} to {local_file_path}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logging.error(f"Object not found: {s3_object_key}")
        else:
            logging.error(f"Download failed: {e}")
        return False

# Usage
download_file_basic(
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/ticks.json',
    local_file_path='data/downloads/AAPL_2024.json'
)
```

### Download with Progress Tracking

```python
def download_with_progress(bucket_name, s3_object_key, local_file_path):
    """Download file with progress bar."""
    s3_client = boto3.client('s3')

    # Get file size first
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_object_key)
        file_size = response['ContentLength']

        # Create custom progress callback
        class DownloadProgress:
            def __init__(self, filename, size):
                self._filename = filename
                self._size = float(size)
                self._seen_so_far = 0
                self._lock = threading.Lock()

            def __call__(self, bytes_amount):
                with self._lock:
                    self._seen_so_far += bytes_amount
                    percentage = (self._seen_so_far / self._size) * 100
                    sys.stdout.write(
                        f"\r{self._filename}  {self._seen_so_far / 1024 / 1024:.2f} MB / "
                        f"{self._size / 1024 / 1024:.2f} MB  ({percentage:.2f}%)"
                    )
                    sys.stdout.flush()

        # Download with progress
        s3_client.download_file(
            bucket_name,
            s3_object_key,
            local_file_path,
            Callback=DownloadProgress(s3_object_key, file_size)
        )
        print()  # New line after progress
        logging.info(f"Download complete: {local_file_path}")
        return True

    except ClientError as e:
        logging.error(f"Download failed: {e}")
        return False

# Usage
download_with_progress(
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/minute/AAPL/2024/01/data.parquet',
    local_file_path='data/downloads/AAPL_minute_data.parquet'
)
```

### Download to Memory

Download data directly into memory without creating a file:

```python
import io
import json

def download_to_memory(bucket_name, s3_object_key):
    """
    Download file directly to memory (no disk I/O).

    Returns:
        File contents as bytes, or None if error
    """
    s3_client = boto3.client('s3')

    try:
        # Create in-memory file object
        file_obj = io.BytesIO()

        # Download to memory
        s3_client.download_fileobj(bucket_name, s3_object_key, file_obj)

        # Reset file pointer to beginning
        file_obj.seek(0)

        # Read and return bytes
        return file_obj.read()

    except ClientError as e:
        logging.error(f"Download failed: {e}")
        return None

def download_json_to_dict(bucket_name, s3_object_key):
    """Download JSON file and parse to Python dictionary."""
    data_bytes = download_to_memory(bucket_name, s3_object_key)

    if data_bytes:
        return json.loads(data_bytes.decode('utf-8'))
    return None

# Usage
data = download_json_to_dict(
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/ticks.json'
)

if data:
    print(f"Downloaded {len(data)} records")
```

### Check if Object Exists Before Downloading

```python
def object_exists(bucket_name, s3_object_key):
    """
    Check if an S3 object exists.

    Returns:
        True if exists, False otherwise
    """
    s3_client = boto3.client('s3')

    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_object_key)
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False
        else:
            # Other error (permissions, etc.)
            logging.error(f"Error checking object existence: {e}")
            return False

def safe_download(bucket_name, s3_object_key, local_file_path):
    """Download file only if it exists."""
    if object_exists(bucket_name, s3_object_key):
        return download_file_basic(bucket_name, s3_object_key, local_file_path)
    else:
        logging.warning(f"Object does not exist: {s3_object_key}")
        return False

# Usage
safe_download(
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/ticks.json',
    local_file_path='data/AAPL_2024.json'
)
```

---

## 5. Optimizing File Transfers

### Understanding TransferConfig

The `TransferConfig` object controls how boto3 handles file transfers. This is critical for data lake operations involving millions of files.

```python
from boto3.s3.transfer import TransferConfig

# Default configuration (what boto3 uses if you don't specify)
default_config = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,  # 8 MB
    max_concurrency=10,
    multipart_chunksize=8 * 1024 * 1024,  # 8 MB
    num_download_attempts=5,
    max_io_queue=100,
    io_chunksize=262144,  # 256 KB
    use_threads=True
)
```

### Multipart Transfers

For large files, boto3 automatically splits them into chunks and uploads in parallel:

```python
def upload_large_file(local_file_path, bucket_name, s3_object_key):
    """
    Upload large file with optimized multipart settings.

    Recommended for files > 100 MB
    """
    s3_client = boto3.client('s3')

    # Configure for large file uploads
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # Files > 100 MB use multipart
        multipart_chunksize=10 * 1024 * 1024,   # 10 MB chunks
        max_concurrency=20,                      # 20 parallel uploads
        use_threads=True
    )

    try:
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_object_key,
            Config=config,
            Callback=ProgressPercentage(local_file_path)
        )
        print()
        logging.info(f"Large file uploaded: {s3_object_key}")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False

# Usage
upload_large_file(
    local_file_path='data/minute_ticks_2024.parquet',  # 500 MB file
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/minute/AAPL/2024/full_year.parquet'
)
```

**What Happens:**
1. File is split into 10 MB chunks
2. 20 chunks upload simultaneously
3. S3 automatically assembles the chunks
4. Much faster than single-threaded upload

### Adjusting Concurrency for Network Speed

```python
def upload_optimized_for_network(local_file_path, bucket_name, s3_object_key, network_speed='fast'):
    """
    Upload with network-appropriate concurrency.

    Args:
        network_speed: 'slow', 'medium', 'fast', or 'very_fast'
    """
    s3_client = boto3.client('s3')

    # Network-appropriate configurations
    configs = {
        'slow': TransferConfig(
            max_concurrency=5,
            multipart_threshold=50 * 1024 * 1024  # 50 MB
        ),
        'medium': TransferConfig(
            max_concurrency=10,
            multipart_threshold=20 * 1024 * 1024  # 20 MB
        ),
        'fast': TransferConfig(
            max_concurrency=20,
            multipart_threshold=10 * 1024 * 1024  # 10 MB
        ),
        'very_fast': TransferConfig(
            max_concurrency=50,
            multipart_threshold=5 * 1024 * 1024   # 5 MB
        )
    }

    config = configs.get(network_speed, configs['medium'])

    try:
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_object_key,
            Config=config
        )
        logging.info(f"Uploaded with {network_speed} network config")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False
```

### Disabling Threads for Debugging

Sometimes threading complicates debugging. Disable for troubleshooting:

```python
def upload_single_threaded(local_file_path, bucket_name, s3_object_key):
    """Upload without threading (easier to debug)."""
    s3_client = boto3.client('s3')

    config = TransferConfig(
        use_threads=False  # Disable threading
    )

    try:
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_object_key,
            Config=config
        )
        logging.info(f"Single-threaded upload complete")
        return True
    except ClientError as e:
        logging.error(f"Upload failed: {e}")
        return False
```

---

## 6. Batch Operations and Parallelization

### Uploading Multiple Files

For data lake backfills, you need to upload thousands of files efficiently:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def upload_directory(local_dir, bucket_name, s3_prefix, max_workers=10):
    """
    Upload entire directory to S3 in parallel.

    Args:
        local_dir: Local directory path
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix (folder) to upload to
        max_workers: Number of parallel uploads

    Returns:
        Dictionary with success/failure counts
    """
    s3_client = boto3.client('s3')
    local_dir = Path(local_dir)

    # Find all files to upload
    files_to_upload = []
    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            # Calculate S3 object key
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
            files_to_upload.append((str(file_path), s3_key))

    print(f"Found {len(files_to_upload)} files to upload")

    # Upload in parallel
    results = {'success': 0, 'failed': 0}

    def upload_file(file_info):
        local_path, s3_key = file_info
        try:
            s3_client.upload_file(local_path, bucket_name, s3_key)
            return True, s3_key
        except Exception as e:
            logging.error(f"Failed to upload {local_path}: {e}")
            return False, s3_key

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        futures = {executor.submit(upload_file, f): f for f in files_to_upload}

        # Process results as they complete
        for future in as_completed(futures):
            success, s3_key = future.result()
            if success:
                results['success'] += 1
                print(f"✓ Uploaded {s3_key}")
            else:
                results['failed'] += 1
                print(f"✗ Failed {s3_key}")

    print(f"\nUpload complete: {results['success']} succeeded, {results['failed']} failed")
    return results

# Usage
upload_directory(
    local_dir='data/ticks/daily',
    bucket_name='us-equity-datalake',
    s3_prefix='ticks/daily',
    max_workers=20  # Upload 20 files simultaneously
)
```

### Downloading Multiple Files

```python
def download_by_prefix(bucket_name, s3_prefix, local_dir, max_workers=10):
    """
    Download all objects with a given prefix (folder).

    Args:
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix to download from
        local_dir: Local directory to download to
        max_workers: Number of parallel downloads

    Returns:
        Number of files downloaded
    """
    s3_client = boto3.client('s3')
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # List all objects with prefix
    objects_to_download = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                objects_to_download.append(obj['Key'])

    print(f"Found {len(objects_to_download)} objects to download")

    # Download in parallel
    def download_object(s3_key):
        # Calculate local file path
        relative_path = s3_key.replace(s3_prefix, '').lstrip('/')
        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3_client.download_file(bucket_name, s3_key, str(local_path))
            return True, s3_key
        except Exception as e:
            logging.error(f"Failed to download {s3_key}: {e}")
            return False, s3_key

    downloaded_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_object, key): key for key in objects_to_download}

        for future in as_completed(futures):
            success, s3_key = future.result()
            if success:
                downloaded_count += 1
                print(f"✓ Downloaded {s3_key}")

    print(f"\nDownload complete: {downloaded_count} files")
    return downloaded_count

# Usage
download_by_prefix(
    bucket_name='us-equity-datalake',
    s3_prefix='ticks/daily/AAPL',
    local_dir='data/downloads/AAPL',
    max_workers=20
)
```

### Listing Objects Efficiently

```python
def list_objects_by_prefix(bucket_name, prefix='', max_keys=1000):
    """
    List S3 objects with pagination support.

    Args:
        bucket_name: S3 bucket name
        prefix: Filter by prefix
        max_keys: Maximum keys to return (None for all)

    Returns:
        List of object keys
    """
    s3_client = boto3.client('s3')

    objects = []
    continuation_token = None

    while True:
        # Build request parameters
        params = {
            'Bucket': bucket_name,
            'Prefix': prefix
        }

        if continuation_token:
            params['ContinuationToken'] = continuation_token

        # List objects
        response = s3_client.list_objects_v2(**params)

        # Add objects from this page
        if 'Contents' in response:
            for obj in response['Contents']:
                objects.append({
                    'Key': obj['Key'],
                    'Size': obj['Size'],
                    'LastModified': obj['LastModified']
                })

                # Stop if we hit max_keys
                if max_keys and len(objects) >= max_keys:
                    return objects

        # Check if there are more pages
        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    return objects

# Usage
# List all AAPL daily tick files
aapl_files = list_objects_by_prefix(
    bucket_name='us-equity-datalake',
    prefix='ticks/daily/AAPL/'
)

print(f"Found {len(aapl_files)} files for AAPL")
for file in aapl_files[:5]:  # Show first 5
    print(f"  {file['Key']} - {file['Size'] / 1024:.2f} KB")
```

---

## 7. Managing Access with Presigned URLs

Presigned URLs allow temporary access to S3 objects without AWS credentials. Useful for:

- **Sharing data** with collaborators
- **Web applications** that need direct S3 access
- **Temporary access** for external systems

### Generate Presigned URL for Download

```python
def generate_download_url(bucket_name, s3_object_key, expiration=3600):
    """
    Generate presigned URL for downloading an object.

    Args:
        bucket_name: S3 bucket name
        s3_object_key: S3 object key
        expiration: URL validity in seconds (default: 1 hour)

    Returns:
        Presigned URL string, or None if error
    """
    s3_client = boto3.client('s3')

    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': s3_object_key
            },
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        logging.error(f"Error generating presigned URL: {e}")
        return None

# Usage
url = generate_download_url(
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/ticks.json',
    expiration=3600  # Valid for 1 hour
)

if url:
    print(f"Share this URL: {url}")
    print("Anyone with this URL can download the file for the next hour")
```

### Using Presigned URLs

```python
import requests

def download_with_presigned_url(presigned_url, local_file_path):
    """Download file using presigned URL (no AWS credentials needed)."""
    try:
        response = requests.get(presigned_url)
        response.raise_for_status()  # Raise error for bad status codes

        with open(local_file_path, 'wb') as f:
            f.write(response.content)

        logging.info(f"Downloaded to {local_file_path}")
        return True
    except requests.RequestException as e:
        logging.error(f"Download failed: {e}")
        return False

# Usage (this can run on ANY machine, no AWS credentials required)
url = "https://us-equity-datalake.s3.us-east-2.amazonaws.com/ticks/daily/AAPL/2024/ticks.json?..."
download_with_presigned_url(url, 'AAPL_data.json')
```

### Generate Presigned URL for Upload

Allow external users to upload files to your data lake:

```python
def generate_upload_url(bucket_name, s3_object_key, expiration=3600):
    """
    Generate presigned URL for uploading an object.

    Returns:
        Dictionary with 'url' and 'fields' for POST request
    """
    s3_client = boto3.client('s3')

    try:
        response = s3_client.generate_presigned_post(
            Bucket=bucket_name,
            Key=s3_object_key,
            ExpiresIn=expiration
        )
        return response
    except ClientError as e:
        logging.error(f"Error generating upload URL: {e}")
        return None

# Usage
upload_url_data = generate_upload_url(
    bucket_name='us-equity-datalake',
    s3_object_key='uploads/user_data.csv',
    expiration=7200  # Valid for 2 hours
)

if upload_url_data:
    print(f"Upload URL: {upload_url_data['url']}")
    print(f"Fields: {upload_url_data['fields']}")
```

```python
def upload_with_presigned_url(presigned_post_data, local_file_path):
    """Upload file using presigned POST URL."""
    try:
        with open(local_file_path, 'rb') as f:
            files = {'file': (local_file_path, f)}
            response = requests.post(
                presigned_post_data['url'],
                data=presigned_post_data['fields'],
                files=files
            )

        response.raise_for_status()
        logging.info(f"Upload successful")
        return True
    except requests.RequestException as e:
        logging.error(f"Upload failed: {e}")
        return False
```

---

## 8. Production Patterns and Best Practices

### Error Handling and Retries

```python
from botocore.exceptions import ClientError
import time

def upload_with_retry(local_file_path, bucket_name, s3_object_key, max_retries=3):
    """
    Upload with automatic retry on failure.

    Implements exponential backoff for transient errors.
    """
    s3_client = boto3.client('s3')

    for attempt in range(max_retries):
        try:
            s3_client.upload_file(local_file_path, bucket_name, s3_object_key)
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
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logging.warning(f"Upload failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Upload failed after {max_retries} attempts")
                return False

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return False

    return False

# Usage
upload_with_retry(
    local_file_path='data/important_data.json',
    bucket_name='us-equity-datalake',
    s3_object_key='critical/data.json',
    max_retries=5
)
```

### Atomic Uploads (Avoid Partial Files)

```python
import tempfile
import shutil

def atomic_upload(data_generator_func, bucket_name, s3_object_key):
    """
    Upload file atomically - either complete upload or nothing.

    Prevents partial/corrupted files in S3.

    Args:
        data_generator_func: Function that generates/writes data to a file path
        bucket_name: S3 bucket name
        s3_object_key: S3 object key
    """
    s3_client = boto3.client('s3')

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tmp') as tmp_file:
        tmp_path = tmp_file.name

        try:
            # Generate data to temporary file
            data_generator_func(tmp_file)
            tmp_file.flush()

            # Upload only if data generation succeeded
            s3_client.upload_file(tmp_path, bucket_name, s3_object_key)
            logging.info(f"Atomic upload complete: {s3_object_key}")
            return True

        except Exception as e:
            logging.error(f"Atomic upload failed: {e}")
            return False

        finally:
            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
            except:
                pass

# Usage
def generate_daily_ticks(file_obj):
    """Example data generator function."""
    import json
    data = {
        'symbol': 'AAPL',
        'ticks': [...]  # Generate tick data
    }
    json.dump(data, file_obj)

atomic_upload(
    data_generator_func=generate_daily_ticks,
    bucket_name='us-equity-datalake',
    s3_object_key='ticks/daily/AAPL/2024/ticks.json'
)
```

### Logging and Monitoring

```python
import logging
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_lake_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

def upload_with_monitoring(local_file_path, bucket_name, s3_object_key):
    """Upload with detailed logging and metrics."""
    import time

    logger = logging.getLogger(__name__)
    s3_client = boto3.client('s3')

    # Log upload start
    file_size = Path(local_file_path).stat().st_size
    logger.info(f"Starting upload: {local_file_path} ({file_size / 1024 / 1024:.2f} MB)")

    start_time = time.time()

    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_object_key)

        # Log success with metrics
        duration = time.time() - start_time
        speed_mbps = (file_size / 1024 / 1024) / duration if duration > 0 else 0

        logger.info(
            f"Upload complete: {s3_object_key} | "
            f"Duration: {duration:.2f}s | "
            f"Speed: {speed_mbps:.2f} MB/s"
        )
        return True

    except ClientError as e:
        logger.error(
            f"Upload failed: {s3_object_key} | "
            f"Error: {e.response['Error']['Code']} | "
            f"Message: {e.response['Error']['Message']}"
        )
        return False
```

### Data Validation

```python
import hashlib

def upload_with_validation(local_file_path, bucket_name, s3_object_key):
    """
    Upload file and validate integrity with MD5 checksum.
    """
    s3_client = boto3.client('s3')

    # Calculate local file MD5
    def calculate_md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    local_md5 = calculate_md5(local_file_path)
    logging.info(f"Local file MD5: {local_md5}")

    try:
        # Upload file
        s3_client.upload_file(local_file_path, bucket_name, s3_object_key)

        # Verify upload with ETag (S3's MD5)
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_object_key)
        s3_etag = response['ETag'].strip('"')

        if local_md5 == s3_etag:
            logging.info(f"Upload verified: checksums match")
            return True
        else:
            logging.error(f"Upload corrupted: checksum mismatch")
            # Delete corrupted file
            s3_client.delete_object(Bucket=bucket_name, Key=s3_object_key)
            return False

    except ClientError as e:
        logging.error(f"Upload/validation failed: {e}")
        return False
```

---

## 9. Complete Implementation Example

Here's a production-ready data lake uploader for the US Equity Data Lake:

```python
"""
US Equity Data Lake Uploader

Production-ready S3 uploader with:
- Parallel uploads
- Progress tracking
- Error handling and retries
- Logging and monitoring
- Data validation
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


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
    - Parallel uploads with thread pool
    - Automatic retry with exponential backoff
    - Progress tracking and logging
    - Upload validation
    """

    def __init__(
        self,
        bucket_name: str,
        region: str = 'us-east-2',
        max_workers: int = 20,
        max_retries: int = 5
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
        self.region = region
        self.max_workers = max_workers
        self.max_retries = max_retries

        # Configure S3 client
        self.s3_config = Config(
            region_name=region,
            max_pool_connections=max_workers,
            retries={'mode': 'standard', 'total_max_attempts': max_retries},
            tcp_keepalive=True
        )
        self.s3_client = boto3.client('s3', config=self.s3_config)

        # Configure transfer settings
        self.transfer_config = TransferConfig(
            multipart_threshold=10 * 1024 * 1024,  # 10 MB
            max_concurrency=20,
            multipart_chunksize=10 * 1024 * 1024,
            use_threads=True
        )

        # Setup logging
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
                # Prepare extra args
                extra_args = {}
                if metadata:
                    extra_args['Metadata'] = metadata

                # Upload file
                self.s3_client.upload_file(
                    local_path,
                    self.bucket_name,
                    s3_key,
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

                # Don't retry permanent errors
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
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        f"Upload failed for {s3_key} (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
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
        file_pattern: str = '*'
    ) -> Dict[str, any]:
        """
        Upload entire directory in parallel.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix (folder)
            file_pattern: Glob pattern for files (default: all files)

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
            # Submit all tasks
            futures = {
                executor.submit(self.upload_file, local_path, s3_key): (local_path, s3_key)
                for local_path, s3_key in files_to_upload
            }

            # Process results
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Progress update
                completed = len(results)
                percent = (completed / total_files) * 100
                print(f"\rProgress: {completed}/{total_files} ({percent:.1f}%)", end='')

        print()  # New line after progress

        # Calculate statistics
        total_duration = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = total_files - successful

        stats = {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'total_duration': total_duration,
            'average_duration': total_duration / total_files if total_files > 0 else 0
        }

        self.logger.info(
            f"Upload complete: {successful}/{total_files} succeeded "
            f"in {total_duration:.2f}s"
        )

        return stats


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'upload_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    # Create uploader
    uploader = DataLakeUploader(
        bucket_name='us-equity-datalake',
        region='us-east-2',
        max_workers=20
    )

    # Upload entire directory
    stats = uploader.upload_directory(
        local_dir='data/ticks/daily',
        s3_prefix='ticks/daily',
        file_pattern='*.json'
    )

    print(f"\nUpload Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Duration: {stats['total_duration']:.2f}s")
    print(f"  Average per file: {stats['average_duration']:.2f}s")
```

---

## Conclusion

You now have a comprehensive understanding of building a data lake with AWS S3 and Boto3. This tutorial covered:

✅ **Bucket Management** - Creating and organizing S3 buckets
✅ **File Operations** - Uploading and downloading with various methods
✅ **Transfer Optimization** - Multipart uploads, concurrency, threading
✅ **Batch Operations** - Parallel uploads/downloads for large datasets
✅ **Security** - Presigned URLs for temporary access
✅ **Production Patterns** - Error handling, retries, logging, validation

### Next Steps

1. **Review [S3_CONFIGURATION.md](S3_CONFIGURATION.md)** for detailed configuration explanations
2. **Implement the DataLakeUploader** class in your project
3. **Test with small datasets** before running full backfills
4. **Monitor costs** using AWS Cost Explorer
5. **Set up CloudWatch alarms** for monitoring

### Additional Resources

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Boto3 S3 Reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)
- [Data Lake Architecture on AWS](https://aws.amazon.com/big-data/datalakes-and-analytics/)

---

**Author**: US Equity Data Lake Project
**Last Updated**: 2025-01-20
**Version**: 1.0
